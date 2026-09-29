"""Query building for the evidence search: the sentence, the judge, and the fallback."""

from __future__ import annotations

import json
from typing import Any

import pytest

from proofpath.judge import Completion, Judge, JudgeCost, JudgeUnavailable
from proofpath.search.language import looks_english
from proofpath.search.queries import (
    LLM_LIMIT,
    LLM_UNAVAILABLE,
    QUERY_LIMIT,
    SENTENCE_BY,
    plan_queries,
    sentence_query,
)


class ScriptedClient:
    """``JudgeClient`` without a socket: one scripted answer or exception."""

    provider = "fake"
    model = "q-1"

    def __init__(self, answer: str | Exception) -> None:
        self.answer = answer
        self.cost = JudgeCost(model=self.model)
        self.prompts: list[str] = []

    def complete(self, messages: list[dict[str, str]], **_: Any) -> Completion:
        self.prompts.append(messages[-1]["content"])
        if isinstance(self.answer, Exception):
            raise self.answer
        return Completion(text=self.answer, prompt_tokens=1, completion_tokens=1, model=self.model)

    def close(self) -> None:
        pass


def judge(answer: str | Exception) -> Judge:
    return Judge(ScriptedClient(answer))  # type: ignore[arg-type]


def test_sentence_query_strips_links_handles_hashtag_signs_and_emoji() -> None:
    text = "🚨 @newsbot says #ChatGPT was shut down https://t.co/xyz today!"
    assert sentence_query(text) == "says ChatGPT was shut down today!"


def test_sentence_query_cuts_at_a_word_boundary() -> None:
    query = sentence_query("word " * 200)
    assert len(query) <= QUERY_LIMIT
    assert not query.endswith(" ")


def test_without_a_judge_the_sentence_is_the_query() -> None:
    plan = plan_queries(["ChatGPT was shut down in 2025."], None)
    assert plan.queries == (("ChatGPT was shut down in 2025.",),)
    assert plan.by == SENTENCE_BY
    assert plan.notice is None


def test_the_judge_writes_decontextualised_queries() -> None:
    answer = json.dumps(
        {
            "items": [
                {
                    "id": 0,
                    "english": "He shut it down in 2025.",
                    "queries": ["OpenAI ChatGPT shutdown 2025"],
                }
            ]
        }
    )
    plan = plan_queries(["He shut it down in 2025."], judge(answer))
    assert plan.queries == (("OpenAI ChatGPT shutdown 2025",),)
    assert plan.by == "fake q-1"
    assert plan.notice is None


def test_a_rate_limited_judge_falls_back_to_the_sentence_and_says_so() -> None:
    plan = plan_queries(
        ["ChatGPT was shut down in 2025."],
        judge(JudgeUnavailable("HTTP 429 from https://api.test/v1/chat/completions", status=429)),
    )
    assert plan.queries == (("ChatGPT was shut down in 2025.",),)
    assert plan.by == SENTENCE_BY
    assert plan.notice is not None and plan.notice.startswith(LLM_LIMIT)


def test_a_judge_that_is_down_falls_back_with_its_own_words() -> None:
    plan = plan_queries(["ChatGPT was shut down in 2025."], judge(JudgeUnavailable("HTTP 503")))
    assert plan.notice is not None and plan.notice.startswith(LLM_UNAVAILABLE)


def test_a_claim_the_judge_skipped_keeps_its_sentence() -> None:
    answer = json.dumps(
        {
            "items": [
                {"id": 1, "english": "Paris is the capital.", "queries": ["Paris capital France"]}
            ]
        }
    )
    plan = plan_queries(["NASA landed in 1969.", "Paris is the capital."], judge(answer))
    assert plan.queries == (("NASA landed in 1969.",), ("Paris capital France",))


def test_judge_queries_never_raises_on_an_unreadable_answer() -> None:
    built = judge("not json at all")
    assert built.queries(["a claim"]) == {}
    assert built.unavailable is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("OpenAI shut down ChatGPT in 2025.", True),
        ("NASA confirms water on Mars", True),
        ("OpenAI battı.", False),  # noqa: RUF001 - real Turkish letters
        # ASCII cases carry two function words: one is no longer evidence (final review).
        ("OpenAI batti ve bu kapandi.", False),  # ASCII Turkish, "ve" and "bu"
        ("El gobierno cerró la empresa por una deuda.", False),  # "por", "una"
        ("Die Firma ist nicht pleite.", False),  # "ist", "nicht"
        ("André said the deal is done.", True),
        ("Beyoncé released a new album.", True),
        ("Müller won the race in Berlin.", True),
        ("Путин подписал закон.", False),  # Cyrillic
        ("İstanbul büyük bir şehir.", False),  # Turkish-only letters, and "bir" too
    ],
)
def test_looks_english(text: str, expected: bool) -> None:
    assert looks_english(text) is expected


def test_an_english_claim_is_checked_as_written() -> None:
    plan = plan_queries(["OpenAI shut down ChatGPT in 2025."], None)
    assert plan.hypotheses == ("OpenAI shut down ChatGPT in 2025.",)


def test_without_a_judge_a_turkish_claim_has_nothing_to_check() -> None:
    plan = plan_queries(["OpenAI battı."], None)  # noqa: RUF001 - real Turkish letters
    assert plan.hypotheses == (None,)


def test_the_judge_translates_a_turkish_claim_for_the_check_and_the_queries() -> None:
    answer = json.dumps(
        {"items": [{"id": 0, "english": "OpenAI went bankrupt.", "queries": ["OpenAI bankruptcy"]}]}
    )
    plan = plan_queries(["OpenAI battı."], judge(answer))  # noqa: RUF001 - real Turkish letters
    assert plan.hypotheses == ("OpenAI went bankrupt.",)
    assert plan.queries == (("OpenAI bankruptcy",),)


def test_a_rate_limited_judge_leaves_a_turkish_claim_untranslated() -> None:
    plan = plan_queries(
        ["OpenAI battı."],  # noqa: RUF001 - real Turkish letters
        judge(JudgeUnavailable("HTTP 429 from https://api.test", status=429)),
    )
    assert plan.hypotheses == (None,)
    assert plan.notice is not None and plan.notice.startswith(LLM_LIMIT)


def test_a_judge_that_translates_but_writes_no_query_keeps_the_sentence_query() -> None:
    answer = json.dumps({"items": [{"id": 0, "english": "OpenAI went bankrupt.", "queries": []}]})
    plan = plan_queries(["OpenAI battı."], judge(answer))  # noqa: RUF001 - real Turkish letters
    assert plan.hypotheses == ("OpenAI went bankrupt.",)
    assert plan.queries == ((sentence_query("OpenAI battı."),),)  # noqa: RUF001 - real Turkish letters


@pytest.mark.parametrize(
    "english",
    [
        "OpenAI battı.",  # noqa: RUF001 - the claim echoed back, real Turkish letters
        "OpenAI iflas etti ve kapandı.",  # noqa: RUF001 - still Turkish, just other words
    ],
)
def test_an_english_that_is_not_english_is_no_translation(english: str) -> None:
    """Fix round 1 A: a judge that echoes the claim (a local qwen does) must not have
    Turkish checked by the English-only models. Its queries are still kept."""
    answer = json.dumps(
        {"items": [{"id": 0, "english": english, "queries": ["OpenAI bankruptcy"]}]}
    )
    plan = plan_queries(["OpenAI battı."], judge(answer))  # noqa: RUF001 - real Turkish letters
    assert plan.hypotheses == (None,)
    assert plan.queries == (("OpenAI bankruptcy",),)


def test_an_unchanged_echo_of_a_non_english_claim_is_no_translation() -> None:
    """Any language, not only Turkish: a French claim echoed back is still French."""
    claim = "Le chat est dans une maison."  # "dans", "une": two function words
    assert not looks_english(claim)
    answer = json.dumps({"items": [{"id": 0, "english": claim, "queries": ["black cat"]}]})
    assert plan_queries([claim], judge(answer)).hypotheses == (None,)


def test_the_queries_prompt_carries_worked_examples_that_cannot_be_answered() -> None:
    """Fix round 1 (controller, 2026-09-28): a lone non-English claim was echoed by
    qwen3.5:9b, so the prompt shows one translation that changes the verb and one
    English claim kept as is. The examples use generic subjects and ids that are not
    integers, so an echoed example is dropped by the reader, never taken for a claim."""
    from proofpath.judge import _queries_from, _split_prompt, load_prompt

    system, user = _split_prompt(load_prompt("queries").substitute(items="- id: 0\n  claim: x"))
    assert "Examples" in user
    assert "The company went bankrupt." in user
    assert "The Eiffel Tower is in Paris." in user
    assert "OpenAI" not in system + user
    # Only the real item uses the ``- id:`` shape the fakes and the batch counter read.
    assert user.count("id: ") == 1
    example = (
        '{"items": [{"id": "E1", "english": "The company went bankrupt.",'
        ' "queries": ["company went bankrupt"]}]}'
    )
    assert _queries_from(example, 1) == {}


@pytest.mark.parametrize(
    "text",
    [
        # Final review, Important 2: short foreign function words that English text
        # and names also print. Every one of these used to be called "not English".
        "MIT researchers found a new battery design.",
        "Smith et al. reported the effect in 2020.",
        "Los Angeles has the largest port in the country.",
        "El Niño raised ocean temperatures in 2023.",
        "The AMA endorsed the policy last year.",
        "Y2K cost companies billions of dollars.",
        "The launch is at 5pm EST today.",
        "Le Monde reported the story first.",
        "Guillermo del Toro and Leonardo da Vinci were both named.",
        "Uğur Şahin founded BioNTech in 2008.",  # a name keeps its Turkish letters
        "The μ-opioid receptor binds morphine.",  # a Greek symbol is not Greek text
    ],
)
def test_english_text_with_names_acronyms_and_borrowed_words_is_english(text: str) -> None:
    assert looks_english(text)


def test_one_foreign_function_word_is_not_evidence_but_two_are() -> None:
    assert looks_english("The deal with Pfizer is que sera.")
    assert not looks_english("La empresa que cerró es grande.")  # "que", "es"


def test_an_english_echo_of_an_english_claim_is_accepted() -> None:
    from proofpath.search.queries import _translation

    claim = "OpenAI shut down ChatGPT in 2025."
    assert _translation(claim, claim) == claim


def test_an_echo_of_a_non_english_claim_is_still_refused() -> None:
    from proofpath.search.queries import _translation

    claim = "OpenAI battı."  # noqa: RUF001 - real Turkish letters
    assert _translation(claim, claim) is None


@pytest.mark.parametrize(
    "english",
    ["The company went bankrupt.", "The Eiffel Tower is in Paris.", "the eiffel tower is in paris"],
)
def test_a_copied_prompt_example_is_no_translation(english: str) -> None:
    """Ledger T11: an example answer copied into the answer would be checked as the
    claim. The example's own claim still translates to it."""
    from proofpath.search.queries import _translation

    assert _translation("OpenAI iflas etti ve bu kapandı.", english) is None  # noqa: RUF001
    assert _translation("Şirket battı.", "The company went bankrupt.") == (  # noqa: RUF001
        "The company went bankrupt."
    )


def test_the_refused_examples_are_the_ones_the_prompt_prints() -> None:
    from proofpath.judge import load_prompt
    from proofpath.search.queries import PROMPT_EXAMPLES

    prompt = load_prompt("queries").template
    for english, claim in PROMPT_EXAMPLES.items():
        assert english in prompt and claim in prompt


@pytest.mark.parametrize(
    ("error", "head"),
    [
        (JudgeUnavailable("HTTP 429 from https://api.test", status=429), LLM_LIMIT),
        # Groq answers 413 when a request busts its per-minute token budget.
        (JudgeUnavailable("HTTP 413 from https://api.test", status=413), LLM_LIMIT),
        # A detail that merely mentions 429 is not a limit: the status decides.
        (
            JudgeUnavailable(
                "HTTP 429 from https://api.test; local ollama q: HTTP 404", status=404
            ),
            LLM_UNAVAILABLE,
        ),
        (JudgeUnavailable("request failed: ConnectError"), LLM_UNAVAILABLE),
    ],
)
def test_the_limit_notice_is_decided_by_the_status_alone(
    error: JudgeUnavailable, head: str
) -> None:
    plan = plan_queries(["ChatGPT was shut down in 2025."], judge(error))
    assert plan.notice is not None and plan.notice.startswith(head)


# --- round 2: short foreign sentences, and the judge's translation ---------------------

FOREIGN_ROUND_2 = [
    "El gobierno subió los impuestos.",  # "los" + an accented lowercase word
    "Le président a démissionné hier.",  # accented lowercase words + "hier"
    "Die Regierung hat das Gesetz beschlossen.",  # "hat", "das"
    "Il governo ha approvato la legge.",  # "ha", "la"
    "Los precios de la gasolina subieron un 20%.",  # "la", "un"
]
# Read as English without a judge: every signal these carry is either capitalised
# ("Las" opens the sentence, exactly as "Los" opens "Los Angeles has…") or a word the
# rule leaves out on purpose ("de", which English prints in "de facto" and names).
# No rule over these signals can call them foreign and keep the English cases English;
# with a judge, round 2 (a) checks the judge's translation instead.
ENGLISH_WITHOUT_A_JUDGE = [
    "Las vacunas causan autismo.",
    "De regering heeft de wet aangenomen.",
    "Enflasyon yüzde 80 oldu.",  # one signal: a single accented lowercase word
]


@pytest.mark.parametrize("text", FOREIGN_ROUND_2)
def test_short_foreign_sentences_are_not_english(text: str) -> None:
    assert not looks_english(text)


@pytest.mark.parametrize("text", ENGLISH_WITHOUT_A_JUDGE)
def test_sentences_with_one_signal_read_as_english_without_a_judge(text: str) -> None:
    assert looks_english(text)


def test_borrowed_accented_words_are_one_signal_not_many() -> None:
    assert looks_english("The café serves crème brûlée every night.")


@pytest.mark.parametrize("text", ENGLISH_WITHOUT_A_JUDGE)
def test_with_a_judge_a_sentence_that_is_not_certainly_english_is_checked_translated(
    text: str,
) -> None:
    answer = json.dumps(
        {"items": [{"id": 0, "english": "The claim, in English.", "queries": ["q"]}]}
    )
    plan = plan_queries([text], judge(answer))
    assert plan.hypotheses == ("The claim, in English.",)


def test_a_certainly_english_claim_is_checked_as_written_whatever_the_judge_says() -> None:
    claim = "OpenAI shut down ChatGPT in 2025."
    answer = json.dumps({"items": [{"id": 0, "english": "OpenAI closed ChatGPT.", "queries": []}]})
    assert plan_queries([claim], judge(answer)).hypotheses == (claim,)


def test_an_echo_of_a_not_certainly_english_claim_keeps_the_claim() -> None:
    claim = "Los Angeles has the largest port in the country."
    answer = json.dumps({"items": [{"id": 0, "english": claim, "queries": []}]})
    assert plan_queries([claim], judge(answer)).hypotheses == (claim,)
