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
        judge(JudgeUnavailable("HTTP 429 from https://api.test/v1/chat/completions")),
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
        ("OpenAI batti ve kapandi.", False),  # ASCII Turkish, "ve"
        ("El gobierno cerró la empresa.", False),
        ("Die Firma ist pleite.", False),  # "ist"
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
        judge(JudgeUnavailable("HTTP 429 from https://api.test")),
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
    claim = "Le chat est noir."
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
