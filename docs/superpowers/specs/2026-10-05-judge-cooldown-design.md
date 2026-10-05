# Judge: back to the provider once its minute limit resets — design (2026-10-05)

This amends evidence-search spec §11 ("Local fallback judge"), which made the switch
to the local model last for the rest of the run. OPEN-ITEMS §22 points here.

## 1. Decision (user, 2026-10-05)

The user asked whether the judge could go back to Groq once its per-minute limit
resets, instead of staying on `qwen3.5:9b` for the rest of the run. They chose option
A, **a cooldown with return**, over two others: waiting on the provider (B), and A plus
proactive pacing from the `x-ratelimit-*` headers (C).

§11 had two goals, and both are kept:

- **Nobody waits.** With a fallback behind it, the primary still gets no retry and no
  `Retry-After` sleep. While the provider cools down, the local model answers.
- **The provider is not hammered.** It is asked again only once the time it named
  itself has passed, and never on every batch.

## 2. Behaviour

1. **Which failures cool down.** Only a **429** whose `Retry-After` is at most
   `COOLDOWN_MAX_S = 120` seconds. Groq's per-minute token and request limits fit this.
   A 429 without the header counts as `COOLDOWN_DEFAULT_S = 60`. The switch stays
   **sticky** for the rest of the run, exactly as before, for:
   - a 429 with a longer `Retry-After` (a daily limit: the RPD or TPD tier);
   - **413** (the request alone breaks the TPM, so asking again cannot help);
   - 401 and 403 (the key);
   - 5xx, a timeout, no answer, or an empty answer.
2. **Carrying the wait.** `JudgeUnavailable` gains `retry_after: float | None`.
   `JudgeClient` fills it from the same `_retry_after_seconds` parse it already uses,
   including when it is `fail_fast`. Never a body, a URL or a key.
3. **Routing.** `FallbackClient` keeps `_cooldown_until`, a clock reading or `None`.
   - On a cooling failure it switches as today and sets
     `_cooldown_until = now + retry_after`. That covers the notice, the warm-up and the
     local answer for the failed request.
   - While `now < _cooldown_until`, requests go to the local model.
   - The first request at or after `_cooldown_until` goes to the primary again.
     - If it answers, routing returns to the primary. A **back notice** is announced
       once: `Groq answering again — local ollama <model> stood in for N requests`.
     - If it fails again with a cooling 429, a new cooldown starts from the new
       `Retry-After`, and that request goes to the local model. No second switch
       notice, and no second warm-up.
     - If it fails in a sticky way, the switch becomes sticky from then on, as today.
   - The clock is injected (`clock: Callable[[], float] = time.monotonic`) so tests
     need no sleep. The existing lock guards every read and write of
     `_cooldown_until`. Only one thread probes the primary when the cooldown ends;
     the others keep using the local model until the probe has answered.
4. **Notices.**
   - The switch notice for a cooling 429 says the switch is temporary:
     `Groq limit reached — judging with local ollama <model> until it resets (~Ns)`.
   - `switched` keeps the latest notice: the switch, or the back notice once the
     provider answers again. The report footer therefore tells the truth at the end
     of the run.
   - Every opinion's `model` field still names the model that gave it, and the
     report's cost line still splits `api_calls` from `local_calls`.
   - A sticky switch keeps today's wording, unchanged.
5. **`reset()`** (a new run) clears `_cooldown_until`, as it already clears the rest.
6. **Local failures.** These are unchanged (`_dead`, `JUDGE_FALLBACK_FAILED`). If the
   local model is dead while the provider cools down, the request fails as today. The
   next request after the cooldown still probes the provider.

## 3. Tests (no network, no sleep)

Use `respx`, or the existing fake clients, plus a fake clock. They cover:

- a short 429 switches and then comes back after `Retry-After`, with one switch
  notice and one back notice;
- requests inside the cooldown go to the local model and send nothing to the primary;
- a second short 429 at the probe restarts the cooldown with no new notice;
- a long 429, a 413, a 401 and a 5xx each stay sticky with today's notice;
- a 429 without `Retry-After` uses 60 s;
- `reset()` clears the cooldown;
- with concurrent requests at the end of the cooldown, only one probes;
- `JudgeUnavailable.retry_after` is filled on the `fail_fast` path.

## 4. Out of scope

- Proactive pacing from `x-ratelimit-remaining-tokens` (option C).
- Waiting on the provider (option B).
- Any change to the batch caps (`TOKEN_CAP`, `_REVIEW_TOKENS`).
