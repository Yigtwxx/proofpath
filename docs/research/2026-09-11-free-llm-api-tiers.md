# Free LLM API tiers for the optional judge — surveyed 2026-09-11

Context: spec §11 makes the judge opt-in and free by default. This survey read the
official pricing, rate-limit and terms pages of 17 providers on 2026-09-11 to pick
which free tier the judge adapter (Phase 9) should default to. Free tiers get
withdrawn without notice — Cerebras did so in August 2026 — so **the judge must
always degrade to the local ONNX NLI path**, never fail.

Workload: ~5 calls per document, ~5-6k input + 1-2k output tokens each.

## Withdrawn or never free

| Provider | Status |
|---|---|
| GitHub Models | retired 2026-07-30 |
| Together AI | free models removed 2025-11-13; $5 minimum prepay |
| Cerebras | free tier withdrawn ~2026-08-17; now $5 credit for 30 days, card required |
| DeepSeek, xAI | prepaid only |
| OpenAI | no signup credit; daily complimentary tokens only with a paid balance + data sharing |
| Anthropic | small one-time credit, then prepaid |
| Fireworks | $1 one-time, 10 RPM without a card |

## Usable, no card

| Provider | Best free model | RPM | RPD | Token cap | Strict JSON | Trains on data | OpenAI-compatible |
|---|---|---|---|---|---|---|---|
| **Groq** | `openai/gpt-oss-120b` | 30 | 1,000 | 8K TPM / 200K TPD | yes (constrained) | no | yes |
| **Gemini API** | `gemini-3.8-flash`, `2.5-pro` | unpublished (login-only since 2026-09-02) | — | — | yes | **yes, outside EEA/UK/CH** | yes |
| NVIDIA build.nvidia.com | gpt-oss-120b, nemotron-3-super, deepseek-v4 | 40 | 10,000 | none published | per model | no | yes |
| Cloudflare Workers AI | `llama-3.3-70b-instruct-fp8-fast` | 300 | 10K Neurons/day (~370K input tokens) | — | Llama family only | no | yes |
| Mistral | large-3 / small-4 | unpublished | — | $10/month credit | schema-in-prompt | **yes by default**, opt-out | yes-shaped |
| OpenRouter `:free` | `nvidia/nemotron-3-super-120b-a12b:free` | 20 | 50 (1,000 after a one-time $10) | — | per endpoint | per provider | yes |
| SambaNova | gpt-oss-120b, Llama 3.3 70B | 20 | 20 | 200K TPD | best-effort | unclear | yes |
| Cohere trial | command-a-plus | 20 | ~33 (1,000/month) | — | yes | yes; non-commercial only | yes |
| Hugging Face | gpt-oss-120b via providers | — | — | $0.10/month | per provider | provider-dependent | yes |

## Recommendation

1. **Groq `openai/gpt-oss-120b`** as the default judge: no card, no training,
   strict JSON schema. Constraint: 8K TPM, so batches must stay under ~7k tokens
   and the adapter paces at about one call per minute — acceptable for 2-4 calls
   per document. 200K TPD ≈ 30 documents per day.
2. **Gemini 3.8 Flash** as an explicit opt-in with a data warning: strongest free
   model, but quotas are opaque and Google trains on free-tier prompts for users
   outside the EEA/UK/CH (Turkey included). proofpath sends third-party passage
   text, so this must never be silent.
3. **Cloudflare Workers AI Llama 3.3 70B**: explicit no-training clause, but JSON
   mode is Llama-only, non-streaming and can fail with "JSON Mode couldn't be met".

NVIDIA's catalog has the largest quota and the best models, but its Trial ToS
forbids production use and allows discontinuation at any time: fine for our own
development runs, not a documented backend.

Design consequence: every usable provider exposes an OpenAI-compatible
`chat/completions` endpoint, so one adapter configured by `base_url + model + rpm +
tpm` covers all of them. Ollama stays the offline default.

## Riskiest to depend on

OpenRouter `:free` (roster rotates weekly, 50 RPD), NVIDIA (trial-only ToS),
Gemini (quotas unpublished, training on inputs), Hugging Face ($0.10 "subject to
change"), Groq model churn (Llama 3.3 70B dropped from free on 2026-08-16 — pin to
gpt-oss-120b and handle 404s).

Unverified: Gemini and Mistral numeric limits (login-gated), SambaNova training
stance, Anthropic trial amount.
