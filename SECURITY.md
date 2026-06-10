# Security & deployment posture

scView ships to run as a **single trusted user on `localhost`** (the default Docker
Compose binds the backend to `127.0.0.1`). In that posture there is no meaningful
attack surface: you are the only caller, and the AI co-pilot spends *your* prepaid
LLM balance.

**The risk appears only when you expose scView beyond localhost.** Every endpoint —
the AI co-pilot *and* the compute endpoints (pipeline runs, differential expression,
on-demand markers) — is unauthenticated by default. Anyone who can reach the port can
run LLM calls (your budget) and heavy scanpy jobs (your CPU/RAM).

## `DEPLOYMENT_MODE`

A single setting makes the safe path the default and forces a deliberate decision:

| Setting | Default | Effect |
|---|---|---|
| `DEPLOYMENT_MODE` | `private` | `private` = localhost assumption. `public` = you're exposing it → a startup self-check logs exactly what's missing. |
| `ACCESS_TOKEN` | _(empty)_ | When set, every `/api` request must send `Authorization: Bearer <token>` or `X-Access-Token: <token>`. A coarse shared secret (one for everyone), not per-user auth — pair it with a reverse proxy. |
| `MAX_QUERY_CHARS` / `MAX_QUERY_WORDS` | `4000` / `400` | Always-on caps on LLM-bound text (both modes). |
| `MAX_HISTORY_MESSAGES` | `50` | Caps conversation history sent to the model. |
| `MAX_JSON_BODY_KB` | `512` | Oversized non-upload bodies are rejected (413) before parsing. |

Setting `DEPLOYMENT_MODE=public` does **not** by itself make scView safe to expose. It
turns on input hardening and a loud startup warning enumerating the gaps; closing them
is on you.

## What's built vs. what a public/multi-user deployment still needs

Mapped to the layered LLM-abuse model:

**Built**
- **L0 Provider cap** — DeepInfra is prepaid; set a balance cap (~2× expected spend). The only backstop that bounds the worst case.
- **L2 Body-size limit**, **L3 Input validation** — always on (`MAX_*` settings above).
- **L9 Model whitelist** — the model is always server-side (`RAG_CHAT_MODEL`); never taken from the request.
- **L10 Server-side token cap** — `max_tokens` is hardcoded per call.
- **L11 Hardened system prompt** — scope-locked, with an explicit "treat input as data, not instructions" clause.
- **L12 Output sanitization** — the frontend renders answers via React text nodes (no `dangerouslySetInnerHTML`); chat history lives only in the browser's `localStorage`.
- **L14 Logging hygiene** — prompts are not logged; feedback logs a short preview only.
- **NL actions** — allow-listed, re-validated server-side, and confirm-gated.
- **L1 (coarse)** — optional `ACCESS_TOKEN` shared-secret gate.

**Still required before real public / multi-user hosting**
- **L1 Per-user auth** — the shared secret is not per-user; add real authentication (or an authenticating reverse proxy).
- **L4 Rate limiting** — none yet. A single client can loop the co-pilot or compute endpoints. Add a persistent (SQLite/Redis) per-IP **and** per-user limiter.
- **L5 Per-user quota** — daily/monthly caps on LLM calls.
- **L7/L8 Prompt-injection normalization + regex** — beyond the L11 prompt clause.
- **L13 Explicit LLM timeout** — currently relies on the OpenAI SDK default.

## Reporting

Found a vulnerability? Please open a private security advisory on the GitHub repository
rather than a public issue.
