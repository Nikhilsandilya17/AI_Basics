# 09 — Project Cheat Sheet: Godric & AgentHub

> **Goal:** Turn your two projects into interview weapons. Every concept from Files 01–06 has a "I built this" story here. Use the elevator pitch → architecture → concept mapping → Q&A structure.

---

# PART A: GODRIC — PhonePe's LLM Gateway

## A1. Elevator pitch (30 sec, memorize)

"Godric is PhonePe's internal LLM gateway — a centralized, multi-tenant platform that gives every team in the company one OpenAI-compatible API to call any LLM. Under the hood it does model and provider routing across OpenAI, Azure OpenAI, Anthropic, and our internal serving platform; token-based rate limiting and metering per client; input/output guardrails (prompt-injection detection, LLM-as-judge content tagging, faithfulness checks); and a benchmarking system that scores models on golden datasets with cost, latency, and similarity metrics. It's essentially an AI Gateway — the same system you'd design if I asked you to 'design a multi-provider AI gateway at scale.'"

## A2. Architecture (draw this on the whiteboard)

```
Client apps (any team at PhonePe)
   │  OpenAI-compatible API: /v1/chat/completions, /v1/embeddings,
   │  /v1/audio/speech, /v1/images/generations, /v2/rerank ...
   ▼
godric-proxy  (read-only edge: SSO auth (Olympus-IM), URL rewriting, entity cache)
   ▼
godric-service  (control plane + execution plane)
   ├── Entity Registry: LLM / Embedder / Reranker / TTS / Image / Video entities
   │      (spec, pricing, capacity, lifecycle: PREVIEW→READY→DEPRECATED→REMOVED)
   ├── Gateway: targets (multi-LLM) + routing config + guardrail spec
   ├── QuotaManager: tokens/min, per-5-min, per-day — per client, per LLM (Aerospike/Yoda counters)
   ├── Guardrail Engine: parallel rule eval, input & output, STRICT/LENIENT criteria
   ├── Estimate APIs: /v1/estimate/cost, /v1/estimate/latency (pricing + prefill/decode rates)
   ├── Benchmark: datasets + jobs (queue → worker → metrics → reports)
   └── Metering: every call → events (Foxtrot audit + Pinot analytics)
   ▼
Providers: MACHINA (internal vLLM/SGLang/MLflow serving) · OpenAI · Azure OpenAI (incl. Anthropic model family)
```

## A3. Concept mapping (project ↔ Files 01–06 — this is the gold)

| Concept (file) | How Godric implements it |
|---|---|
| **Provider routing, hosted vs self-hosted, hybrid** (File 02 §3.4, File 03 §2.5) | MACHINA (self-hosted, internal) + OpenAI/Azure/Anthropic (hosted) behind ONE API — the hybrid gateway pattern, literally |
| **Model registry & lifecycle** (File 01 §2.6) | LLMEntity with pricing/capacity/auth + lifecycle states (PREVIEW/READY/DEPRECATED/REMOVED); model IDs are `namespace:llmId` |
| **Multi-tenancy, noisy neighbour** (File 03 §1.11) | Namespace-scoped entities; per-client quotas layered on per-LLM defaults |
| **Token-based rate limiting** (File 03 §1.10) | QuotaManager: tokens per minute / 5-min / day, enforced BEFORE the call (`assertAvailability`), metered AFTER (actual usage events) |
| **Token metering & cost attribution** (File 03 Part 4) | Per-call token events → Foxtrot (audit) + Pinot (analytics); usage dashboards overall/granular/quota; cost estimation API from pricing specs |
| **Cost & latency estimation** (File 02 §1.6–1.7, File 03 §4.1) | `/v1/estimate/cost` = (input·inputCost + cached·cachedCost + output·outputCost)/1M — cached-token pricing is modeled! Latency = input/prefillRate + output/decodeRate — prefill/decode split built into the estimate |
| **Guardrails: input & output moderation** (File 06 §3.8) | Guardrail entities attached to gateways; rules run on INPUT and OUTPUT sides, in PARALLEL with inference (input) and after (output) |
| **Guardrail rules** (File 06 §3.2–3.7) | Prompt-injection classifier (ProtectAI model), LLM-as-judge tagging rule (scores configurable tags 0–100 via structured JSON output), keyword blacklist, Granite Guardian harmful-content, output relevancy + FAITHFULNESS via DeepEval — groundedness as a guardrail! |
| **Shadow vs live guardrails** (File 06 §2.10) | GuardEvalMode: LIVE (block) vs SHADOW (monitor only, async) — canary-style rollout of safety policies |
| **Golden datasets & eval** (File 06 §2.1) | Benchmark module: versioned datasets (CSV/JSON), locked, with status flows |
| **Offline metrics: latency, cost, token usage, similarity** (File 06 Part 1, File 01 §3.1) | Benchmark metrics: TOKEN_USAGE, LATENCY (ms), COST (USD, incl. cached tokens), TEXT_SIMILARITY (embedding cosine vs reference); weighted assertions with operators (<, >, contains...) and passingScore |
| **Weighted scoring, weighted assertions** (File 01 §3.1) | Each assertion has weightage → weighted benchmark score → PASS/FAIL vs passingScore |
| **Async job execution / at-least-once queues** (File 05 §4.2–4.3) | Benchmark jobs on an Aerospike queue with leases, heartbeats, TTL, checkpointing, crash recovery (KILLED/CRASHED states) |
| **Streaming, SSE** (File 02 §1.5) | SSE streaming on chat completions; even normalizes Anthropic SSE events → OpenAI-style chunks (provider adaptation) |
| **Provider abstraction / API normalization** (File 03 §2.5) | JSONata/Jolt transformation specs per provider — request/response AND SSE chunks normalized to OpenAI schema |
| **RAG chains** (File 04) | `/v1/chain` — RagChain = LLM + prompt template + generation params (RAG execution as a first-class entity) |
| **Batch/offline inference** (File 01 §1.3) | Kafka-sourced offline inference jobs with throughput monitoring (OpenTSDB) |
| **Graceful degradation on quota-system failure** (File 06 §4.6) | Quota checks fail-open on quota-system errors (availability > strictness trade-off — say this consciously!) |

## A4. Likely questions + answers

**Q: "Why build an LLM gateway at all?"**
A: One control point for: (1) auth and per-team cost control — LLM spend is real money, quotas stop runaway costs; (2) provider abstraction — teams write to ONE API; we swap/add providers (OpenAI ↔ Azure ↔ internal MACHINA) without app changes; (3) uniform observability — every token, cost, and latency measured once; (4) central guardrails — safety policy applied once, not per-team; (5) benchmarks — data-driven model choice per use case. Same reasons ChatGPT-scale companies build gateways.

**Q: "How does a request flow through guardrails?"**
A: Gateway exec selects a target LLM (routing config), enriches the request, and if a guardrail is attached: input evaluation runs IN PARALLEL with the inference call (CompletableFuture on a dedicated pool) — we don't add input-guard latency on the critical path; then output evaluation runs against the response. Any FAILED rule → request rejected with the inference ID and reason. Every guardrail eval is emitted as an event for audit. Guardrails can also run in SHADOW mode — evaluate and log but don't block — which is how we canary new safety rules.

**Q: "How do you rate limit an LLM platform?"**
A: By tokens, not requests — a chat call and a 50k-token RAG call are not the same cost. We enforce per-minute, per-5-minute, and per-day token quotas per client per LLM, checked before dispatch, with counters in a fast store (Aerospike/Yoda). Actual usage is metered from every call's usage events for dashboards and alerts. One deliberate trade-off to mention: on quota-system failure we fail OPEN (serve the request) — we chose availability over strict enforcement, and alerting covers the gap.

**Q: "How do you pick which model a team should use?"**
A: Data, not vibes — the benchmark module: a team's real prompts as a versioned golden dataset, each candidate model runs the same request template, metrics are computed per row (cost via pricing specs, latency, token usage, embedding similarity vs reference answers), assertions are checked with weights, and the report gives a score vs passingScore. That's how routing decisions and the entity registry stay honest.

**Q: "What was a hard technical problem?"** (prepare your own story; good candidates from the code)
- Provider normalization: Anthropic via Azure with different SSE event shapes — solved with per-provider transformation specs including streaming-event mapping.
- Running input guardrails without adding latency — parallel execution with inference, criteria evaluator short-circuits as results arrive.
- Benchmark job reliability — at-least-once queue with leases/heartbeats/checkpoints so long benchmark runs survive worker crashes.

**Q: "What would you improve?"** (always have this ready — shows judgment)
- Routing strategy is currently random across targets → add cost/capability/latency-aware routing and cascades (the File 03 §1.5 machinery).
- Guardrail rules are chat-completion-scoped → generalize to other workflows (agents, embeddings).
- Add semantic/prompt caching at the gateway layer (File 03 §2.1–2.2) — cached tokens are already priced, so the metering is ready for it.

---

# PART B: AGENTHUB — PhonePe's AI Operating Layer

## B1. Elevator pitch (30 sec, memorize)

"AgentHub is PhonePe's internal AI agent platform — the 'AI operating layer for enterprise work.' Teams build Agents, Skills, and workflow DAGs on top of indexed knowledge bases (Confluence, Jira, GitLab, Google Drive), connected to enterprise systems through MCP servers — Jira, Confluence, Gmail, Drive, Slack. An orchestrator agent routes work to specialist agents, with a completion-critic that verifies drafts before they reach users, workflow guardrails, RBAC (User/Builder/Admin), full audit trails, and per-agent token budgets. It's organized into team workspaces — legal, compliance — that run real operational workflows like contract review and third-party risk assessment, triggered by email, Jira events, or schedules."

## B2. Architecture (whiteboard version)

```
Users (User / Builder / Admin roles)  ·  Slack (/agenthub)  ·  MCP clients
        │
        ▼
agenthub-ui (React/TS) → Flask backend (application/)
        │                        │
        │   Orchestrator Agent (Pydantic AI, SSE streaming)
        │     ├─ web search / file analysis / code execution tools
        │     ├─ dynamic specialist discovery (roster in system prompt + search_agents)
        │     ├─ delegation w/ CONFIRMATION_REQUIRED side-effect gate
        │     └─ Completion Critic (verifies draft vs action trace before showing user)
        ▼
   Celery + RabbitMQ workers (master / kb / execution / scheduler)
        ├─ Workflow engine: DAG, 16 node types (Agent, HTTP, Python, Gmail, Condition...)
        │    triggers: MANUAL, SCHEDULED, EMAIL, SLACK, JIRA, WEBHOOK, RSS, FILE_CHANGE
        ├─ KB indexing: 7 source adapters (Confluence, GitLab, Jira, GDrive, Web, URL, LMS)
        │    incremental sync, vector store + semantic services
        └─ Skills: filesystem bundles (SKILL.md + scripts/) injected into agent context
Storage: Elasticsearch (agents/workflows) · Aerospike (conversations, teams, MCP sessions)
Governance: scope=global|team + acl=public|private|custom · RBAC · audit trail · token budgets
External tools: jira-mcp (~70 Jira tools), google_workspace_mcp, SlackHub (Slack proxy)
Godric = the model gateway underneath (provider routing, quotas, guardrails)
```

## B3. Concept mapping (project ↔ Files 01–06)

| Concept (file) | How AgentHub implements it |
|---|---|
| **Agent loop, tools, memory** (File 05 Parts 1–2) | Pydantic AI agents with tool builders; semantic memory service (stores summaries in vector store, retrieved when relevant); conversation store in Aerospike |
| **Supervisor/orchestrator + sub-agents + delegation** (File 05 §3.4, §3.7–3.11) | Orchestrator agent discovers specialists dynamically (`search_agents`/`invoke_agent`), delegates, gets only summaries back; completion-critic reviews children's outputs |
| **Router pattern** (File 05 §3.3) | Specialist agents registered with handles/tags; roster rendered into orchestrator's system prompt (they fixed lexical-ES-search routing with prompt-roster — a great war story: "what is Minerva?" question) |
| **Human-in-the-loop / side-effect gates** (File 05 §4.5) | `CONFIRMATION_REQUIRED:` gate on side-effectful delegated calls; MCP `run_agent` refuses mutating agents without `confirm=True`; approval fingerprints |
| **Retries & self-correction** (File 05 §4.1) | ModelRetry with recovery instructions (bounded budget, default 2 rounds); tool retries ("Web search timed out. Retried once."); model fallback on 429 |
| **Durable/async execution & triggers** (File 05 §4.2–4.3, File 01 §1.3) | Celery queues per worker type; scheduled (Beat), event-triggered (EMAIL, SLACK, JIRA, WEBHOOK, RSS, FILE_CHANGE) — long-running workflows survive worker restarts |
| **Workflow agents (deterministic skeleton + LLM steps)** (File 05 §3.5) | The DAG workflow engine: 16 node types where LLM/agent nodes sit inside a deterministic graph — exactly the "default to workflow agents" pattern |
| **Skills vs sub-agents — cost spectrum** (File 05 §3.5–3.6, File 03 cost levers) | MCP tools (no model) → Skills (instructions+scripts injected into an existing agent's context, near-zero marginal cost) → sub-agents (full reasoning loop). "Don't pay for a full reasoning agent when someone needed a recipe." |
| **Context management / compaction** (File 02 §1.3, File 04 §3.12) | Layered compaction: lossless elision → bounded truncation → lossy summarization; docstore offloading of oversized tool results; rolling cross-turn summaries; fail-open |
| **RAG pipeline** (File 04) | 7 KB source adapters with incremental sync; vector search; per-agent KB references `{{kb_name}}` in prompts; re-ranking strategy configurable (semantic/keyword/hybrid) in agent settings |
| **Guardrails — workflow level** (File 06 §3.8) | WorkflowGuardrails at create/update: GmailQueryGuardrail (blocks overbroad Gmail queries), EventVolumeGuardrail (blocks >80k events/day) |
| **Output evaluation — completion critic** (File 06 §2.5, §3.8) | Critic model checks draft answer vs action trace (complete? quality_ok? issues?) BEFORE user sees it; review-gated streaming — rejected drafts never shown |
| **Agent analytics / online metrics** (File 06 §1.1, File 01 §3.2) | Total queries, positive feedback %, vagueness score, prompt adherence, avg tokens, feedback themes |
| **Cost budgets** (File 05 §4.7, File 03 §1.10) | Per-agent token limit (default 8192); admin dashboard: monthly allocation, budget ($150/mo), per-agent breakdown by model, projected month-end at 85% warning |
| **Multi-tenancy & permissions** (File 03 §1.11, File 04 §4.1–4.2) | scope (global/team) + ACL (public/private/custom) on agents, workflows, tools, MCP servers; team key = BU::team::pod; KBs are permission-gated at function level; ES query enforces `scope=team AND (user OR (acl=public AND teams∈user_teams))` |
| **Fail-open/fail-closed + circuit breakers** (File 06 §4.4–4.6) | Team-ACL cache: stale-while-revalidate (3 days), circuit breaker on team-service (3 failures → 60s fast-fail), FAIL-CLOSED when Aerospike down (authz errors are safer than leaks — contrast with Godric's quota fail-open; knowing WHICH to choose is the senior signal) |
| **MCP (tool standard)** (File 05 §1.4) | Consumes MCP (jira-mcp ~70 tools, google_workspace_mcp, SlackHub) AND exposes MCP outward (FastMCP at /mcp, 33 tools, tiered; run_agent with confirm gate) |
| **Product strategy** (File 05 §0 autonomy trade-off) | Personal agentic assistance (2025-26) → team-based agent systems (2026, HITL) → enterprise context+action graph (2027+). Layers: Harness / Integrations+Execution / Governance |

## B4. Likely questions + answers

**Q: "What's the difference between an agent, a skill, and a workflow in AgentHub?"**
A: Cost spectrum, ascending autonomy: an MCP tool is pure execution, no model. A Skill is an instruction-plus-optional-scripts bundle (a directory with SKILL.md + scripts/) injected into a calling agent's context — near-zero marginal cost, great for fixed procedures (checklists, format conversions, extraction). A sub-agent is a full reasoning loop — use when the next step depends on judgment about the previous output. A workflow is a deterministic DAG with LLM/agent nodes inside — predictable, testable. We default to the cheapest tier that works; that discipline is why the platform's unit economics hold.

**Q: "How do you surface the right skills/agents without stuffing them all in context?"**
A: Context budgeting — skills are indexed for local vector search; only the top 3–5 relevant skill names/descriptions hit the initial prompt; the full SKILL.md content loads only when invoked. Same philosophy for specialists: the orchestrator gets a roster in its system prompt plus a `search_agents` tool for discovery, instead of every agent's definition upfront.

**Q: "How do you make sure agents don't do something harmful?"**
A: Layers: (1) tool permissions — RBAC per team, private/public scoping, per-agent tool allow-lists; (2) side-effect gates — delegated calls with side effects require explicit confirmation (`CONFIRMATION_REQUIRED`); MCP `run_agent` on mutating agents needs `confirm=True`; (3) workflow guardrails at creation (overbroad Gmail queries, event-volume bombs); (4) completion critic verifies output against the action trace before the user sees it; (5) full audit trail of every delegated call. And underneath, Godric applies input/output guardrails at the model layer. Defense in depth — no single layer is trusted.

**Q: "How do you handle a conversation that outgrows the context window?"**
A: Layered compaction with escalating lossiness: lossless elision first (drop redundant formatting), bounded truncation of stale tool results (oversized results offloaded to docstore with references), then lossy summarization of older turns (rolling summaries). Each stage fails open — degradation is gradual, never a hard error.

**Q: "A workflow triggers on every Jira ticket creation — how do you prevent cost/runaway?"**
A: EventVolumeGuardrail at workflow-creation time — queries the event platform and blocks workflows projected above 80k events/day. At runtime: per-agent token limits, model fallback on 429s, bounded retry budgets, and the admin token dashboard with per-agent/per-model breakdown and projected month-end cost (alerts at 85% of budget).

**Q: "How do agents talk to Jira and Gmail?"**
A: MCP — Model Context Protocol — an open standard for exposing tools to models. We run dedicated MCP servers: jira-mcp exposes ~70 Jira tools (plus Confluence tools) with a read-only mode; google_workspace_mcp handles Gmail/Drive/Calendar/Sheets with OAuth. AgentHub ALSO exposes its own MCP server outward (33 tools: introspection, agent/workflow building, execution) so other platforms and sidekick agents can use AgentHub itself as a tool — with a confirm gate on executing agents that have mutating tools.

**Q: "What would you improve?"**
A: (prepare your own; candidates) richer trajectory-level evaluation for agents (we measure vagueness/prompt adherence/feedback, but trajectory eval on golden tasks is the next step); automated model routing per task (roadmap item — today model choice is manual per agent); cross-team memory sharing with stronger isolation guarantees.

---

# PART C: The two projects together (the strongest story)

**The one-slide story:** "AgentHub is where teams build AI work; Godric is how that work reaches models safely and cheaply. AgentHub's agents call LLMs through Godric's OpenAI-compatible gateway — so every token is metered, quota'd, guarded (prompt-injection classifier, faithfulness checks), and routable across providers (internal vLLM on MACHINA, OpenAI, Azure, Anthropic) without app changes. Benchmarks decide which models deserve traffic; guardrails decide what traffic is safe; quotas decide what traffic is affordable."

Map to the interview tracks: **Godric = LLM Serving & AI Infrastructure (File 03) + Evals/Safety (File 06). AgentHub = Agentic Systems (File 05) + RAG (File 04) + multi-tenancy.** Between the two you have personally worked on ~80% of the AI system design syllabus.

---

## Final prep checklist (Day 2 evening)

- [ ] Recite both elevator pitches out loud, timed (30s each)
- [ ] Draw both architectures from memory on paper
- [ ] For each Part A3/B3 row: say one sentence connecting the concept to the implementation
- [ ] Have 2 war stories per project ready (one failure + fix, one hard problem + solution)
- [ ] Have one "what I'd improve" per project (never say "nothing")
- [ ] Re-run File 08 questions you marked weak
