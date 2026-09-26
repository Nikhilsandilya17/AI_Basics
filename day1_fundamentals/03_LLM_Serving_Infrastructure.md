# 03 — LLM Serving & AI Infrastructure (Track 13)

> **Goal:** Know what happens AFTER an LLM app faces real production traffic — GPUs, scaling, caching, routing, cost math, and distributed inference. Every section here maps to real production systems.

---

## Part 1: Serving & Scaling

### 1.1 GPU vs CPU Inference

**Plain words:** a CPU is a few genius workers; a GPU is thousands of average workers doing simple math simultaneously. LLM inference = millions of small matrix multiplications, massively parallel → GPU territory.

**Full explanation — why the hardware shapes the workload:** a CPU core is designed to execute complex instruction streams fast, one after another — brilliant at branching logic, business rules, serialization. A GPU packs thousands of simpler cores that all perform the SAME operation on DIFFERENT data at the same instant (SIMD). A model forward pass is a handful of giant matrix multiplications — each one is "multiply and add these millions of number pairs" — exactly the shape GPUs were built for. That's the entire reason LLM serving = GPUs.

**When CPU is still right (know the boundary):** small models (≤1–3B), short contexts, low traffic, and workloads that are mostly pre/post-processing (tokenization, validation, business logic — normal code). A 1B model classifying support tickets at 100 req/min is a CPU workload; a 70B chat model is not.

**The architectural point to make in interviews:** the serving stack AROUND the model — auth, routing, guardrails, metering — is CPU work and scales like normal services. The model itself is GPU work and scales by GPU economics. Keep them decoupled: the gateway tier scales independently of the inference fleet — exactly why a gateway sits between apps and model endpoints.

### 1.2 GPU Memory — the fundamental constraint

**The equation to memorize (it governs everything in this track):**
```
GPU memory = model weights (fixed) + KV cache (per concurrent request) + activations/overhead
```

**Worked example (practice doing this live on a whiteboard):**
- Model: 8B parameters, FP16 → 16 GB weights (2 bytes × 8B).
- Context: 8k tokens per request → KV cache ≈ 2 GB per request (order of magnitude for this class — File 02 §2.4's formula).
- GPU: one 80 GB H100.

Arithmetic: 80 − 16 (weights) − ~5 (overhead/activations) = ~59 GB available for KV cache → at ~2 GB per concurrent request ≈ **25–30 concurrent requests per GPU.** Not thousands. Not hundreds. Thirty.

**The punchline and its consequences:** concurrency per GPU is limited by MEMORY, not compute. The GPU may sit at 40% compute utilization while requests queue — the KV cache is full. Three levers raise concurrency:
1. Shrink the fixed part — quantize weights (File 02 §3.1): 16 GB → 8 GB frees 8 GB → 4 more concurrent requests.
2. Shrink the per-request part — cap context length: 8k → 4k halves KV per request → doubles concurrency.
3. Waste less — PagedAttention (§1.9): naive preallocation wastes 60–80% of KV memory; paging recovers it → 2–4× more concurrency from the same hardware.

Every capacity conversation in LLM serving is these three levers.

### 1.3 Model Replicas

**Plain words:** N identical copies of the model behind a load balancer — standard scale-out, with two GPU twists.

**The twists, concretely:**
1. **Cost floor:** each replica holds the full model in GPU memory — a 70B FP16 replica burns 140 GB (2+ GPUs) before serving a single token. Replicas are expensive even when idle; right-sizing replica count IS cost engineering.
2. **Cold start:** loading tens of GB of weights from storage and warming up takes MINUTES, not seconds. Reactive scaling ("traffic spiked, spin a pod") fails — the spike is over before the replica is warm. So: warm pools (pre-started replicas absorbing bursts) or over-provisioning during known peaks, and scale-down that is slow and hysteresis-driven (§1.6).

### 1.4 Request Queues

**Plain words:** when all replicas are busy, requests wait — with timeout, priority, and fairness rules.

**Full explanation — walk through what a well-designed LLM queue needs, requirement by requirement:**

1. **Timeout per queued request** — waiting forever is worse than a fast failure. When the clock runs out, return a clear 429-style error with retry-after guidance rather than letting clients hang and retry blindly.
2. **Priority classes** — interactive chat users should jump ahead of a nightly batch job scoring 100k documents. Priority queues make that explicit; without them, batch jobs poison interactive latency.
3. **Fairness across tenants** — without fairness controls, one tenant's burst floods the queue and everyone else's TTFT explodes (the noisy neighbour — §1.11). Weighted-fair or per-tenant-cap queuing fixes it.
4. **Admission control** — better to REJECT early with a clear error than QUEUE forever: a bounded queue that sheds load protects in-flight requests from latency collapse (load shedding — File 06 §4.6).
5. **Observability** — queue depth and queue-wait-time are first-class metrics: they're your autoscaling signal (§1.6) and your "users are suffering" alarm. Queue depth trending up at 2 PM daily = capacity story before users complain.

**Why queues matter more than people think:** queue wait is part of user-perceived latency — `latency = queue + TTFT + generation` (File 02 §1.7). A model that answers in 200ms behind an 8-second queue is a slow product. Capacity planning (Part 4) exists to keep P99 queue wait near zero.

### 1.5 Model Routing (inside your fleet)

**Plain words:** choosing WHICH model handles a request, by policy.

**The four routing dimensions, with reasoning:**
1. **Capability:** classify the request's difficulty → easy to small models, hard to large. Classification can be a cheap classifier model, a small LLM, or the small model's own confidence (logprobs — escalate when uncertain). This is the cascade pattern from File 02 §3.3.
2. **Context length:** a hard constraint filter — long prompts must go to models that fit them; no policy overrides arithmetic.
3. **Cost:** among models of equivalent MEASURED quality (benchmarks — File 06 §2.1), route to the cheapest. Quality-equivalence must be proven, not assumed.
4. **Residency/compliance:** some tenants' data must stay in specific regions/providers — a policy filter, not an optimization.

**The cascade pattern, concretely (the workhorse of cost engineering):** request arrives → small model answers → confidence check → high confidence: return (most requests end here, cheap); low confidence: escalate to large model (few requests, but the hard ones come back right). The engineering crux is the confidence signal — miscalibrated and you escalate everything (expensive) or nothing (wrong answers). 
### 1.6 Autoscaling (GPU edition — genuinely different from CPU)

**CPU world:** pod starts in seconds, is cheap, is stateless — autoscaling is easy and boring.

**GPU world — four differences, each with its consequence:**
1. **Scarcity:** GPUs are node-bound (a pod needing a GPU must land on a node that has one — the device plugin), often quota-limited; you can't "just scale out" — there may be nothing to scale onto.
2. **Cold start:** minutes (weights + warmup — §1.3), so reactive scaling lags spikes badly.
3. **Cost:** 10–100× CPU per unit; flapping (scale up, scale down, scale up) burns money with nothing to show.
4. **Batching headroom — the unique superpower:** a replica can absorb a spike by batching harder (File 02 §2.5): throughput rises substantially while per-request latency degrades only slightly. CPU services have NO equivalent cushion — GPU fleets have built-in burst tolerance.

**The practical recipe (recite it):** scale on queue depth and TTFT (not CPU% — it lies for memory-bound serving), with hysteresis (scale up fast, scale down slow), warm pools for known spiky windows, per-tenant quotas so one tenant's spike doesn't cause fleet-wide scale events, and admission control as the last resort (§1.4).

### 1.7 Streaming Responses (production view)

Streaming (File 02 §1.5) hits infrastructure in four places — each a real production issue:

1. **LB/proxy buffering:** SSE must pass through unbuffered — the classic silent bug (users stare at nothing, then the whole answer pops at once). Config: disable response buffering for streaming routes; test through the FULL path, not just direct-to-model.
2. **Timeouts:** per-chunk idle timeouts ("no token for 5s = dead") plus a total wall-clock cap — the two-dimensional timeout pattern from File 02 §1.5.
3. **Moderation on streams:** output guardrails can't wait for full completion on high-risk flows — either incremental checks on accumulated text or a completion check that gates the FINAL answer (File 06 §3.8). Decide explicitly; don't discover it during an incident.
4. **Mid-stream failure:** partial output was already shown — you cannot un-show it. Design the truncation notice UX, billing for partial output, and client reconnect semantics.

**What the SSE wire actually looks like (recognize this on sight — providers differ only in details):**
```
data: {"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"The ref"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"und policy"}}]}

data: {"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}],
       "usage":{"prompt_tokens":842,"completion_tokens":210}}

data: [DONE]
```
Note the four things every streaming infra must handle: deltas arrive in VARIABLE-sized chunks (not one token per frame — don't build a parser that assumes that); the `finish_reason` + trailing `usage` frame is how you BILL (post-call metering — §1.10 reads exactly this field); `[DONE]` is the sentinel ending the stream; and each frame is a complete JSON object on one `data:` line (multi-line JSON must be handled, not naively split on newlines). Knowing this wire format is what separates "read about streaming" from "shipped streaming."

**Worked end-to-end latency budget for a streamed RAG answer (put the numbers together — this is a favorite interview drill):**
```
Query processing (rewrite):        250ms
Embedding + hybrid search:          60ms
Reranking (cross-encoder):        180ms
── context assembled, LLM called ──
Queue wait:                        120ms
Prefill (2,800-token prompt
  at ~5,000 tok/s):                560ms   ← TTFT ≈ 1.2s total to first word
Decode (300 tokens at 50 TPS):    6,000ms
Total:                             ~7.2s, of which the user "feels" 1.2s
```
The insight to state: streaming converts 7.2s of waiting into 1.2s of waiting + 6s of watching words appear — same total, radically different perceived latency. And the budget shows WHERE to optimize: TTFT is dominated by prefill (prompt size) and queue; the tail is decode-bound. Every number in that budget maps to a section of this file and File 02 — being able to produce this table on a whiteboard IS the interview answer.

### 1.8 Continuous Batching (production view)

Covered fully in File 02 §2.6. The serving-stack view to add here: the scheduler owns ADMISSION — which waiting requests join the running batch at each decode step, constrained by KV memory availability (§1.2's equation). Admission is the meeting point of queueing (§1.4) and memory management (§1.9) — this is why serving engines are scheduler projects at heart, not just model loaders.

### 1.9 KV-Cache Management (PagedAttention)

**The problem, quantified:** naive serving pre-allocates CONTIGUOUS KV memory per request for its MAXIMUM possible context. A request allowed 32k tokens but actually using 2k wastes 30k tokens' worth of reserved memory. Measured in early serving stacks: 60–80% of KV memory wasted to preallocation and fragmentation. That waste is directly lost concurrency — §1.2's arithmetic again.

**The fix — OS memory paging applied to KV cache (vLLM's PagedAttention), step by step:**
1. KV cache is split into fixed-size BLOCKS (pages), like process memory in an OS.
2. A request's KV grows block-by-block, on demand — allocate only what's used. No max-preallocation, no fragmentation (a request's logical sequence maps to physical blocks via a page table — non-contiguous is fine).
3. Identical prefixes are SHARED: 1,000 concurrent requests with the same 2k-token system prompt → ONE copy of those blocks in memory, referenced by all 1,000 page tables. Copy-on-write if a request diverges (which they don't, for prefixes).

**The effect:** 2–4× more concurrent requests per GPU from the same hardware — the single most citable serving optimization. Know its name, its mechanism (pages + sharing), and its effect size.

### 1.10 Token-Based Rate Limiting

**Plain words:** classic rate limiting counts requests/sec. LLM cost and load scale with TOKENS — a 10-token request and a 10,000-token RAG request are not the same, and a request counter treats them identically.

**The full limit stack (dimensions stack, not one number):**
- **Tokens per minute** — the cost driver; per user, per team, per tenant.
- **Input vs output separately** — output tokens are priced 3–5× (Part 4) and take 3–5× longer to generate; some systems limit them independently.
- **Max context per request** — a single giant prompt is a cost bomb; cap it.
- **Concurrent requests** — GPU memory protects itself (§1.2).
- Plus classic requests/min as the coarse outer layer.

**The enforcement design (interview gold — two sides):**
- **Pre-call enforcement:** check quota BEFORE calling the provider — reject or truncate before spending money. Prevents the spend.
- **Post-call metering:** read the actual usage from the response's usage field and record it — for billing, dashboards, and anomaly detection. Proves the spend and catches abuse.

Both sides needed: pre-check without post-metering drifts from reality; post-metering without pre-checking lets a burst through first.

### 1.11 Multi-Tenancy

**Plain words:** many teams/customers share one LLM platform.

**Full explanation — build the picture problem by problem:** you've built a gateway; 40 teams onboard. Within a month, all three classic problems arrive, usually in this order:

1. **Noisy neighbour:** the risk team's nightly job scores 200k documents through the same capacity serving everyone's chat traffic. At 2 AM their requests queue; an analyst with a late-night query waits 30 seconds behind them. Same hardware, one workload hurting everyone. The fix ladder, in order: per-tenant token quotas (hard consumption ceiling — §1.10), priority queues (real-time interactive > batch), off-peak scheduling for batch jobs, and — if contention persists — physically separate capacity pools for latency-sensitive vs batch workloads.
2. **Fairness & billing:** 40 teams sharing a bill means every token must be attributable — per-tenant metering (tokens, cost, by model, by use case), monthly showback/chargeback, per-tenant dashboards so teams see their own burn. This bookkeeping is why finance loves gateways.
3. **Isolation:** tenant A's prompts must never appear in tenant B's context, caches, logs, or evaluation sets. Enforcement: cache keys that include tenant IDs (a cache-key bug = cross-tenant data leak — a real incident class), tenant-scoped guardrail configs and rate limits, access-controlled logs, tenant-separated eval sets. Isolation failures are severity-1 security incidents, not bugs.

**The monthly showback table (what multi-tenant metering produces — and the political effect worth mentioning in interviews):**

| Team | Requests | Tokens (M) | Cost | vs last month |
|---|---|---|---|---|
| Support bot | 1.9M | 412 | ₹6.1L | +12% |
| Code assistant | 210k | 288 | ₹5.4L | +41% ⚠ |
| Doc search (RAG) | 3.1M | 731 | ₹7.8L | −8% |
| Analytics copilot | 95k | 44 | ₹0.9L | +3% |

The moment every team sees THIS table monthly: cost conversations move from "AI is expensive" (vague, unactionable) to "code assistant grew 41% — worth it? optimize it? quota it?" (specific, actionable, per-team). Attribution changes BEHAVIOR — the same effect that cloud cost tags had on compute waste. This table is also where the anomalies surface first (§4.9's cost-attack detection starts exactly here).

---

## Part 2: Optimisation (the caching + fallback layer)

The order of potency to memorize: **model routing > prompt caching > semantic caching > response caching.**

### 2.1 Prompt Caching (provider prefix caching)

**Plain words:** if two requests share the SAME long prefix (system prompt, tool definitions, RAG preamble), the provider computes that prefill ONCE and reuses it. Subsequent requests pay ~10–25% for cached prefix tokens and get much faster TTFT.

**Why it works mechanically:** prefill is causal and sequential — an identical prefix produces an identical KV cache, which is reusable (File 02 §2.4). Providers cache those blocks on their side and charge cached tokens at a steep discount. **The design rules that make or break it (each rule exists because of a real failure mode):**
1. **Static content FIRST, dynamic LAST:** [system prompt → few-shot examples → tool schemas → retrieved context → user message]. Caches match a PREFIX — the first differing token resets everything after it. A timestamp in position 3 of your "shared" system prompt zeroes your hit rate.
2. **Byte-stability:** the shared prefix must be EXACTLY identical every request — a session ID, a random example order, or trailing-whitespace drift kills hits. Audit templates for accidental dynamism.
3. **Warm-up and traffic shape:** the first request pays full prefill; later ones ride the cache. Steady, repetitive workloads (support bots, RAG pipelines with fixed preambles, agents with fixed tool suites) reach 60–90% prefix overlap — the biggest cheap win in LLM cost engineering. One-off diverse traffic gets nothing.
4. **Cross-replica coherence (self-hosted):** prefix cache hits only help if the request lands on the replica holding the prefix — consistent hashing on prefix routes same-prefix traffic together (§5.2).

### 2.2 Semantic Caching

**Plain words:** before calling the model, embed the incoming query and check for a SEMANTICALLY similar past question (cosine similarity above threshold). "How do I get my money back?" ≈ "What's the refund process?" → serve the cached answer.

**Where it shines:** FAQ-heavy products — 30–60% hit rates on real support traffic; cached answers are free and instant.

**Where it bites (the interview gold — the failure mode matters more than the hit rate):** a wrong "close enough" match returns a confidently wrong answer — WORSE than a miss. "Close my account" vs "How do I close my account?" — one is a command, one is a question; semantically near, behaviorally opposite. Mitigations: high similarity thresholds (accept lower hit rates), narrow cache scope (only FAQ-validated, non-personalized answers — never account-specific queries), exact-match fast path, and treating cache hits like any other output — sampled and monitored (File 06 §2.8).

**Infra note:** needs an embedding model + vector store + similarity service — real machinery for a cache. Worth it at FAQ scale; over-engineering below that. Say exactly that.

### 2.3 Response Caching
Exact-match caching of full responses (deterministic settings + stable inputs). Lower hit rate on free-form chat, but perfect for: tool results, fetched documents, completed agent sub-steps (File 05), anything repeated verbatim. Cheapest to implement; always add it before semantic (simple before clever).

### 2.4 Model Fallback

**Plain words:** the chosen model fails (timeout, 429 rate-limit, 5xx) → automatically retry on a second model.

**The design details that separate seniors:**
1. **Cross-provider chains:** same-provider fallback shares the outage — the chain must cross providers (File 06 §4.1).
2. **Capability-equivalent fallbacks:** prefer a fallback of similar measured quality (GPT-class → Claude-class), and PROVE equivalence with golden-set benchmarks (File 06 §2.1) — otherwise your failover silently degrades quality and nobody notices until customers do.
3. **Partial failure semantics:** what about in-flight STREAMS when a provider dies mid-answer (§1.7)? Retry idempotently with a fresh request, or reconnect — decide explicitly.
4. **Retry budget:** attempts × timeout < user's patience. A 3-attempt chain of 30s timeouts is a 90-second user experience — cap the total, fail fast, degrade gracefully (File 06 §4.6).

### 2.5 Provider Routing

**Plain words:** one request, many providers — choose by policy: cheapest / fastest / most capable / residency-compliant.

**The full mechanics:** the gateway holds provider credentials, normalizes request/response formats (per-provider transformation specs — even Anthropic's SSE events get mapped to OpenAI-style chunks), routes per policy, and fails over on health. Policies compose: "residency-compliant AND cheapest, failing over to fastest."

**Why this layer wins architecturally:** apps write to ONE API; providers become swappable details. Price negotiations, new-model onboarding, outage failover, per-tenant residency rules — all handled centrally without app changes. This is the "AI Gateway" case study (File 07) in one paragraph.

---

## Part 3: Infrastructure (the tools layer)

### 3.1 vLLM — the default open serving engine

**Full picture:** vLLM is what most companies deploy when self-hosting. The three pillars to recite:
1. **PagedAttention** (§1.9) — KV cache in OS-style pages; on-demand allocation; prefix sharing.
2. **Continuous batching** (§1.8 / File 02 §2.6) — requests join/leave at each decode step; no idle seats.
3. **Prefix caching** — identical prompt prefixes computed once, reused across requests (the serving-side twin of provider prompt caching, §2.1).

Plus: tensor-parallel serving of big models, quantization support, and an OpenAI-compatible API (drop-in for apps written against OpenAI). Reported: several-fold throughput vs naive serving. "We self-host with vLLM" should trigger those three mechanisms.

### 3.2 NVIDIA Triton Inference Server

**Full picture:** the general-purpose model server — any framework (PyTorch, ONNX, TensorFlow, sklearn), CPU or GPU, with two flagship features: **dynamic batching** (collect requests arriving in a small window, execute as one batch — a few ms of added latency for big throughput) and **pipeline/ensemble graphs** (preprocess → model → postprocess deployed as one unit). Use Triton when serving heterogeneous model fleets; use vLLM when it's purely LLMs (the ecosystem nests — Triton can run vLLM as a backend).

### 3.3 Kubernetes GPU Workloads

**The mechanics, piece by piece:**
- **Device plugin:** the nvidia device plugin advertises GPUs as a schedulable resource; pods request `nvidia.com/gpu: 1`; the scheduler binds pods only to nodes with free GPUs. Consequence: GPU pods are NODE-COUPLED — node autoscaling and pod autoscaling must coordinate.
- **MIG (Multi-Instance GPU):** slice one physical A100/H100 into isolated hardware instances (e.g., 7 × ~10GB slices with their own memory and compute). The right answer for "how do I run small models on big-GPU nodes" — a 1B model wasting a full 80GB GPU is pure waste; seven small models sharing one GPU via MIG is the fix. Time-slicing (the older alternative) shares without isolation — MIG is the clean split.
- **The pains (know them — they're real):** GPU node pools are scarce (quota, procurement lead time); pods must land on GPU nodes; cold start is minutes (§1.3); GPU pods are expensive to over-provision.

### 3.4 GPU Scheduling (beyond default K8s)

At fleet scale, GPUs are a shared, expensive resource needing real scheduling policy:
- **Gang scheduling:** all replicas of a multi-GPU model must start TOGETHER or not at all — half-started tensor-parallel groups hold GPUs while doing nothing.
- **Priority preemption:** batch inference yields to latency-sensitive serving — the scheduler evacuates low-priority pods when serving needs capacity.
- **Bin-packing:** consolidate workloads to fill nodes completely — a node with 1 of 8 GPUs free serves nothing and strands that GPU.
- **Organizational layer:** companies run GPU allocation like cloud cost centers — per-team quota, showback/chargeback, utilization reviews. Mentioning ANY of this signals production awareness: GPUs are scheduled like the scarce shared resource they are.

---

## Part 4: AI Economics & Capacity Estimation (learn to do this on a whiteboard)

### 4.1 The ingredients (each with its meaning and how to estimate it)

- **Tokens per Request** — input (prompt + context) and output separately; priced differently. Estimate from the USE CASE: chat ≈ 500-in/300-out; RAG ≈ 3–8k-in/300-out (context-dominated); agents accumulate tens of k per SESSION (multi-turn × multi-tool).
- **Requests per Day** — daily total drives COST; peak-hour rate drives CAPACITY. Both needed: `daily ÷ 86,400 = average; peak ≈ 3–5× average` for consumer products.
- **Input Tokens** — the prompt side: system prompt + instructions + retrieved context + conversation history. Priced LOW (say $3/M) but consumed in BULK in RAG/agent workloads (thousands per request). Estimate per use case: chat ≈ 500, RAG ≈ 3–8k (context-dominated), agents accumulate across a session. Input is usually the token VOLUME problem — and the main target of token-reduction and prompt caching.
- **Output Tokens** — the completion side: the answer itself (plus any reasoning tokens the model emits before answering). Priced HIGH (3–5× input, say $15/M) AND generated slowly (one token per decode step — File 02 §1.7). Estimate: chat ≈ 300–500, capped outputs lower. Output is usually the unit-PRICE problem — and the main target of output caps.
- The asymmetry to internalize: input costs by volume, output costs by price and time. A request with 5k input + 200 output may cost MORE in input dollars than output — but the output took longer to generate. Every cost analysis separates the two; never talk about "tokens" as one number.
- **Concurrent Users** — peak in-flight requests. Little's Law: `concurrent = arrival rate × response time` (100 req/s × 3s = 300 in flight). This drives GPU count via §1.2's memory arithmetic.
- **GPU Throughput** — tokens/sec per GPU under batching; depends on model size and batch depth (single H100 on a 70B: hundreds to low-thousands of aggregate tok/s; on an 8B: several thousand).
- **Cost per Request / per Customer** — the unit-economics numbers the business tracks; your dashboards feed them.
- **Model Pricing** — per-million-token rates with input/output/cached tiers (cached at 10–25% — §2.1).
- **GPU Utilisation / Cache-Hit Ratios / Batch Utilisation** — the three efficiency ratios. EVERY optimization in this file moves one of these three numbers: utilization (fewer wasted GPU-hours), cache-hit (fewer recomputed tokens), batch depth (amortized memory traffic). When someone proposes an optimization, ask which ratio it moves and by how much.

### 4.2 The six cost-optimisation levers (the PDF's exact list — memorize all six together)

1. **Model Routing** — send easy traffic to small/cheap models, hard traffic to large models; cascade when uncertain (§1.5). Usually the biggest lever — often 60–70% of traffic is simple.
2. **Smaller-Model Fallback** — the same lever on the resilience path: when the big model fails or is rate-limited, fall back to a cheaper capable one (doubles as outage protection — File 06 §4.2).
3. **Prompt Caching** — cache shared prefixes at 10–25% of full price (§2.1). Requires static-content-first prompt design; free money for template-heavy apps.
4. **Semantic Caching** — serve semantically-similar FAQ queries from cache (§2.2). Free hits on repetitive traffic; risky on nuanced queries.
5. **Token Reduction** — fewer tokens in and out: trim retrieved context (rerank to top-4 instead of stuffing top-20 — File 04 §2.4), summarize stale history, cap output length, compress verbose system prompts.
6. **Batching** — self-hosted only: batch requests to raise tokens/sec per GPU (File 02 §2.5). Doesn't reduce hosted-API spend, but increases GPU throughput per unit cost.

**How to present them in an interview:** never list levers — apply them BY MEASURED IMPACT: measure cost per use case first, then apply the lever that moves the biggest number, then re-evaluate quality (§4.3's worked model does exactly this).

### 4.3 The worked cost model (memorize the SHAPE; adapt the numbers to whatever the interviewer gives)

**Scenario: RAG assistant, 1M requests/day, 3k input + 400 output tokens. Pricing: $3/M input, $15/M output.**

Step 1 — daily tokens: input 3B, output 0.4B.

Step 2 — daily cost:
- Input: 3,000M × $3 = $9,000/day
- Output: 400M × $15 = $6,000/day
- **Total: $15k/day ≈ $450k/month.** (Say the number out loud — the shock is part of the answer's rhythm.)

Step 3 — apply the six PDF levers, with realistic effects (each lever, expected magnitude):
1. **Model routing** — 70% of queries are simple → small model at ~1/10 price on that slice → that 70% of spend drops ~65%. Usually the biggest lever.
2. **Smaller-model fallback** — the same lever as the fallback path (doubles as resilience — File 06 §4).
3. **Prompt caching** — 2k of the 3k input is a fixed RAG preamble at ~20% price: per-request input cost ≈ (2k×$3×0.2 + 1k×$3)/1M ≈ $0.0108 vs $0.012 — AND TTFT collapses for those tokens.
4. **Semantic caching** — 30% of traffic is FAQ-similar → served free.
5. **Token reduction** — rerank and cut context 3k → 1.2k (fewer, better chunks — File 04 §2.4); cap outputs at 300.
6. **Batching** — self-hosted only: raises tokens/sec per GPU (throughput per GPU-hour), doesn't change hosted API spend.

Step 4 — combined realistic outcome: **$450k → ~$120–160k/month**, with quality held (see step 5).

Step 5 — the "without materially hurting quality" half (what makes it an engineering answer, not an accounting one): golden-set evals before AND after every lever (File 06 §2.1); canary rollout with auto-rollback on quality metrics; report cost AND quality in the same sentence ("cost −68%, groundedness −0.4%"). Never present cost cuts without the quality proof — that's how optimizations become slow quality death marches.

This exact exercise is the course's flagship drill — and it's fully rehearsed in File 07, Case Study 6.

### 4.4 Self-hosted capacity math (the other direction)

```
GPUs needed ≈ (peak concurrent requests × KV per request) ÷ (usable GPU memory − weights)
```
Then sanity-check tokens/sec throughput (concurrent × avg TPS demand ≤ fleet throughput), and compare monthly amortized GPU cost vs hosted API cost at your volume — the crossover decision (File 02 §3.4) made numeric. At 20k requests/day, hosted is obviously right; at 5M/day with steady patterns, self-hosted usually wins IF utilization stays high — which is why §3.4's scheduling and Part 4's utilization ratio exist.

---

## Part 5: Advanced Distributed AI Infrastructure

### 5.1 Distributed Training (concepts)

**Data Parallelism — the default:** every GPU holds a FULL copy of the model; each processes a different slice of data; after each step, gradients are averaged across all GPUs (the "all-reduce" collective). Simple, scales well — but the model must FIT on one GPU. Training LLMs would be impossible for one GPU → the next two patterns exist.

**Tensor Parallelism — splitting the math:** ONE layer's matrix multiply is SPLIT across GPUs — each holds a slice of every weight matrix; partial results combine at each layer boundary. Fine-grained communication every layer → needs NVLink-class interconnects → used WITHIN a node. Analogy: four people each multiplying different columns of the same spreadsheet, then stapling the results.

**Pipeline Parallelism — splitting the layers:** the model is cut by LAYERS — GPU1 holds layers 1–20, GPU2 holds 21–40, like an assembly line. One request flows through stages. Communication happens only at stage boundaries → less chatter → works ACROSS nodes. Weakness: "bubbles" — stage 2 idles while stage 1 processes, until you feed multiple micro-batches to keep all stages busy.

**The sentence to say (one sentence, full marks):** *"Tensor parallel within a node, pipeline parallel across nodes, data parallel on top."* That's how frontier training clusters are laid out.

**Distributed Checkpoints:** training state saved across ALL GPUs such that any node dying mid-run costs only the work since the last checkpoint. Multi-week runs MUST survive hardware that doesn't — checkpoint frequency is the trade (checkpoint overhead vs re-work on failure).

### 5.2 Distributed Inference

- **Model parallelism (inference):** a model bigger than one GPU is split at SERVE time — tensor/pipeline parallel among the replica's GPUs (70B FP16 = 140 GB → 2–8 GPUs acting as one replica). The cost: every token crosses GPU boundaries — inter-GPU communication adds latency; a parallel replica is SLOWER per token than a model fitting on one GPU. Parallelism buys capacity, not speed.
- **Request scheduling:** which request goes to which replica/partition — load balance across replicas, KV-aware placement (route requests to the replica already holding their prefix), tenant affinity where isolation demands it.
- **Dynamic batching:** the server collects requests arriving within a small window and forms them into one batch before execution — Triton's classic feature; the serving-side cousin of continuous batching.
- **KV-cache distribution:** with model parallelism, a single request's KV blocks live across multiple GPUs (each holds its layers' share). And prefix-cache hits now require the right ROUTE as well as the right cache: a request whose prefix is cached on replica 3 must LAND on replica 3 — consistent hashing on the prompt prefix keeps hit rates high.
- **Disaggregated prefill/decode (the frontier concept):** prefill is compute-bound; decode is memory-bound (File 02 §2.3) — so run them on SEPARATE hardware pools, each optimized for its job: compute-heavy nodes for prefill, bandwidth-optimized nodes for decode, with the KV cache handed off between pools. Early in industry adoption; knowing the idea AND its why (the two phases stress hardware differently) signals current-awareness.

### 5.3 Flagship: Design ChatGPT at Scale (condensed — full walkthrough in File 07)

Client → gateway (auth, token rate limits) → model router (small/large, cascade) → serving tier (vLLM-class: continuous batching, PagedAttention, prefix caching) → SSE streaming back → guardrails + metering at every hop → multi-region, provider fallback. The model is stateless; conversation state lives in storage. Cost governed by Part 4's levers. Every component of that sentence is a section of this file — that's why this track is the backbone of AI system design interviews.

---

## Tough Interview Follow-ups (with full answers)

**Q: "Requests queue even though GPU utilization shows 40%. Why?"**
A: "GPU utilization" is about COMPUTE. LLM serving is frequently memory-bound: KV cache is full, so the scheduler can't admit new requests even with compute headroom (§1.2). Fixes: quantize weights (more room for KV), cap context, PagedAttention (recover the 60–80% preallocation waste), or add replicas. Also check the queue isn't POLICY — per-tenant quotas (§1.10) also queue requests, by design. Diagnose which before acting.

**Q: "Walk me through cutting inference cost 40% — your first week, in order."**
A: Days 1–2: MEASURE — cost per request by use case × model × tenant; input/output/cached split; cache-hit rates. Days 3–4: apply levers by measured impact — route easy traffic to small models (usually biggest), fix prompt caching (static content first — a common silent miss), trim RAG context via reranking, cap output tokens, semantic cache for FAQ. Day 5: evals before/after each change (golden sets), canary rollout, dashboard cost AND quality side by side. Never cut without measuring quality in the same change — and set burn-rate alerts so savings don't erode (§4.1's ratios).

**Q: "Why is GPU autoscaling not like CPU autoscaling?"**
A: Four differences: scarcity (node-bound, quota-limited), minutes-long cold start (weights), 10–100× cost (flapping burns money), and the unique cushion — batching headroom (replicas absorb spikes by batching harder, trading slight latency for throughput before new replicas are needed — §1.6). So: scale on queue depth/TTFT, hysteresis, warm pools — not plain CPU-style HPA.

**Q: "Why does your gateway exist — apps could just call OpenAI?"**
A: Five reasons, on fingers: (1) cost control — per-team token quotas and metering stop runaway spend; (2) provider abstraction — swap/add providers without app changes; (3) safety in ONE place — guardrails on every call regardless of which team wrote the app; (4) uniform observability — tokens, latency, cost per team; (5) routing policy — cheap/capable/resident per request. And it's how you survive provider outages (multi-provider fallback). 

**Q: "One tenant's batch job degrades another tenant's chat users. Fixes, in order?"**
A: (1) Per-tenant quotas/token buckets — the ceiling; (2) priority queues — real-time over batch; (3) schedule batch jobs off-peak; (4) if contention persists, dedicated capacity pools for latency-sensitive vs batch workloads. Then verify with per-tenant TTFT dashboards — the noisy neighbour is only real if the metrics show it (§1.11).

**Q: "You're at 10M tokens/day and growing 10%/month. When do you move from hosted to self-hosted?"**
A: Build the crossover math (§4.3): monthly hosted spend vs amortized GPU cost at your concurrency — but three gates before committing: (1) utilization — can you keep GPUs >60–70% busy around the clock? Idle GPUs make self-hosting MORE expensive; (2) ops maturity — serving stack, upgrade cadence, on-call (File 02 §3.4's ledger); (3) traffic steadiness — spiky traffic wastes self-hosted capacity; steady baselines suit it. The pragmatic path: self-host the steady baseline on an internal platform, keep hosted for bursts — hybrid via the gateway.

---

## Rapid-Fire Flashcards

1. GPU memory = ? → weights + KV cache × concurrent + activations.
2. Concurrency limited by? → MEMORY, not compute — the punchline of §1.2.
3. Three levers on concurrency? → quantize weights, cap context, PagedAttention.
4. Queue wait is part of? → user-perceived latency (and TTFT).
5. Model routing dimensions? → capability, context fit, cost, residency; cascade = small-first-escalate.
6. GPU autoscaling differences? → scarce, slow cold start, costly, batching headroom; scale on queue/TTFT with hysteresis.
7. Token-based rate limiting? → tokens/min + in/out split + context cap + concurrency, per tenant; enforce pre-call, meter post-call.
8. Noisy neighbour fix ladder? → quotas → priorities → off-peak → separate pools.
9. Prompt caching rules? → static first, byte-stable, prefix-only matching; ~10–25% price on cached tokens.
10. Semantic caching? → embed + similarity threshold; FAQ gold, nuance landmine; high thresholds, narrow scope.
11. Model fallback? → cross-provider, capability-equivalent, parity-proven, budgeted retries.
12. Provider routing? → cheapest/fastest/capable/resident policy at a gateway; composes with failover.
13. vLLM = ? → PagedAttention + continuous batching + prefix caching.
14. Triton = ? → any-framework model server; dynamic batching + pipeline graphs.
15. MIG? → one GPU sliced into isolated instances for small models.
16. Gang scheduling? → multi-GPU replicas start all-or-nothing.
17. Little's Law? → concurrent = arrival rate × response time.
18. Output token price? → 3–5× input.
19. Six cost levers? → routing, smaller-model fallback, prompt caching, semantic caching, token reduction, batching.
20. Three efficiency ratios? → GPU utilization, cache-hit rate, batch utilization — every optimization moves one.
21. Parallelism trio? → data (model copies), tensor (split layer math, in-node), pipeline (split layers, cross-node).
22. Big-model rule? → TP within node, PP across nodes, DP on top.
23. Disaggregated prefill/decode? → separate pools per phase, KV handoff — frontier.
24. Cost exercise shape? → tokens/day × price → levers → evals prove quality held → ratios keep it held.

---
