# 02 — LLM System Design (Track 12)

> **Goal:** Understand what an LLM actually IS and how it generates text — deeply enough to explain tokens, attention, prefill/decode, and KV cache on a whiteboard. These are the most-asked LLM fundamentals in AI engineer interviews.

---

## Part 1: LLM Fundamentals

### 1.1 Tokens

**Plain words:** An LLM cannot read letters or words. It reads **tokens** — small chunks of text, roughly ¾ of a word in English.

**The story that explains why tokens exist (told like the spam-filter story in File 01):**

You're building a translator between English and a machine. You must turn text into numbers. Option A: **one number per letter**. 'c'=3, 'a'=1, 't'=20. It works... but look at what you've done: the model now must learn meaning from LETTERS — and letters carry almost nothing. "cat" and "car" differ by one letter. A page becomes thousands of meaningless units, and any relationship between distant parts of the page must stretch across all of them. Option B: **one number per word**. Better units! But now count your dictionary: English has hundreds of thousands of words — plus names ("Nikhil"), typos ("refnud"), slang, code identifiers (`getUserBalance`), and every word of every other language. Any word missing from your list is literally UNREADABLE. A model serving 2026's internet with a frozen 2019 vocabulary would choke on "rizz" on day one.

So both pure options fail — and the compromise that won (subword chunks) is beautifully simple: keep a vocabulary of ~100k units where common words stay whole ("the", "hello", "computer"), rare words decompose into meaningful pieces ("unbelievable" → "un" + "believ" + "able"), and ANY text — new words, typos, Hindi, code — can ALWAYS be written as some sequence of known chunks, the same way you can spell any word with 26 letters. You never hit an unreadable input, and you never waste vocabulary slots on rare words.

**Now see it in action (trace these out loud once):**
- `"Hello world"` → ["Hello", " world"] = 2 tokens
- `"unbelievable"` → ["un", "believ", "able"] = 3 tokens — a 12-letter word, only 3 units
- `"strawberry"` → ["str", "aw", "berry"] = 3 tokens — three chunks, NOT ten letters
- `"getUserBalance()"` → ["get", "User", "Balance", "()"] = 4 tokens
- A 1,000-word essay ≈ 1,300 tokens

**The strawberry story (the single best example of why tokenization matters):** in 2023, people asked ChatGPT "how many r's are in strawberry?" and it answered **2**. The internet exploded: how can a "genius" model fail at counting letters? Now you know exactly why: the model never SEES letters. It sees "berry" as ONE opaque unit — token #47281, one number. To count r's it must RECALL, from training data, how that chunk is spelled — memory, not observation. And memory can be fuzzy. Newer models fix this by "thinking step by step": they write the word out letter by letter ("s-t-r-a-w-b-e-r-r-y") — which turns one opaque token into ten simple tokens, each of which IS a single letter they can actually look at. The lesson interviewers love: **tokenization is not a cosmetic detail — it shapes what models can and cannot do.**

**Why engineers care more about tokens than words (four consequences):**
1. **Pricing** — providers charge per token, not per word or per request. Every cost conversation is a token conversation.
2. **Limits** — context windows are measured in tokens (§1.3).
3. **Speed** — generation happens token-by-token; response time scales with OUTPUT token count (§1.7).
4. **Capacity math** — every cost/latency estimation begins with token counts (File 03 Part 4).

**Numbers to memorize (anchor everything to these):**
- 1 token ≈ 4 characters of English ≈ ¾ of a word
- 1,000 tokens ≈ 750 words
- 1 million tokens ≈ a full novel
- Code tokenizes worse than English: ~3–4 bytes per token vs ~4.5–5 for prose (identifiers like "getUserBalance" split heavily; whitespace runs and syntax fragments split too)
- Non-Latin scripts (Devanagari, Tamil, Thai) often cost 2–4× more tokens than English for the same meaning (§1.2)

### 1.2 Tokenization

**Plain words:** The process of chopping text into tokens and mapping each chunk to an ID number from a fixed vocabulary (e.g., 100,000 entries). The model only ever sees ID sequences; the tokenizer is the translator on both sides — input AND output.

**Full explanation — how BPE (Byte Pair Encoding) actually works, because it IS the standard:**

BPE starts with the smallest units (single characters/bytes) and repeatedly merges the most frequently-occurring PAIR into a bigger unit, keeping the best merges:

1. Start: text is pure characters. Vocabulary = the alphabet (~256 byte values).
2. Count all adjacent character pairs across the whole corpus. "th"+"e" occurs 10 million times — the most frequent pair. Merge it: add "the" as a new vocabulary entry.
3. Recount. Now "in"+"g" is the most frequent remaining pair. Merge → "ing" joins the vocabulary.
4. Repeat thousands of times. Stop when the vocabulary reaches its budget (50k–200k entries — chosen at model design time).

The result is a frequency-optimized dictionary: frequent words ("the", "and", "hello") are single cheap tokens; rare words ("zeppelin", "Rückerstattung") split into pieces; NOTHING is unrepresentable.

**Practical consequences — each one is a real interview talking point, with its business impact:**

1. **Non-English costs more.** English dominates training corpora, so the merges were learned on English. Hindi in Devanagari splits into many tokens per word — a Hindi chat costs perhaps 2× the tokens of the equivalent English chat → 2× cost, 2× generation time, and worse context-window fit. Business consequence to cite: multilingual products must budget per-language, or route per-language to models with better tokenizers (a gateway feature — File 03 §2.5).

2. **Numbers tokenize unpredictably.** "1234567" may split as "123" + "45" + "67" or "12" + "345" + "67" — the model sees fragments, not the number's structure. One root cause of LLM arithmetic weakness, and one reason "think step by step" and "write code for math" prompts genuinely help: spelling digits out or computing in code sidesteps the fragmentation.

3. **Whitespace matters — more than anyone expects.** " hello" and "hello" are DIFFERENT tokens (leading space included). Chat templates (the special formatting wrapping system/user/assistant turns) must match what the model was trained on EXACTLY — one wrong space can degrade output. This is a real production bug class: teams that hand-roll prompt plumbing instead of using the provider's official template. Lesson to state: "we treat chat-template construction as library code, not string concatenation."

4. **The vocabulary is frozen at build time.** New words ("rizz", "COVID") didn't exist when older models' BPE was built — they tokenize into fragments the model reads awkwardly. Newer model generations ship updated vocabularies; this is one of the quiet reasons to upgrade model versions.

5. **Special tokens.** The vocabulary reserves special IDs for control: BOS (beginning of sequence), EOS (end of sequence — how the model "stops talking"), and roles/tags for chat formatting. The model stopping mid-rant is literally it emitting the EOS token.

**Memory hook:** *Tokenizer = the model's dictionary. Everything read or written passes through it — and what it can't express, the model can't think.*

### 1.3 Context Windows

**Plain words:** The model's **short-term working memory** — the maximum number of tokens it can consider at once, counting input AND output together.

**Full explanation — the window is a BUDGET you spend, not a feature you get for free:**

**Rule 1 — input and output share the budget.** A window of 8k with a prompt of 7k leaves at most ~1k of generation before the model literally runs out of room (it stops, or errors, depending on the provider). Budget tokens like memory: system prompt + retrieved context + history + reserve for the answer. A worked budget (memorize this pattern, you'll reuse it in every design answer):

```
Window: 8,192 tokens
- System prompt + tool definitions:   1,500
- Retrieved RAG context (top-4):      3,000
- Conversation history:              2,200
- Reserve for the answer:            1,400
- Safety margin:                        92
```
The moment history grows, something else must shrink — trimming history, or fewer retrieved chunks, or smaller reserve. This arithmetic IS the job of "context management."

**Rule 2 — cost and latency grow with usage.** Input tokens cost money per call, and longer prompts mean longer prefill → slower TTFT (§2.3). A 100k-token RAG context is real money per request and a real wait before the first word. "It fits in the window" and "it's sensible to put it there" are different claims.

**Rule 3 — quality can DEGRADE with length: "lost in the middle."** Research (Liu et al., "Lost in the Middle") showed models attend best to content at the BEGINNING and END of long contexts; information buried in the middle gets less attention and worse recall. So stuffing more context does not monotonically improve answers — past some point it actively hurts. Practical implications to recite: put critical instructions at the start; rank retrieved chunks so the best are first or last, never the middle (File 04 §2.5); prefer selective retrieval over stuffing; summarize stale history instead of hauling it along.

**Rule 4 — the window RESETS every request.** The model itself has NO memory between requests. Chat "memory" is an illusion created by the application: the ENTIRE conversation is re-sent as the prompt each turn. Turn 10 re-sends turns 1–9 — which is why long chats get expensive, and why every chat product eventually implements trimming/summarization (context management — File 04 §3.12, File 05 Part 2).

**The whiteboard story (say this in interviews — it makes Rule 4 unforgettable):** imagine the model is an expert consultant who, for EVERY question, walks into a fresh room with a whiteboard. Whatever is written on the board when you ask — that's ALL they know. They read the board, answer, and LEAVE. The room's whiteboard is wiped for the next person. Want them to "remember" the conversation? YOU write the whole conversation on the board before they walk in. Turn 10 means writing turns 1–9 plus the new question — every single time. If the board (8k tokens) fills up, you — the engineer — choose what stays: erase the stale middle (summarize history), leave only the last few exchanges verbatim, or move details to a filing cabinet in the corner (a vector store) and pin a sticky note "ask the cabinet if you need the 2024 pricing." That choice — what stays on the board — is the entire craft of context management. The model is brilliant; the memory is YOUR infrastructure.

### 1.4 Prompt vs Completion

**Plain words:**
- **Prompt** = everything that goes IN — system prompt, conversation so far, retrieved context, the new question.
- **Completion** = what the model GENERATES as a continuation of that text.

**Full explanation — the deep point that explains three behaviors at once:** LLMs are completion engines. Given a text prefix, they predict a plausible continuation. That's ALL they do. Chat works because models were TRAINED on conversation-formatted data — so "continue this conversation" produces helpful, on-role replies. Three consequences follow directly:

1. **Prompt injection vulnerability** (File 05 §5.3): instructions hidden in data look identical to instructions in the prompt — to the model it's one text stream to continue. It can't structurally tell "my owner's instructions" from "text I was asked to summarize."
2. **Stop tokens:** the model ends its turn by emitting a special stop token — or when YOU force a stop (max_tokens). "Running out of tokens" (§1.3 budget) manifests as mid-sentence truncation.
3. **Everything is the prompt:** system prompt, retrieved documents, tool results, user text — all serialized into one sequence. That's why ordering matters (§1.3 lost-in-middle), why prefixes can be cached (File 03 §2.1), and why "the prompt" in an architecture diagram is really a whole assembly pipeline (the context builder — File 04 §2.5).

### 1.5 Streaming

**Plain words:** Instead of waiting for the whole answer, the server sends each token the moment it's generated. Live cricket vs. highlights after the match.

**The story that sells it (feel the numbers):** a 500-token answer at 50 tokens/second takes 10 seconds. Watch a user's face in each world: without streaming, they stare at a spinner for 10 seconds — by second 3 they think it's broken; by second 7 they're re-submitting. With streaming, words start appearing at 0.2 seconds and keep flowing — the user READS as it generates. Same total time, but the first version feels 5–10× slower than the second. Perceived latency is almost entirely TIME TO FIRST TOKEN — streaming is how you buy that. No streaming in a chat product is not a "small UX issue"; it's shipping a product that feels broken while working correctly.

**The engineering implications, one by one:**

1. **Transport:** SSE (Server-Sent Events) over HTTP is the standard — a long-lived HTTP connection where the server pushes `data:` frames. WebSockets when you need bidirectional (voice, multi-party). The wire format for OpenAI-style streaming: each SSE frame carries one chunk (one or a few tokens + role/finish metadata), ending with a `data: [DONE]` sentinel.

2. **The buffering trap (the classic silent production bug — a war story every team lives once):** the team ships streaming. Demo works perfectly — tested against the model API directly. In production... the spinner spins for 8 seconds, then the whole answer appears at once. Nothing "failed." The culprit: a reverse proxy/load balancer between the app and the user, doing its standard job of collecting the complete response before forwarding (great for HTML pages, fatal for streams — the whole point of the stream is delivery AS IT GENERATES). Fix: disable response buffering on streaming routes (X-Accel-Buffering: no, flush zones). The lesson: test streaming THROUGH THE FULL PATH — proxy, LB, CDN, gateway — not just direct-to-model. If a stream "doesn't stream," suspect the middleboxes first.

3. **Downstream complexity:** output validation/moderation now runs on streams — you cannot fully validate a response you haven't fully received (File 06 §3.8 handles this: guardrails on completion, or incremental checks on accumulated text). Error handling changes: what if the stream dies at token 300? Clients need reconnect and partial-response handling; billing must handle partial outputs.

4. **Timeout semantics change:** a 30s response-timeout is wrong for a stream — the right pattern is a per-chunk timeout ("no token for 5 seconds = dead") plus a total wall-clock cap. Timeouts become two-dimensional.

5. **Backpressure:** if the client reads slowly (mobile on bad network), server-side buffers fill; well-designed streaming stacks apply backpressure rather than buffering unboundedly. (Backpressure in one sentence: bounded buffers fill → the producer is made to wait → generation slows to the slowest reader, instead of memory growing until the server dies. TCP receive windows already do this at the transport layer; your app must not defeat it with "helpful" buffering.)

### 1.6 Time to First Token (TTFT)

**Plain words:** How long until the user sees the first word. THE headline latency metric for chat UX.

**Full explanation — what composes it, with a worked breakdown:**
```
TTFT = network + queue wait + prefill time (processing your ENTIRE prompt)
```
The prefill component is proportional to prompt length: a 10,000-token prompt must be fully processed before the FIRST output token can exist (§2.3 explains why). The queue component is pure infrastructure: if all model replicas are busy, your request waits, and that wait lands squarely in your TTFT (File 03 §1.4).

Worked example — same model, two different prompts:
- Short chat turn: 600-token prompt, prefill at ~5,000 tok/s → ~120ms + queue ~0 → TTFT ≈ 200ms. Feels instant.
- Heavy RAG turn: 8,000-token context → prefill ~1.6s + queue → TTFT ≈ 2s. Feels sluggish — before a single word is generated.

**The levers to reduce TTFT (recite this list — it's a complete answer):**
1. Shorter prompts — trim context, fewer/tighter retrieved chunks (File 04 §2.5).
2. Prompt/prefix caching — shared prefill computed once (File 03 §2.1); biggest lever for template-heavy apps.
3. Smaller model for prefill-heavy simple tasks — prefill cost scales with model size.
4. More replicas / better scheduling — attacks the queue component (File 03 §1.4, §1.6).

### 1.7 Tokens Per Second (TPS)

**Plain words:** Generation speed once the first token is out.

**The full latency decomposition (memorize — this ONE equation structures all LLM latency debugging):**
```
Total response time = queue wait + TTFT (prefill) + (output_tokens ÷ TPS)
```

**The key insight — TTFT and TPS have DIFFERENT bottlenecks:**
- TTFT ← prompt length, queue wait, prefill compute.
- TPS ← model size, GPU memory bandwidth, batching (§2.3–2.5).

So "the app feels slow" decomposes into a diagnosis: long prompt with instant answers → prefill problem (shorten prompts, cache prefixes, more replicas). Short prompt with slow generation → decode problem (smaller model, more batching headroom). Two different problems, two different fix lists — and mixing them up is how teams waste weeks on the wrong lever.

**Worked example (do this on a whiteboard):**
- Prompt: 2,000 tokens. Prefill speed: 4,000 tok/s → TTFT = 0.5s.
- Answer: 400 tokens. TPS: 50 → generation = 8s.
- Total ≈ 0.5 + 8 = 8.5s — generation dominates → fix decode (smaller model, tighter output caps).
- Now a RAG turn: prompt 8,000 tokens → TTFT 2s; answer 300 tokens at 50 TPS = 6s → total 8s — now prefill is 25% of the budget → fix context length too.

**Anchor numbers:** small models on good GPUs: 100+ TPS per stream. 70B-class: 20–60 TPS per stream unbatched — but batching many users onto one GPU recovers aggregate throughput enormously (§2.5). These anchors let you sanity-check any "our latency will be X" claim in a design discussion.

---

## Part 2: Model Execution — how the LLM actually generates text

### 2.1 The core mechanism: next-token prediction

An LLM does exactly one thing: given the tokens so far, output a probability distribution over the whole vocabulary for the NEXT token. Sample one (or pick the top one). Append it. Repeat until a stop token. That loop is **decoding**. Chat, reasoning, code — all of it emerges from this one repeated operation plus a lot of training.

**The sampling controls you must know (they turn this mechanism into product behavior):**

- **Temperature** — how adventurous the sampling is. Mechanically: temperature divides the logits before softmax. At 0 (greedy): always pick the highest-probability token → near-deterministic output; right choice for factual/extraction/classification tasks. At 0.7–1.0: probability mass spreads → varied, creative, but riskier for facts. Debugging corollary to state: "same prompt, different answers" is usually temperature > 0 — and an LLM API is never a deterministic contract even at 0 (numerical non-determinism across hardware/batches exists).
- **Top-p (nucleus sampling)** — sample only from the smallest set of tokens whose cumulative probability reaches p (e.g., 0.9) — cuts the long tail of nonsense tokens while keeping variety. Often used WITH moderate temperature.
- **Top-k** — hard limit to the k most likely tokens (a cruder cousin of top-p).
- **Max tokens** — hard cap on output length. Also a COST control (File 03 Part 4): unbounded outputs are unbounded spend.

**Worked sampling example (do this once — it makes temperature/top-p concrete forever).** The model's next-token distribution after "The refund was processed in 3–5":

| token | probability |
|---|---|
| business | 0.42 |
| working | 0.31 |
| business- | 0.09 |
| calendar | 0.06 |
| weeks | 0.04 |
| (8,000 more tokens) | 0.08 total |

- **Temperature 0:** always "business" — same output every time. Right for a support bot answering a factual question.
- **Temperature 0.7:** probabilities flatten somewhat; "working" now wins maybe 1 time in 3 — natural phrasing variety, still on-task.
- **Temperature 1.4:** "weeks" and weirder tokens start winning — creative writing only.
- **Top-p 0.9:** cut everything after the cumulative 0.9 mark ("business"+"working"+"business-" = 0.82; adding "calendar" reaches 0.88; "weeks" crosses 0.92 — so top-p 0.9 keeps roughly the first four and never samples "weeks" or the 8,000-token tail). Notice what top-p does: it ADAPTS the cut to the distribution — a confident model (one token at 0.9) keeps one option; an uncertain model keeps many. That adaptiveness is why top-p beat top-k as the default.

### 2.2 Attention — the engine inside

**First — what a Transformer actually is (the 2-minute primer that frames everything below):**

A Transformer is a stack of identical blocks (a "layer"), repeated N times (e.g., 32–100 layers in modern LLMs). Each block does exactly two things:

1. **Attention** (§2.2 below) — every token LOOKS AT all previous tokens and pulls in what's relevant. This is how context flows BETWEEN positions ("it" learns it means "the animal").
2. **Feed-forward network (FFN)** — each token's representation is then processed INDEPENDENTLY through a small neural network (two matrix multiplies with a nonlinearity). This is where the model's stored KNOWLEDGE mostly lives — the FFN acts as a giant learned memory/lookup: "given what this token now means in context, what should I add to it?"

So one layer = *communicate (attention) → think (FFN)*. Stack 80 of those, and each token's representation gets progressively refined: early layers capture syntax and simple patterns, middle layers build up meaning and relationships, late layers sharpen toward "what's the next token going to be?" At the top, a final projection turns the last token's representation into a probability over the whole vocabulary — that's the "next-token distribution" from §2.1.

Why this architecture won: (1) attention connects ANY two positions directly (one hop — no information decaying across a long chain, unlike older recurrent designs), enabling long-range understanding; (2) everything is matrix multiplications — perfectly parallel on GPUs (File 03 §1.1), unlike sequential recurrence; (3) it scales — more layers, more heads, more data keeps improving, which is the entire basis of the LLM era. **Everything else in this file — prefill/decode, KV cache, batching, quantization — is engineering around this one architecture's math.**

**Plain words:** Attention lets each token "look at" every previous token and decide which ones matter for predicting the next one.

**The classic example first (say this one in interviews):** "The animal didn't cross the street because **it** was too tired." What is "it"? The animal — because tired animals make sense and tired streets don't. When the model processes "it", attention connects it strongly to "animal" (high weight) and weakly to "street" (low weight). That IS attention: learned, dynamic relevance between positions — different for every token, every sentence.

**The Q/K/V mechanism, step by step (whiteboard-ready):**

Every token's representation is projected into three vectors:
- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I contain / how can I be found?"
- **Value (V)** — "the information I actually hold"

The library analogy, fully: every book has an index card (K) describing what it's about. Your search query (Q) is matched against all index cards. The better the match, the more of that book's content (V) you take home. You leave with a BLEND — mostly the best-matching books, a little of the others.

**Now watch it work as a story — resolving "it" in our example sentence (follow the four steps for the token "it"):**

The sentence so far: "The animal didn't cross the street because it". The model must now understand the word "it" — and its meaning is ambiguous until attention resolves it. Four steps happen INSIDE one attention head:

1. **Score:** "it" projects its Query vector — essentially "*I'm a pronoun, I'm looking for my referent — likely a recent singular noun*." Every previous token's Key gets matched against it via dot product. "animal"'s Key ("I'm a singular noun, an agent capable of being tired") matches strongly → high score. "street"'s Key ("I'm a noun, a surface, not capable of tiredness") matches weakly → low score. "the", "because", "cross" — near zero.
2. **Scale:** scores get divided by √d — a purely numerical trick so the softmax in step 3 doesn't saturate (skip the detail unless asked; keep the flow).
3. **Softmax:** raw scores become percentages that sum to 100% — the attention spotlight: "animal" 72%, "street" 4%, "tired"-candidates and the rest sharing the remainder.
4. **Blend:** "it"'s new representation = 72% of animal's Value + 4% of street's Value + scattered bits. The token "it" has now literally BECOME mostly-animal in the model's internal representation — the ambiguity is resolved, and every later layer works with a token that "means" the animal.

That's all attention is — repeated for every token, every layer, every head: *score everything against me, convert to percentages, absorb a weighted blend.* The full formula if you can write it: `softmax(QKᵀ/√d) · V`.

**Multi-head attention — why many small attentions beat one big one:** run several small attentions in parallel (e.g., 32 heads, each with smaller vectors), each with its own learned Q/K/V projections. Different heads learn different relations — empirically, some heads track pronoun references, others syntax agreement, others long-range topic. Outputs concatenate and mix. Like 32 specialists each reading the sentence for a different purpose, then comparing notes. This is also why attention cost scales with (context length)² — every token attends to every token — the quadratic cost that makes long contexts expensive (§1.3).

**Causal (left-to-right) masking:** during generation, each position may only attend to positions BEFORE it — a triangular mask zeroing all "future" weights. Necessary for training (the model mustn't peek at the answer) and it's why generation is inherently left-to-right sequential — the constraint that creates the decode bottleneck next.

### 2.3 Prefill vs Decode — the two phases of every request

This distinction unlocks half of Track 13. Internalize both phases, their hardware characters, and their consequences.

**The story first — one request, watched as a movie (follow it start to finish):**

A user asks a chatbot a question with a 2,000-token prompt. Press play.

- **The reading phase (prefill):** all 2,000 prompt tokens enter the model AT ONCE, together, in one giant parallel sweep. The GPU — thousands of cores doing the same operation on different data — devours it. For our 2,000 tokens this takes maybe half a second. While reading, the model does something crucial for later: for EVERY token it computes and saves those K and V vectors (§2.2) — like a student reading the exam paper and TAKING NOTES on every line, because those notes are what they'll consult while writing the answer. This phase IS your TTFT — the user sees a spinner until prefill completes.

- **The writing phase (decode):** now the answer must be written — and here the movie changes genre. Token 1 comes out... then token 2... then 3. Each single token is a full forward pass through the whole model — and each one NEEDS the previous token to exist first (causal masking: you can't predict the 4th word before the 3rd exists). The brilliant parallel machine is suddenly forced to work one-word-at-a-time, like a racehorse pulling a rickshaw. The student consults their NOTES (the KV cache — §2.4) instead of re-reading the paper — that's what makes each step fast at all.

- Why is writing so much slower than reading? Reading 2,000 tokens in parallel ≈ 0.5s; writing 300 tokens one-at-a-time ≈ 6s. Prefill uses every core at once; decode can't be parallelized — token N waits for token N−1. Same GPU, same model, 10× difference — purely because of the shape of the work.

**Now the two phases, formally:**

**Prefill (once, at request start):**
- Your ENTIRE prompt is processed — all tokens simultaneously, in parallel. GPUs are massively parallel machines; this phase is what they were built for.
- Computes and stores the K and V for every prompt token → the KV cache (§2.4).
- **Compute-bound:** big matrix multiplications; GPU compute is the limit. This phase IS your TTFT (§1.6).

**Decode (repeats per output token):**
- ONE new token per forward pass: the new token attends to all previous tokens (via the KV cache), producing the next-token distribution. Sample. Append. Repeat.
- **Serial by nature** (causal masking — token N needs N−1 to exist).
- **Memory-bandwidth-bound, NOT compute-bound** — the counterintuitive fact below, and the single most valuable thing to understand in this file.

**The non-obvious insight (major interview gold) — why decode is memory-bound:** to generate ONE token, the entire model's weights must stream from GPU memory to the compute units. The arithmetic units do one small multiply per weight — they sit mostly IDLE waiting for data. You pay a full pass over billions of weights to produce ONE tiny token. So decode per-token speed is limited by how fast memory feeds compute, not by how much compute exists.

**And this fact IS the reason batching works (§2.5):** with 50 concurrent requests, the weights are read from memory ONCE per step and serve 50 token generations simultaneously — the memory traffic is amortized 50×. Serving 1 user vs 50 users costs nearly the same memory traffic; the GPU's compute finally has something to do. This one insight explains continuous batching, GPU economics, and why throughput scales with batch size — all of Track 13's serving machinery follows from it.

**Memory hook:** *Prefill = read the whole question in one gulp (parallel, GPU-friendly). Decode = write the answer one word at a time (serial, memory-hungry).*

### 2.4 KV Cache — the trick that makes LLMs usable

**The story first — watch the disaster without it:** you're generating token #500 of an answer. That token needs ATTENTION over the previous 499 tokens (§2.2) — and attention needs each one's K and V vectors. Without any cache, here's your per-token reality: to write word 500, recompute K/V for words 1–499. Then for word 501... recompute K/V for words 1–500 AGAIN. Word 502: again. Each step re-does the ENTIRE history. Count the total work: 1 + 2 + 3 + ... + N ≈ N²/2 — for a 1,000-token answer, ~500,000 computations of which ~99% are pure re-doing of work you already did. The model would be unusably slow — this isn't an optimization, it's THE thing that makes generation viable at all.

**The fix (and it's beautifully simple):** while READING the prompt (prefill — §2.3), you computed K/V for every prompt token anyway. DON'T THROW THEM AWAY — store them in GPU memory. Now, writing word 500 doesn't recompute anything: it computes K/V only for the NEW token (one small increment) and reuses all 499 stored vectors. Total work drops from quadratic (N²/2) to linear (N). Same answer, same math, thousands of times less compute. The exam-student picture: read the paper once, TAKE NOTES on every line, then write the answer consulting your notes — instead of re-reading the whole paper before writing each word.

**But the notes cost paper — the memory bill (be able to reason through this live):**

```
KV cache size = 2 (K and V) × num_layers × num_kv_heads × head_dim
               × context_length × batch_size × bytes_per_value
```
What matters is the SCALING: cache grows LINEARLY with context length AND with concurrent requests. Worked intuition for a 70B-class model: serving 32k-token contexts can consume tens of GB of KV cache — per batch. The notes pile up fast: every word of every conversation, held in GPU memory for the duration of the request.

**And this one bill explains five facts you must know (trace each back to the cache):**
1. GPU memory = weights + KV cache + activations (File 03 §1.2). Once memory is full, no new requests fit — **concurrency per GPU is memory-limited, not compute-limited.** The GPU might sit at 40% compute with requests queuing — out of "paper."
2. That's why a 70B model serves fewer concurrent users than an 8B — its weights eat the memory the KV cache needs.
3. That's why long context is expensive — not just tokens-billed, but memory-per-request.
4. That's why quantization helps TWICE: smaller weights (more room for cache) and optionally smaller KV values (FP8 KV cache).
5. And that's why vLLM's PagedAttention exists — the memory WASTE in naive preallocation was 60–80%, and paging fixed it (File 03 §1.9).

**Memory hook:** *KV cache = the notes you take while reading the question, so you never re-read while writing the answer. Notes cost paper (GPU memory) — and the paper runs out before the compute does.*

### 2.5 Batching

**Plain words:** serving many user requests together on the same GPU, so one pass over the model weights serves all of them.

**The full logic (recap from §2.3, because this is the economics of the entire industry):** decode is memory-bandwidth-bound — streaming the weights from memory dominates each step. Serving 1 user or 50 users costs nearly the SAME memory traffic per step (one weights read), but produces 50× the tokens. So aggregate throughput scales almost linearly with batch size while per-user speed barely degrades (each user's tokens still arrive at a similar rate; everyone shares the memory read). **GPUs are only efficient when batched** — this single sentence justifies most of Track 13.

**The catch:** batching only helps when requests actually arrive together — which is exactly what continuous batching fixes.

### 2.6 Continuous Batching (a.k.a. iteration-level / in-flight batching)

**The problem with static batching, quantified:** collect a group of 50 requests and run them as one batch; finish when ALL complete. Request A wanted 20 tokens (done in 0.4s); request B wants 800 (16s). A's GPU slot sits IDLE for 15.6 seconds — you reserved memory for its KV cache and did nothing with it. Measured on real traffic, static batching wastes 30–50% of GPU capacity, because output lengths are wildly variable in LLM workloads.

**Continuous batching — schedule at the level of EACH DECODE STEP:**
- A request finishes mid-step → removed instantly, its KV memory freed at once.
- A new request is waiting → admitted at the very next step, into the freed slot.
- The GPU never runs an empty seat; memory never sits reserved-but-idle.

**The analogy (interviewers remember analogies):** static = a tour bus that departs only when every passenger has reached their own stop — finished passengers still occupy seats the whole way. Continuous = a metro — trains run continuously; passengers hop off at their station and new riders board at the very next one.

**Where you see it:** this mechanism (plus PagedAttention) is why vLLM reports several-fold throughput over naive serving, and it is THE core idea of modern serving engines — say exactly that. Note the limit: admission is bounded by KV memory — which is why §2.4's memory math and §1.9-of-File-03's paging matter operationally.

---

## Part 3: Model Optimisation

### 3.1 Quantization

**The story first — the engineer's Friday problem:** you need to serve a 70B model. Its 70 billion weights are stored as FP16 — 2 bytes each. Open the calculator: 70B × 2 = **140 GB**. Your GPU has 80 GB. The model doesn't fit — you need 2 GPUs just to HOLD the thing, before serving a single user, before even one token of KV cache (§2.4). Meanwhile your colleague serves an 8B model on one GPU with room to spare, and your budget has one GPU in it. What do you do?

The insight that saves the day: those 2 bytes per weight store numbers with a precision the model doesn't fully need. A weight's value is something like 0.0004721 — but does the model's BEHAVIOR depend on the difference between 0.0004721 and 0.00047? Not meaningfully. So shrink the boxes: store each weight in 1 byte (INT8) or half a byte (INT4), keeping a small scale factor to map back. Watch the arithmetic cascade:

- FP16 (2 bytes) = 140 GB → needs 2+ GPUs just to hold it
- INT8 (1 byte) = 70 GB → fits on one GPU
- INT4 (0.5 byte) = 35 GB → fits with 45 GB left over — and that leftover is KV cache room, which is CONCURRENT USERS (§2.4's memory equation)

And it's FASTER, not just smaller: decode is memory-bandwidth-bound (§2.3) — smaller weights = less memory traffic per step = higher TPS, roughly proportionally. One change attacks memory, cost, AND speed at once.

**How it works, mechanically:** quantization maps ranges of FP16 values onto a smaller grid of representable values (256 levels for 8-bit, 16 for 4-bit), with a scale factor per weight-group. Two families to name: **weight-only quantization** (weights quantized, activations stay high precision — most of the memory/speed win, lower quality risk; GPTQ and AWQ are the named methods) and full quantization. **GGUF** is the file format family popularized by llama.cpp for local models — a name to recognize.

**The trade-off (always state it, with the professional rule):** distinct FP16 values can collide when rounded to the same 4-bit level — outputs shift slightly; quality degrades measurably. Empirically: 8-bit is near-lossless for most tasks; 4-bit is usually acceptable but real. The professional rule: quantize to fit your hardware budget, then VALIDATE with evaluations on your golden set (File 06) — never assume quality held. "We quantized and it's fine" is not an engineering statement; "we quantized and groundedness dropped 0.6%, within tolerance" is.

### 3.2 Model Compression (beyond quantization)

**Distillation — the main technique, told as a story:** you have a 70B "teacher" that classifies support tickets brilliantly — and a bill that makes you cry. You want a 7B "student" that does this ONE task as well. Option A: train the 7B from scratch on the labeled tickets — it learns the mapping, but slowly, and it misses everything the teacher knows about *near-misses*. Option B (distillation): have the 7B learn by imitating the teacher's full output DISTRIBUTION. The difference is the secret: on a fraud ticket, a hard label says "fraud." The teacher's output says "fraud 91%, unusual 7%, normal 2%" — which quietly encodes "*this was nearly an 'unusual' case; here's exactly how close it was*." That per-example richness is far more signal than a single correct answer; the student learns the teacher's judgment, not just its verdicts. Result: a 7B student often retains ~90% of a 70B teacher's quality on a NARROW task at ~10× lower serving cost. The use-case pattern: pick your highest-volume single task (ticket tagging, extraction, classification) and distill a specialist for it — not "a smaller ChatGPT."

**Pruning:** delete weights that contribute little — neural networks are heavily over-parameterized, and many weights are near-zero. Structured pruning removes whole neurons/channels (real speedup, some quality loss, harder to do well); unstructured sparsity is mostly a research topic. Mention if asked; less practically important today than quantization and distillation.

**MoE — Mixture of Experts (the architecture behind frontier models), explained:** in a dense model, EVERY parameter is used for EVERY token. In MoE, the feed-forward layers are split into many "expert" sub-networks, and a small learned ROUTER activates only a few experts per token (e.g., 400B total parameters, ~12B active per token). The effects, both sides:
- **Win:** huge total knowledge (capacity) with cheap per-token compute (speed/cost) — this is how frontier models got smart AND cheap-ish simultaneously.
- **Costs:** ALL experts must sit in memory (memory-hungry — the opposite trade from quantization); routing and serving complexity; training instability.
- Interview framing: **MoE trades memory for compute; quantization trades quality for memory.** Knowing that these levers push in different directions is the sign of understanding, not memorizing.

### 3.3 Small vs Large Models

**Plain words:** small (1–13B) vs large (70B+). Your engineering job: route tasks to the right size — the model is a resource, and like any resource, right-sizing it is the engineering.

**The full comparison:**

| | Small model (1–13B) | Large model (70B+) |
|---|---|---|
| Cost per request | 10–100× cheaper | expensive |
| Speed (TTFT + TPS) | fast | slower (memory-bound decode is worse with more weights — §2.3) |
| Classification, extraction, formatting | ~equal quality | overkill |
| Multi-step reasoning, complex code, subtle nuance | notably weaker | strong |
| Fine-tuning cost | cheap, effective | expensive |
| Hardware | one GPU | multiple GPUs or hosted |

**Task-size matching (the skill being tested):** simple tasks (classify, extract, format, sentiment, route, simple rewrites) → small model: fast, cheap, quality-equivalent. Hard tasks (multi-step reasoning, complex code, subtle instruction-following, tool-use planning) → large model.

**The cascade pattern (the production pattern to recite):** try the small model first; if its confidence is low (logprob-based confidence, self-reported confidence, or a cheap verifier check), escalate to the large model. Best of both: most requests cost little; the hard ones still come back right. Confidence measurement is the engineering crux — a wrong confidence signal escalates everything (expensive) or nothing (wrong answers). This pattern is what production model routing in LLM gateways implements.

### 3.4 Hosted vs Self-Hosted Models

**Plain words:** call an API (OpenAI, Anthropic, Azure, Bedrock) vs run open-weights models (Llama, Mistral, Qwen) on your own GPUs.

**Worked crossover math (do this once — then you can do it live with any numbers the interviewer gives):** a workload at 2M requests/day, 1,000 input + 300 output tokens per request.
- **Hosted:** 2M × 1,000 = 2B input/day + 0.6B output/day. At $3/M and $15/M: $6,000 + $9,000 = **$15,000/day ≈ $450k/month.**
- **Self-hosted:** peak concurrency via Little's Law: 2M/day ≈ 23 req/s average, peak ≈ 100 req/s × ~3s = 300 concurrent. An 8B model (quantized to fit) at ~30 concurrent per GPU (File 03 §1.2) → ~10 serving GPUs + 2 spares + 2 for embedding/reranking ≈ **14 H100s ≈ $2.5–3.5/GPU-hour ≈ $30k/month infra** + serving platform + an ops team.
- **The crossover appears:** at ~$450k/month hosted vs ~$30k + team, self-hosting pays for a LOT of engineering — IF utilization stays high. At 200k requests/day (10× less), hosted = $45k/month vs the SAME fixed GPU fleet underutilized → hosted wins. **Same architecture, opposite answer, purely from volume.**
- And both answers assume quality parity — the self-hosted 8B must actually match the hosted frontier model on YOUR golden set (File 06 §2.1); a 20% quality drop "saves" money while losing the product.

**Hosted — the full ledger:**
- ✅ Zero ops; elastic scale on demand; access to the best frontier models; no upfront GPU investment; model updates arrive for free.
- ❌ Per-token cost scales FOREVER; data leaves your network (regulatory/privacy constraints — banking, health); rate limits (per-minute caps that bursty apps hit); provider outages are YOUR outages; vendors deprecate or silently change models (File 06 §2.13); limited customization (no deep fine-tuning of frontier models).

**Self-hosted — the full ledger:**
- ✅ Data stays inside (the hard requirement in regulated industries — say "regulatory" and you've justified it); at high volume, unit economics beat per-token pricing IF GPUs stay utilized; full control — pin versions, fine-tune deeply, customize serving; no vendor dependency.
- ❌ You own: GPU capacity planning and procurement (scarce — waitlists are real), serving infrastructure (vLLM etc.), upgrades, security patches, and the ops team that knows GPUs. Idle GPUs are money burning silently.

**The crossover-point answer (interviewers want the SHAPE of the reasoning, not a number):** at low/medium volume, hosted wins — you pay only for usage and skip the ops entirely; idle self-hosted GPUs cost 24×7 regardless of traffic. At high, STEADY volume, self-hosted wins on unit economics — the GPUs stay busy, amortizing their cost. Hence the pragmatic industry answer: **hybrid** — steady baseline traffic on self-hosted (internal serving platform), bursty/experimental/heavy traffic on hosted APIs — and that hybrid is exactly why multi-provider gateways exist: the choice between worlds becomes a routing policy at runtime instead of a one-time architecture commitment.

---

## Part 4: Architectural Decisions — the ladder

This is a guaranteed interview question: "When would you use prompting vs RAG vs fine-tuning vs a different model?" The framework, rung by rung, with the full reasoning for each.

### 4.1 Rung 1: Prompting (always first)

Better prompts + context in the prompt window solve most perceived "capability" problems. Includes: system prompt engineering, few-shot examples (2–5 input→output demos in the prompt — the model pattern-matches the format), structured output instructions/schemas, and chain-of-thought ("think step by step before answering").

**Why first, always:** hours to iterate, trivially reversible, zero training, zero infra. Cheap experiments before expensive commitments — the same logic as File 01's "rules before ML." If a prompt change fixes it, every other rung is over-engineering.

### 4.2 Rung 2: RAG (when the problem is KNOWLEDGE)

If the model needs facts it doesn't know or that CHANGE — internal docs, this week's numbers, your product's data — retrieve the relevant facts per request and hand them over. Update the index, not the model. Full treatment in File 04.

**Why not fine-tune for this (the follow-up you'll get):** knowledge in weights goes stale the moment documents change; every update = a new training run; weights can't cite sources (no traceability); and models are poor at fact-lookup from weights anyway — they pattern-complete, they don't query.

### 4.3 Rung 3: Fine-Tuning (when the problem is BEHAVIOR)

**Plain words:** continue training an existing model on YOUR examples so its weights absorb a pattern.

**The right reasons, each with its reasoning:**
1. **Consistent format/style at scale** — every response must be valid JSON in your schema; your brand voice; your document format. Prompting achieves this 95% of the time; fine-tuning achieves it 99.9% — and at millions of requests, that difference is real.
2. **A narrow skill at volume** — tagging support tickets with YOUR taxonomy, extracting fields from YOUR document format, performed millions of times. Teach a small model to match a big model on ONE task (distillation logic, §3.2), then serve it at 10× lower cost — fine-tuning as a COST play, not just a quality play.
3. **Domain tone/jargon** that prompting can't reliably produce.

**The wrong reasons (interviewers test these — know all four):**
1. **Injecting fresh knowledge** — stale the moment docs change; RAG's job.
2. **One-off tasks** — fine-tuning is a one-way door: changing behavior again = training again; prompting is reversible in seconds.
3. **Fewer than ~500–1,000 quality examples** — the model won't generalize; it will memorize noise (overfit).
4. **Needing traceability** — weights can't cite; retrieval can.

**The vocabulary to carry (define each correctly):**
- **SFT (supervised fine-tuning):** show input→output example pairs; the model learns the mapping.
- **RLHF / DPO:** train against human PREFERENCES (chosen vs rejected responses) — this is how chat models become "helpful" rather than merely predictive. Know the concept; not the algorithms.
- **LoRA (Low-Rank Adaptation) — the method that made fine-tuning practical:** instead of updating all billions of weights, train tiny low-rank adapter matrices (often <1% of parameters) alongside the frozen base, then merge them. Fine-tuning fits on one GPU; adapters can be swapped per-task at serve time. LoRA is to fine-tuning what containers are to deployment — the thing that made it operationally real.

### 4.4 The decision rule (memorize this answer, it's a complete interview answer by itself)

*"My ladder, cheapest first:*
1. *Prompt engineering — hours to iterate, fully reversible.*
2. *RAG — if the gap is knowledge: facts the model lacks or that change.*
3. *Fine-tuning — if the gap is behavior: consistent format/style/skill at volume — or cost: a fine-tuned small model replacing a big one on a single task.*
4. *Bigger model — only if capability genuinely isn't there after 1–3.*

*The diagnostic question: is the model failing on what it KNOWS (→ RAG) or how it BEHAVES (→ fine-tune)? And I re-evaluate with data at each rung — never skip rungs on vibes."*

### 4.5 Small vs Large (inside decisions)
After the ladder: right-size per use case (§3.3), cascade when uncertain, and let the routing layer (File 03 §1.5) execute the decision in production.

---

## Tough Interview Follow-ups (with full answers)

**Q: "Why is reading a long prompt fast but generation slow?"**
A: Prefill processes the entire prompt in parallel — GPU compute-bound, which GPUs excel at. Decode generates serially, one token per forward pass, and each pass is memory-bandwidth-bound: the whole model's weights stream from GPU memory to produce one token (§2.3). Batching amortizes the weight reads across concurrent requests (§2.5). So TTFT scales with prompt length, generation time with output length, and they have different bottlenecks — which is exactly how I'd debug "it's slow": decompose first, then fix the phase that's actually broken.

**Q: "What is the KV cache and what breaks without it?"**
A: Stored K and V vectors for all past tokens, computed during prefill. Without it, each new token would need K/V recomputed for the entire history — total work quadratic in output length (§2.4). With it, each step computes only the new token's K/V — linear. Cost: GPU memory, linear in context length × batch size — often the binding constraint on concurrency per GPU. That constraint is why PagedAttention, context caps, and quantized KV all exist.

**Q: "Users complain responses got slower at peak hours. Diagnose before touching anything."**
A: Decompose: queue wait + TTFT + generation time, compared peak vs off-peak. If queue wait or TTFT grew → capacity problem — replicas saturated, requests queuing; add replicas, improve batching, prefix-cache long shared prefixes. If generation time grew → decode-bound — route easy traffic to smaller models, cap outputs. The decomposition tells you which lever; only then act. And check whether it's one tenant (noisy neighbour — File 03 §1.11) before buying capacity.

**Q: "When is fine-tuning the WRONG choice?"**
A: Four cases: the gap is knowledge (fresh/changing facts belong in an index, not weights); fewer than ~500 quality examples; prompting achieves it (reversible beats retraining); you need source traceability (RAG cites, weights can't). And if quality-vs-cost is the motivation, check distillation — a small model fine-tuned on one task may beat a general bigger model.

**Q: "Why does the model perform worse on the middle of a long document?"**
A: "Lost in the middle" — attention weight distributes across all positions, and empirically models favor the start and end of long contexts; the middle gets less attention and worse recall. Mitigations: put critical content at the edges; rank retrieved chunks so the best are first or last; summarize stale history instead of hauling it; chunk-and-rank rather than one giant context (File 04 §2.5).

**Q: "Your app costs 2× more for Hindi users than English. Why, and what do you do?"**
A: Tokenization — the BPE vocabulary was learned on English-dominated corpora, so Devanagari text splits into more tokens per word; same meaning, more input AND output tokens → more cost and latency (§1.2). Options: route Hindi traffic to a model with a better multilingual tokenizer (a gateway routing feature — File 03 §2.5); compress Hindi contexts more aggressively; or accept and budget per-language. The key point is knowing the CAUSE is tokenization, not some vague "model struggling with Hindi."

**Q: "What limits how many users one GPU can serve?"**
A: GPU memory: weights (fixed) + KV cache per concurrent request (grows linearly with context length × users — §2.4) + activations. Once memory fills, requests queue even with compute headroom (File 03 §1.2's worked example). Levers: quantize weights, cap context length, PagedAttention (kills the 60–80% preallocation waste), or accept fewer concurrent users each served faster.

**Q: "Temperature 0 and the model still gives different answers sometimes. Why?"**
A: Even at temperature 0, sampling isn't the only non-determinism: floating-point non-associativity means batch composition changes results at the last decimal, and ties in top-probability tokens can break differently across hardware/kernels. Also batching: your request rides in different batches run-to-run, perturbing numerics. Lesson: never build a product that REQUIRES bit-identical LLM outputs — anchor determinism in your code, not the model.

---

## Rapid-Fire Flashcards

1. Token? → subword chunk (~4 chars, ~¾ word); model sees tokens, never letters.
2. BPE? → iteratively merge the most frequent pairs; frequent words = 1 token, rare words split, nothing unrepresentable.
3. Strawberry lesson? → model sees "berry" the token, not letters; spelling out loud converts opaque tokens into countable ones.
4. Whitespace significance? → " hello" ≠ "hello"; templates must match training exactly — templates are library code.
5. Context window includes? → input AND output; budget like memory.
6. Lost in the middle? → long contexts favor start and end; put the best content at the edges.
7. Chat memory illusion? → app re-sends the whole history each turn; window resets per request.
8. Streaming transport + trap? → SSE; middleboxes must NOT buffer; per-chunk timeouts.
9. TTFT = ? → queue + prefill (prompt length); THE chat UX metric.
10. TPS = ? → decode speed; memory-bandwidth and model-size bound.
11. The decomposition equation? → total = queue + TTFT + output_tokens/TPS — diagnose before fixing.
12. Temperature 0 vs high? → greedy/near-deterministic vs varied/creative; top-p trims the nonsense tail.
13. Attention Q/K/V? → Q = what I seek, K = how I'm found, V = what I hold; softmax(QKᵀ/√d)V blends relevant context.
14. Multi-head? → parallel small attentions tracking different relations; cost ~ quadratic in context.
15. Causal masking? → attend only backwards → generation is inherently serial.
16. Prefill? → whole prompt in parallel; fills KV cache; compute-bound → TTFT.
17. Decode? → one token per pass; serial; memory-bandwidth-bound → generation time.
18. The one insight? → decode reads ALL weights per token → batching amortizes it → GPU economics in one sentence.
19. KV cache? → stored K/V of past tokens; linear instead of quadratic; costs GPU memory linear in context × batch.
20. Static vs continuous batching? → tour bus (idle seats) vs metro (hop off/on per station); 30–50% waste eliminated.
21. Quantization? → FP16→INT8/4; smaller weights AND less memory traffic (faster decode); validate quality after.
22. Distillation? → small student learns the teacher's soft distributions; ~90% quality at ~10% cost on narrow tasks.
23. Pruning? → remove near-zero weights; less practical than quantization today.
24. MoE? → many experts, router activates few per token; trades memory for compute.
25. Ladder? → prompt → RAG (knowledge) → fine-tune (behavior/cost) → bigger model; cheapest first.
26. LoRA? → train tiny adapters, merge; fine-tuning on one GPU.
27. SFT vs RLHF/DPO? → input→output examples vs human preferences (helpfulness).
28. Hosted vs self-hosted? → pay-per-token + no ops vs privacy + unit economics at steady volume; hybrid via gateway.

---
