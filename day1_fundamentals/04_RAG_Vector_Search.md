# 04 — RAG & Vector Search (Track 14)

> **Goal:** Go far beyond "Document → Embedding → Vector DB → LLM". Understand each pipeline stage deeply — why it exists, how it works, where it breaks.

---

## Part 0: Why RAG exists (the mental model)

**Plain words:** LLMs are frozen — they only know what they trained on. Your company's docs, this week's numbers, private knowledge don't exist for them. RAG = don't teach the model the facts; **hand it the relevant facts at question time.**

**Full explanation — why not the alternatives, each eliminated in turn:**
- *Why not a bigger model?* No model knows your private data, and world knowledge is frozen at training time. Bigger ≠ more current, and no size knows your Confluence.
- *Why not fine-tuning?* (File 02 §4.3) Fine-tuning shapes BEHAVIOR, not a searchable knowledge base: you'd retrain every time docs change (daily), it can't cite sources, and weights are a terrible database — they can't "look up," they pattern-complete. Facts injected into weights also leak and blend imprecisely.
- *So: RAG.* Facts live in a retrievable index that updates in minutes; the model gets exactly the relevant context per question and can cite it. Knowledge problem → retrieval problem.

**Analogy:** closed-book exam (pure LLM) vs open-book exam (RAG). The model doesn't memorize the textbook; it looks up the right page, then answers using it — with the page number shown (citations).

**The two pipelines (memorize the shape — every RAG discussion walks this):**
```
INDEXING (offline):  Documents → Parse → Chunk → Embed → Index → Vector DB
RETRIEVAL (online): Query → Query Processing → Retrieval → Hybrid Search → Reranking → Context Builder → LLM
```

**The iron law of RAG quality (recite this in every RAG answer):** *retrieval quality caps answer quality.* If the right chunk was never retrieved, the best LLM in the world cannot answer correctly — worse, it will answer anyway, from memory or thin air, with full confidence (hallucination). This is why RAG debugging always starts in the retrieval half: it's cheap to measure (no LLM needed — §4.5), and it gates everything downstream.

---

## Part 1: Indexing Pipeline (offline)

### 1.1 Parse

**Plain words:** extract clean text from messy sources — PDFs, HTML, slides, images, audio, spreadsheets.

**Full explanation — why this stage kills enterprise RAG projects:** a PDF is a VISUAL layout, not a text document. The text is positioned glyphs on pages: two-column articles interleave lines left-right; headers/footers/page-numbers pollute every page; tables are spatial arrangements the parser reads as word soup ("Qty 5 Price 200 Total 1000" — meaningless without structure). After parsing, the "content" can be unrecognizable — and every downstream stage (chunking, embedding, retrieval, answering) inherits the damage. **Garbage in the index = garbage answers forever, silently.**

**The engineering reality — budget serious time for parsing. Approaches in increasing sophistication:**
1. **Basic extractors** (PyMuPDF, pdfplumber) — fine for clean text PDFs, terrible for layouts.
2. **Structure-aware parsers** (Unstructured, LlamaParse, Azure Document Intelligence) — detect headings, tables, columns; preserve structure.
3. **LLM-based parsing** — an LLM reads the page (image or text) and outputs clean structured text; expensive per page but handles anything; right for high-value documents.
4. **Tables deserve special handling:** parse into structured rows, or keep them as images for multimodal retrieval (the model reads the table image directly).

**Interview line:** "In enterprise RAG, parsing quality dominates answer quality — most 'the model is hallucinating' complaints I've seen are actually 'the parser mangled the document' complaints."

### 1.2 Chunking

**Plain words:** split long documents into pieces small enough to retrieve individually.

**The full reasoning — why chunk at all, and why the size is a trade-off dial:**
- **Too BIG:** the embedding of a 5,000-token chunk must summarize everything in it — it becomes "about everything," which means "about nothing" in vector space (a diluted embedding matches nothing precisely). Plus big chunks waste context budget (File 02 §1.3).
- **Too SMALL:** a 50-token fragment like "...increased by 15% quarter over quarter" retrieves beautifully (specific!) but answers nothing (context-free — increased WHAT?). The chunk must carry enough surrounding meaning to be useful alone.

**Strategies, one by one, with when to use each:**
1. **Fixed-size with overlap:** split every ~500 tokens with 10–20% overlap. The overlap protects sentences that straddle a boundary from being destroyed. Simple, robust, a genuinely strong baseline — start here, measure, then evolve.
2. **Recursive/structural:** split on document structure first (markdown headings, sections, paragraphs), fall back to size limits when a section is too big. Respects the author's semantics — a "Pricing Policy" section stays whole. Better than fixed on structured corpora.
3. **Semantic:** split where meaning shifts — compute embedding similarity between consecutive sentences; cut where similarity drops. Fancy, marginal gains over structural in most real corpora; costs an embedding pass.
4. **Parent-child (small-to-big) — the pattern to name-drop:** INDEX small precise child chunks (they match queries well) but at answer time EXPAND to the parent section (it carries the full context the answer needs). Best of both worlds; standard practice in serious RAG systems.

**Numbers to anchor:** sweet spot typically 256–1024 tokens with 10–20% overlap; overlap in tokens, not sentences; tune against retrieval evals (§4.5), not vibes — the right number is a property of YOUR corpus.

### 1.3 Embed

Convert each chunk into a vector (~1024 numbers) capturing MEANING such that similar meanings land near each other in vector space (full treatment: §3.1). This is a batch job over your whole corpus — cost is per token, one time per document version. 10M chunks × 500 tokens × $0.02/M ≈ $100 — indexing is rarely the cost problem; serving is.

### 1.4 Index & the Vector Database

Store vectors + metadata (doc ID, section, ACL groups, timestamp, title, version) in a vector DB. Indexing builds the ANN structure (§3.4) for fast approximate search. Full DB comparison in §3.2.

**Index freshness (production concern, applies to §4.4 too):** documents change constantly. The indexing side must be event-driven: document store emits change events (or CDC) → queue → parse → chunk → embed → upsert. Delete = tombstone removal. Partial updates = idempotent chunk IDs (doc_id + section_hash) so re-indexing REPLACES rather than duplicates. A stale index serves confidently wrong answers with citations — the worst trust-destroying failure in RAG.

---

## Part 2: Retrieval Pipeline (online)

### 2.1 Query Processing

**Full explanation — why this stage exists:** users don't type well-formed research queries. They type fragments, follow-ups, pronouns: "and what about the free tier?" Retrieve with THAT and you get garbage — the query itself must be repaired before search.

**The toolkit, each technique with its reasoning:**
1. **Query rewriting:** an LLM (small one) rewrites the fragment into a standalone, well-formed query. Fixes fragments and typos.
2. **Conversational resolution:** "what about the free tier?" after discussing pricing → standalone query = "What does the free tier of the pricing plan include?" Requires folding in chat history — without this, multi-turn RAG breaks on every follow-up.
3. **Query expansion:** generate synonyms/variants ("refund" → "money back, reimbursement, cancellation policy") and search all — widens recall at the cost of extra searches.
4. **HyDE (Hypothetical Document Embeddings):** the counterintuitive-but-effective trick — ask an LLM to WRITE a hypothetical answer to the query, then search using THAT text. Why it works: queries are short question-language; documents are long answer-language — an answer matches documents better than a question does. Weird but effective; good flex to mention.
5. **Decomposition:** complex questions ("Compare our 2025 and 2026 refund policies for enterprise customers") split into sub-queries, retrieved separately, contexts merged.

**Cost note:** each technique is an LLM/embedding call of its own — balance quality gain vs added latency/cost per query. Query rewriting is almost always worth it; the others are situational.

### 2.2 Retrieval (dense)
Embed the processed query → ANN search over chunk vectors → top-k candidates (typically 50–100 candidates, later narrowed).

### 2.3 Hybrid Search

**Full explanation of WHY this is mandatory:** dense (embedding) search and sparse (keyword/BM25) search fail on OPPOSITE things:

- **Dense fails on exact literals:** query "ERR-4021" or `getUserBalance()` or "version 2.31.4" — the embedding captures "an error code"-ness, not WHICH code. The right document contains the literal string, but semantic matching may rank ten other error-code discussions above it.
- **BM25 fails on paraphrase:** "How do I get my money back?" shares ZERO keywords with the "Refund Process" policy page. Keywords are brittle to vocabulary mismatch.

Combine both and each covers the other's blind spot. **RRF (Reciprocal Rank Fusion) — the standard merging method, understand it by DOING it once:**

Query: "how do I get my money back for order 8841"

| Document | Dense rank | BM25 rank | RRF score = Σ 1/(60+rank) |
|---|---|---|---|
| refunds.html "Refund Process" | 1 | 7 | 1/61 + 1/67 = 0.0164 + 0.0149 = **0.0313** |
| orders.html "Order 8841 details" | 5 | 1 | 1/65 + 1/61 = 0.0154 + 0.0164 = **0.0318** |
| returns.html "Return shipping" | 2 | 12 | 1/62 + 1/72 = 0.0161 + 0.0139 = 0.0300 |
| errors.html "ERR-4021" | 30 | 2 | 1/90 + 1/62 = 0.0111 + 0.0161 = 0.0272 |

Read the table and see the three properties interviewers want: (1) documents ranked high by BOTH methods float to the top (top-2 are strong in both lists); (2) a document ranked #30 by dense but #2 by BM25 still surfaces at #4 — each method can RESCUE documents the other buries; (3) no score calibration needed — BM25 scores and cosine similarities aren't comparable, but RANKS always are. Simple, robust, the default.

### 2.4 Reranking

**Full explanation — the two-stage funnel:** you cannot run the accurate scorer over the whole corpus — so do cheap recall first, expensive precision second.

- **Stage 1 (retrieval, bi-encoder):** query and documents embedded SEPARATELY (document embeddings precomputed offline). Compare with a dot product — instant. But shallow: the two texts never "read" each other; similarity is one vector-vs-vector glance.
- **Stage 2 (reranking, cross-encoder):** query and document concatenated and read TOGETHER through one transformer — the model attends from query words to document words and back — a true relevance judgment. 10–100× more accurate, 10–100× more expensive — hence only run on the top ~50 candidates, keeping the best 3–5.

**Memory hook:** *cheap recall, expensive precision.* Stage 1 fishes broadly with a net; stage 2 examines each catch closely.

### 2.5 Context Builder

The last stage before the LLM — assembling what actually goes in the prompt. The decisions:

1. **Budget:** top-k chunks must fit the token budget after system prompt + history + output reserve (File 02 §1.3). Reranking exists to make k small (top 3–5 beats top 20 for cost AND for "lost in the middle" — File 02 §1.3).
2. **Ordering:** models attend best to start and end of the context. Put the BEST chunk first (or last) — never let the winner sit in the middle.
3. **Formatting/labeling:** label each chunk with source metadata (title, section, URL, date) in the prompt → the model can cite them (§3.11).
4. **Deduplication:** overlapping chunks (from overlap in chunking or repeated documents) waste budget — dedupe before assembly.
5. **Compression (optional):** long chunks can be summarized or trimmed to query-relevant sentences before insertion.

---

## Part 3: Core Topics (each in depth)

### 3.1 Embeddings

**Plain words:** a numeric fingerprint of meaning — a list of numbers (a vector, e.g., 1024 dimensions) where texts with similar MEANING land near each other in space, regardless of the exact words used.

**Full explanation:** an embedding model (a small transformer trained on billions of text pairs — question/answer, paraphrases, title/body) maps text → vector. Geometry carries semantics: "How do I get my money back?" and "refund process" produce vectors pointing in nearly the same direction even though they share no keywords. **Cosine similarity** (the angle between two vectors — 1 = same direction, 0 = unrelated) measures the closeness.

**Properties you must know (each is an interview trap):**
1. **Model-locked:** vectors only compare within the SAME embedding model. Model A's numbers mean something different from Model B's. Mixing them = silent garbage similarity scores.
2. **Therefore: changing the embedding model = RE-EMBEDDING the entire corpus.** A classic migration project that surprises teams ("we upgraded embeddings" → "you re-embed 10M chunks, or your search silently dies").
3. **Query and documents MUST use the same model** — same rule, same trap.
4. Dimensions trade expressiveness vs memory/speed: 384 (fast/cheap) to 3072 (accurate/expensive) — a per-use-case decision.

### 3.2 Vector Databases

**Plain words:** a database whose superpower is "find the nearest vectors to THIS vector," plus metadata filtering.

**The landscape (what to actually say):**
- **Managed:** Pinecone — zero-ops, expensive.
- **Open source:** Milvus, Qdrant, Weaviate — you run them.
- **pgvector (Postgres extension):** for anything under a few million vectors, Postgres is plenty — huge win when your data already lives there. Senior-sounding answer: "don't add a new infra dependency before you need it; pgvector until ~5–10M vectors, then evaluate dedicated stores."
- **Elasticsearch/OpenSearch:** hybrid search natively (BM25 built in for decades) + vector support.

**Selection criteria that matter:** ANN algorithm quality, metadata filtering WITH vector search (§3.6), hybrid support, multi-tenancy, incremental updates, managed vs self-hosted.

### 3.3 Semantic Search
Search by meaning (dense vectors). Strength: paraphrases, cross-lingual (multilingual models match "refunds" with "Rückerstattung"). Weakness (from §2.3): exact identifiers, negation ("not eligible"), rare entities, numbers. The weaknesses are exactly why hybrid exists.

### 3.4 ANN Search (Approximate Nearest Neighbour)

**The problem:** 10M vectors, exact search = compare the query against EVERY vector — 10M comparisons per query, every query. At 100 queries/sec that's a billion comparisons/sec — dead on arrival.

**The idea:** an index structure that finds almost-certainly-the-right neighbours while touching only a tiny fraction of the data (thousands instead of millions).

**The trade-off triangle to recite:** **recall vs latency vs memory.** ANN parameters buy recall with latency (explore more candidates) or memory (bigger index). Perfect recall = exact search = slow. Every vector DB dial (ef, nprobe, M) is a point on this triangle.

### 3.5 HNSW (Hierarchical Navigable Small World)

**The algorithm behind most vector DBs — be able to explain it in plain words:**

**The structure — a country's road system in layers over the same cities:**
- Top layer: a few "highway" nodes with long links between distant regions.
- Each lower layer: more nodes, shorter links.
- Bottom layer: EVERY vector is a node, connected to nearby neighbours.

**The search:** start at the top layer. Greedily hop: from the current node, look at neighbors, move to whichever is closest to the query; repeat until no neighbor is closer (local minimum on this layer). Then drop one layer down (finer roads, more nodes) and continue. At the bottom layer you're doing fine-grained search within the right neighbourhood. Never compared against the whole corpus — each layer eliminates huge regions wholesale.

**A worked HNSW search trace (follow it once — you can then explain HNSW forever).** Query vector Q; 10M vectors; 3 layers; say each hop inspects ~10 neighbors:
```
Layer 2 (top, ~100 "highway" nodes):
  start → A (dist 0.9) → B (0.6) → C (0.5) → no neighbor of C is closer → drop
Layer 1 (~10,000 nodes):
  continue from C → D (0.34) → E (0.21) → no closer neighbor → drop
Layer 0 (all 10M vectors):
  continue from E → F (0.12) → G (0.07) → no closer neighbor → DONE
  Return: G (0.07) and its neighbors as the result set
```
Total distance computations: ~10 hops × 10 neighbors × 3 layers ≈ **300 comparisons instead of 10,000,000** — five orders of magnitude fewer, with recall typically 95–99% of exact search. That "95–99% recall at ~300 comparisons" sentence is the whole value proposition of ANN, and this trace is the proof you can recite. The one-hop-at-a-time character also explains the runtime dial: ef = "how many neighbors do I examine per hop" — raise ef from 10 to 50 and you do ~5× the work for a recall bump of a few points (the recall-latency dial, §3.4, now concrete).

**Parameters (know both):**
- **M** — links per node. More links = better recall + more memory + slower build.
- **ef (search width)** — how many candidates to consider during traversal. Higher = better recall, slower query. The runtime dial.

**Production characteristics:** excellent recall/latency, memory-hungry (graph + vectors all in RAM), slow to build, and bad at deletions (graphs are hard to un-build — tombstones; hence periodic index rebuilds at scale, relevant to §1.4 freshness).

### 3.6 Metadata Filtering
Filter DURING search: `tenant = X AND acl_groups ∩ user_groups AND updated > 2026-01-01`. THE mechanism for multi-tenant and permission-aware RAG (§4.1–4.2).

**The pre-filter vs post-filter bug (real interview gold):** POST-filtering (search top-k, then remove unauthorized) can return 2 of k after filtering — you paid for k retrieval and serve 2 chunks, and you may have missed the best authorized chunks. PRE-filtering (apply filter inside the ANN search) keeps k valid results. Vector DB support for efficient pre-filtering is a real selection criterion.

### 3.7 BM25 (the sparse half)

**Plain words:** the classical keyword relevance formula — the algorithm behind search engines for decades.

**How it scores a document for a query, in two ideas:**
- **TF (term frequency):** the document scores higher when query words appear more often... but with diminishing returns (10 mentions barely beats 5 — a saturating function, so one keyword spam doesn't win).
- **IDF (inverse document frequency):** words RARE in the corpus count MORE. "The/and/of" appear everywhere → near-zero weight. "ERR-4021" appears in 3 documents → enormous weight — those 3 documents will be found.

**Why it survives in the LLM era:** it's unbeatable at exactly what dense search is worst at — exact tokens, IDs, code identifiers, version numbers. Modern hybrid = BM25 + dense + RRF (§2.3).

### 3.8 Hybrid Search
§2.3 — dense + sparse, fused with RRF. The production default; pure-dense RAG is a known gap interviewers probe ("user searches an error code and gets nothing — why?").

### 3.9 Reranking
§2.4 — bi-encoder recall, cross-encoder precision; keep top 3–5.

### 3.10 Grounding

**Plain words:** forcing the model to answer ONLY from the provided context.

Mechanics: system prompt instructs "Answer only using the provided context. If the context doesn't contain the answer, say you don't know." Optionally: cite the chunk for each claim; penalize uncited claims (detectable hallucination signal).

**The honest caveat to state:** grounding reduces hallucination but cannot eliminate it — the model can still over-extrapolate from context or mis-cite. That's why groundedness is MEASURED (File 06 §1.3) rather than assumed.

### 3.11 Citations
Attach source metadata to every chunk and instruct citation. Purposes: user trust (verify-able answers), audit trails (compliance), and hallucination detection (a claim with no supporting chunk = suspect). Engineering requirement: chunk→source mapping must survive the WHOLE pipeline (parse → chunk → embed → retrieve → prompt) — losing it in any hop kills citations.

### 3.12 Context Management
The context builder (§2.5) + conversation history budgeting: retrieved context + history + output must fit the window. Techniques: sliding window over turns, SUMMARIZE older turns (rolling summaries), offload tool results out of history after use. 
---

## Part 4: Production RAG

### 4.1 Multi-Tenant RAG

**Two isolation architectures, trade-off explicit:**
- **Hard isolation (index per tenant):** each tenant gets its own index/collection. Simple security story (wrong-index access is structurally impossible), per-tenant tuning; but N indexes = overhead, cost, operational mess at many tenants.
- **Soft isolation (shared index + tenant_id metadata filter):** efficient, one ANN structure to maintain; one filter bug away from cross-tenant leakage.

The senior answer: soft isolation for scale, hard isolation for the most sensitive tenants or few-tenant deployments — and ALWAYS per-tenant quotas and metering (noisy neighbour, File 03 §1.11).

### 4.2 Permission-Aware RAG (the classic interview trap)

**The trap, fully told:** employee A can see doc X, employee B cannot. Index the docs naively, and B's semantic search happily retrieves X's content into the LLM's context — the LLM answers with it, citations and all. RAG silently bypasses every file permission the company ever set. It's not a hypothetical — it's THE first security bug in every enterprise RAG deployment.

**The rules (memorize):**
1. Enforce ACL at RETRIEVAL time — the metadata filter INSIDE the search (§3.6 pre-filter), never after, never by prompt ("only use documents the user may see" — prompts are not security).
2. Permissions change AFTER indexing: a doc declassified or restricted post-indexing must update the index's ACL metadata — track permission versions; re-check live ACL for high-stakes deployments.
3. "Permission drift" (docs whose ACLs changed since indexing) is a real production failure mode — audit it periodically.

### 4.3 Document Updates
Event-driven incremental: changed docs → re-parse → re-chunk → re-embed → upsert, with IDEMPOTENT chunk IDs (doc_id + section_hash) so re-indexing replaces rather than duplicates; deletions tombstoned. Half-updated documents must not be served (version gating).

### 4.4 Index Freshness
Define an SLA ("doc updated → searchable within 5 minutes") and architect for it: webhooks/CDC → queue → embed workers → upsert. Freshness vs cost dial: continuous streaming (fresh, always-on workers) vs scheduled crawls (cheap, stale). Freshness failures show up as "the bot answered from last month's policy" — catastrophic for trust because the citation looks legitimate.

### 4.5 Retrieval Evaluation

**The discipline that separates production RAG from demos — evaluate in LAYERS, cheap before expensive:**

**Layer 1 — Retrieval metrics (no LLM needed, fast, objective):** build a golden set of questions where you know the RIGHT CHUNKS. Then:
- **Recall@k** — of the chunks that should be found, what fraction made the top-k. THE king metric: if recall@5 is 40%, no prompt tweak will save you.
- **MRR (Mean Reciprocal Rank)** — 1 if the right chunk ranked #1, ½ if #2, ⅓ if #3... averaged. Rewards putting the best chunk FIRST (which matters — §2.5 ordering).
- **NDCG** — position-weighted quality with graded relevance (perfect > good > related). Use when relevance isn't binary.
- **Precision@k** — of what was retrieved, how much was relevant (penalizes stuffing).

**Layer 2 — Generation metrics (LLM-as-judge or human, expensive):** groundedness/faithfulness (is every claim supported by retrieved context? — File 06 §1.3), answer relevance (does it answer the actual question?), citation accuracy (do citations actually support the claims?).

**The workflow:** change chunking → re-run Layer 1 (minutes, free) → only if recall improves, run Layer 2 (expensive) on the improved candidate. Evaluating end-to-end only is like debugging with printf at the final return — you can't localize the fault. 
### 4.6 RAG Caching
Cache at every layer: **embedding cache** (query → its vector, skip re-embedding), **semantic cache** (query → similar past question → served chunks/answer — careful, File 03 §2.2), **response cache** (exact query+doc-version → answer, invalidated on doc change), **provider prefix caching** (the static RAG preamble — instructions, output format — cached at ~20% price; often the biggest input-token saver because RAG prompts share long prefixes).

### 4.7 Cost Optimisation for RAG
Levers specific to RAG: rerank and shrink top-k (20 → 4 chunks: 5× less context, often BETTER answers — less "lost in the middle"), trim/recompress chunks, prefix-cache the static preamble, small model for query processing and reranking, embed only on change, semantic cache for FAQ-class queries, and route easy factual lookups to small models (File 03 §1.5). Then File 03 §4.2's math applies with RAG's input-heavy profile.

---

## Part 5: Case Studies (skeletons — full walkthroughs in File 07)

### 5.1 Design Enterprise Document Q&A
Millions of docs from Drive/SharePoint/Confluence; citations required; permissions respected; minutes-fresh.
Flow: connectors → event-driven indexing (parse → chunk → embed → upsert with ACL metadata) → hybrid retrieval + rerank → grounded generation with citations → evaluation pipeline + feedback loop.
Expect deep-dives on: permission handling (§4.2), tables/diagrams (§1.1 parsing), freshness SLA (§4.4), tenant isolation (§4.1), cost (input-heavy — §4.7).

### 5.2 Design Perplexity (AI search)
The twist vs enterprise RAG: the corpus is THE LIVE WEB — you cannot pre-index the internet. So retrieval happens AT QUERY TIME: decide search queries → call web search APIs → fetch top pages → extract text → rerank → LLM synthesis with citations.
Key differences from doc-QA: no prebuilt index (latency budget ~2–4s end to end), freshness by construction, source quality weighting (trustworthy domains rank higher), caching popular queries becomes the main cost lever, and citation integrity IS the product (one fabricated citation = trust gone).

---

## Tough Interview Follow-ups (with full answers)

**Q: "Users report answers from OUTDATED docs. Debug the pipeline."**
A: Trace the freshness chain end-to-end: (1) did the doc store emit a change event (connector/webhook fired)? (2) did indexing process it (queue backlog? embedder errors?) (3) were chunks REPLACED (idempotent IDs) or duplicated (stale + new chunks both live, and the old one ranks higher)? (4) is retrieval preferring the old version (add recency to rerank; version-gate to current)? (5) short-term mitigation: citations include doc version/date so users can self-verify. Fix the broken link, don't just re-index.

**Q: "Recall@5 is 0.4. What do you fix, in what order?"**
A: Isolate the failing stage before touching anything: (1) is the right text in the INDEX at all (parser mangled it — §1.1)? (2) is it chunked retrievably (context-free fragment or diluted mega-chunk — §1.2)? (3) is it EMBEDDED well (right embedding model for the language/domain)? (4) is retrieval finding it (hybrid missing — the chunk is all identifiers, dense search blind to it — §2.3)? (5) is reranking burying it (§2.4)? Change one stage, re-measure, repeat. Recall@0.4 is almost always stage 1, 2, or 4.

**Q: "Why not fine-tune on our docs instead of RAG?"**
A: Four reasons: freshness (index updates in minutes, weights go stale per retrain), cost per update, citations/auditability (weights can't cite), and fine-tuning is poor at fact storage anyway — it shapes behavior, not knowledge retrieval. Fine-tune for format/tone on top of RAG if needed — they're complementary, not alternatives.

**Q: "Same query, different answers for two users — bug or feature?"**
A: Feature — permission-aware retrieval (§4.2): each user searches only their accessible docs, so retrieved context differs, so grounded answers differ. A system that ignores ACL for "consistency" is the bug — a security bug.

**Q: "When is pure vector search enough, and when do you NEED hybrid?"**
A: Natural-language paraphrase traffic, no exact identifiers → dense alone can work. The moment the corpus contains error codes, SKUs, function names, version numbers, or any literal tokens that users will search — hybrid is mandatory, because dense retrieval is structurally blind to exact literals (§2.3). Enterprise corpora are always full of identifiers, so the default is hybrid.

**Q: "Explain HNSW like I'm a junior engineer."**
A: Layers of a road map over the same cities: top = few highways, bottom = every city with local streets. To find the closest city to a target: start on highways, hop greedily closer; at the right region, drop to state highways; then local streets. At each hop you only look at neighbours, never the whole map — that's how you search 10M vectors touching only thousands. Two dials: M (roads per city — recall/memory) and ef (how many side roads you're willing to explore per hop — recall/latency).

**Q: "Your RAG answers are too generic — the model ignores the specifics in the context. Why?"**
A: Three usual suspects: (1) the specifics never made it into the context (retrieval missed — check recall first — §4.5); (2) they made it but got buried (lost in the middle — reorder so the best chunk is first/last — File 02 §1.3, §2.5); (3) the prompt doesn't force use ("answer ONLY from the context, quote the relevant clause, cite it"). Debug in that order — retrieval, then assembly, then prompt.

---

## Rapid-Fire Flashcards

1. RAG one-liner? → open-book exam: retrieve relevant facts per question, hand them over, cite sources.
2. The iron law? → retrieval quality caps answer quality — debug retrieval first.
3. Indexing pipeline? → Parse → Chunk → Embed → Index → Vector DB.
4. Retrieval pipeline? → Query processing → Retrieval → Hybrid search → Rerank → Context builder → LLM.
5. Why parsing dominates? → PDFs are layouts, not text; garbage index = garbage answers, silently.
6. Chunk too small / big? → context-free fragments vs diluted embeddings.
7. Parent-child chunking? → index precise children, serve the parent.
8. Embedding? → numeric meaning-fingerprint; cosine similarity; model-locked.
9. Changing embedding models? → re-embed everything; vectors never mix.
10. ANN triangle? → recall vs latency vs memory.
11. HNSW? → layered road map, greedy zoom-in; dials M and ef.
12. BM25? → TF with saturation × IDF (rare words weigh most); the sparse half.
13. RRF? → merge ranked lists by 1/(k+rank); no score calibration.
14. Bi vs cross-encoder? → separate precomputed embeddings (fast/shallow) vs read together (accurate/slow) — cheap recall, expensive precision.
15. Pre vs post-filtering? → filter inside search (valid top-k) vs filter after (k shrinks, may miss best) — pre is correct.
16. Permission-aware rule? → ACL filter inside retrieval; never prompt-based, never post-hoc; track permission drift.
17. Recall@k? → fraction of correct chunks that made top-k — the king metric.
18. MRR? → 1/rank of first correct hit, averaged.
19. Eval layering? → retrieval metrics (cheap, no LLM) first; generation metrics (expensive) after.
20. HyDE? → search with a hypothetical ANSWER — answers match documents better than questions.
21. Perplexity's difference? → no prebuilt index; query-time web retrieval + synthesis with citations.
22. RAG caching layers? → embedding, semantic, response (doc-version keyed), provider prefix caching.

---
