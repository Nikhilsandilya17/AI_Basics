# 08 — Interview Question Bank

> **How to use:** cover the answer, answer out loud, then compare. Mark anything you fumble and re-study that file section. Difficulty: 🟢 easy / 🟡 medium / 🔴 hard.

---

## A. ML & LLM Fundamentals (Files 01–02)

1. 🟢 What's the difference between training and inference? (File 01 §1.2)
2. 🟢 What is a token and why do LLMs use them instead of words? (File 02 §1.1)
3. 🟢 What is a context window? Do input and output share it? (File 02 §1.3)
4. 🟢 Why do LLMs have no memory between requests? How is chat "memory" implemented? (File 02 §1.3)
5. 🟡 Explain attention in plain words. What are Q, K, V? (File 02 §2.2)
6. 🟡 What is multi-head attention and why have multiple heads? (File 02 §2.2)
7. 🟡 What are prefill and decode? Which one determines TTFT? (File 02 §2.3)
8. 🔴 Why is decode memory-bandwidth-bound while prefill is compute-bound? (File 02 §2.3)
9. 🟡 What is the KV cache and what happens without it? (File 02 §2.4)
10. 🔴 Walk through the KV cache memory math. Why does it limit concurrency per GPU? (File 02 §2.4)
11. 🟡 What is continuous batching and why does it matter? (File 02 §2.6)
12. 🟡 Why did older models fail at "how many r's in strawberry"? (File 02 §1.1)
13. 🟡 What is quantization? What do you trade? (File 02 §3.1)
14. 🟡 Explain distillation and MoE. (File 02 §3.2)
15. 🟢 Explain precision vs recall with a fraud example. (File 01 §3.1)
16. 🟡 What is model drift vs data drift? How do you detect each? (File 01 §1.8–1.9)
17. 🟡 What's a feature store and what problem does it solve? (File 01 §2.4)
18. 🟢 Deterministic vs probabilistic systems — why does this change testing/monitoring? (File 01 §1.5)
19. 🟡 Latency vs accuracy vs cost — how do you prioritize? (File 01 §1.7)
20. 🔴 Same Hindi query costs 2× the English one. Why? (File 02 §1.2)

## B. Prompting / RAG / Fine-tuning decisions (Files 02, 04)

21. 🟢 When do you use RAG vs fine-tuning vs better prompting? (File 02 §4.4)
22. 🟢 What is LoRA and why does it matter? (File 02 §4.3)
23. 🟡 Why is fine-tuning the wrong tool for injecting fresh knowledge? (File 02 §4.3)
24. 🟢 Explain the indexing and retrieval pipelines of RAG. (File 04 Parts 1–2)
25. 🟡 What happens if chunks are too small? Too big? (File 04 §1.2)
26. 🟡 What is parent-child chunking? (File 04 §1.2)
27. 🟡 Explain embeddings and cosine similarity. (File 04 §3.1)
28. 🔴 You changed embedding models but kept old vectors. What breaks? (File 04 §3.1)
29. 🟡 Why do you need hybrid search? What does BM25 catch that embeddings miss? (File 04 §2.3, §3.7)
30. 🟡 What is RRF? (File 04 §2.3)
31. 🟡 Bi-encoder vs cross-encoder — why two-stage retrieval? (File 04 §2.4)
32. 🔴 Explain HNSW and its parameters M and ef. (File 04 §3.5)
33. 🔴 What is ANN's trade-off triangle? (File 04 §3.4)
34. 🔴 How do you make RAG permission-aware? Why is post-filtering a bug? (File 04 §4.2)
35. 🟡 A user searched an error code and got nothing. Why? (File 04 §2.3)
36. 🟡 What is HyDE? (File 04 §2.1)
37. 🟡 How do you evaluate retrieval quality? What is recall@k? (File 04 §4.5)
38. 🟡 How do you keep the index fresh? (File 04 §1.4, §4.4)
39. 🔴 Users see answers from outdated docs. Debug the pipeline. (File 04 follow-ups)
40. 🟡 What is "lost in the middle" and its implication for context building? (File 02 §1.3, File 04 §2.5)

## C. Serving & Infrastructure (File 03)

41. 🟢 Why GPUs for LLMs? When is CPU fine? (File 03 §1.1)
42. 🟡 What fills GPU memory? (File 03 §1.2)
43. 🔴 Why can requests queue even at 40% GPU "utilization"? (File 03 follow-ups)
44. 🟡 Why is GPU autoscaling different from CPU autoscaling? (File 03 §1.6)
45. 🟢 Why rate-limit by tokens and not requests? (File 03 §1.10)
46. 🟢 What is prompt/prefix caching? Why must static content come first? (File 03 §2.1)
47. 🟡 Semantic caching — benefits and dangers? (File 03 §2.2)
48. 🟡 What is vLLM / PagedAttention? (File 03 §3.1)
49. 🟢 Provider routing vs model routing — dimensions of each? (File 03 §1.5, §2.5)
50. 🟡 Hosted vs self-hosted models — how do you decide? (File 02 §3.4)
51. 🟡 What is a model fallback chain? Why must fallbacks be parity-tested? (File 03 §2.4)
52. 🟡 What is multi-tenancy? What's a noisy neighbour? (File 03 §1.11)
53. 🔴 Data/tensor/pipeline parallelism — explain each; what runs within a node vs across? (File 03 §5.1)
54. 🔴 What is disaggregated prefill/decode? (File 03 §5.2)
55. 🔴 Do the cost math: 1M req/day, 3k in / 400 out tokens. Then cut it 40%. (File 03 §4.2)

## D. Agents (File 05)

56. 🟢 What makes an agent different from a chatbot? (File 05 Part 0)
57. 🟢 Walk through the agent loop. What are stop conditions? (File 05 §1.1)
58. 🟡 How does tool/function calling work mechanically? (File 05 §1.4)
59. 🟢 The four types of agent memory? (File 05 Part 2)
60. 🟡 Workflow agents vs autonomous agents — when each? (File 05 §3.5–3.6)
61. 🟡 ReAct vs planner/executor? (File 05 §3.1–3.2)
62. 🟡 Why are multi-agent systems often better than one giant agent? (File 05 §3.7)
63. 🔴 What is indirect prompt injection? Why is it unsolvable in general? (File 05 §5.3, File 06 §3.3)
64. 🔴 What's your layered defense against prompt injection? (File 05 §5.3)
65. 🟡 Why must agent tool calls be idempotent? (File 05 §4.4)
66. 🟡 What is durable execution and why do agents need it? (File 05 §4.3)
67. 🟡 Design HITL approval for a refund action — how does the workflow wait for a human? (File 05 §4.5)
68. 🟡 Your agent is stuck in a loop calling the same tool. Fix it. (File 05 follow-ups)
69. 🔴 Agent runs cost ₹40; must be ₹5. Levers? (File 05 follow-ups)
70. 🟡 How do you evaluate an agent? (File 06 §2.5)

## E. Evals, Safety, Observability (File 06)

71. 🟢 Why is monitoring an AI system harder than a normal service? (File 06 Part 0)
72. 🟢 What is groundedness? (File 06 §1.3)
73. 🟡 What's in a golden dataset? What becomes a test case? (File 06 §2.1)
74. 🟡 LLM-as-a-judge biases and mitigations? (File 06 §2.3)
75. 🟢 Shadow vs canary deployment — difference and when to use each? (File 06 §2.10–2.11)
76. 🟡 Why version prompts? (File 06 §2.12)
77. 🟡 What is tracing and why is it THE AI debugging tool? (File 06 §2.14)
78. 🟡 What are guardrails? Input vs output moderation? (File 06 §3.8)
79. 🟡 Provider outage playbook? (File 06 §4.1–4.5)
80. 🟡 What are cost attacks and token exhaustion? Defenses? (File 06 §4.8–4.9)
81. 🔴 One-line prompt change doubled complaints. Process-wise, what went wrong? (File 06 follow-ups)
82. 🔴 Answer quality dropped 10% this week. Walk through your debugging. (File 06 follow-ups)
83. 🟡 How do you evaluate safety proactively, not reactively? (File 06 follow-ups)

---

## Scoring guide
- All 🟢 instantly + 80% of 🟡 → interview-ready fundamentals.
- Any 🔴 you can't answer → revisit that file section the same day.
- After the bank, do the File 07 case studies OUT LOUD with a timer (40 min each).
