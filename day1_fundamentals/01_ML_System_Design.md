# 01 — ML System Design (Track 10)

> **Goal of this file:** Understand how ML systems differ from normal software, how a model actually reaches production, and how to talk about all of it confidently in an interview — with zero doubts left.

---

## Part 1: AI System Design Foundations

Before we begin, understand WHY this part exists. When an interviewer asks you to "design a fraud detection system" or "design an AI agent platform," they are testing whether you think like a normal backend engineer who happens to call an AI API, or like someone who understands that ML changes the rules of system design. Every topic in Part 1 is one of those changed rules. Master these and you'll naturally structure AI design answers better than 90% of candidates.

---

### 1.1 Traditional Software vs ML Systems

**Plain words:** Normal software follows rules YOU wrote. ML software learns rules from DATA.

**Full explanation with example:**

Imagine you must build a spam filter.

*Approach 1 — Traditional software:* You sit down and write rules:
```
if email contains "WIN LOTTERY" → spam
if sender is not in contacts AND email contains link → spam
if email has 5+ exclamation marks → maybe spam
```
This works... until spammers write "W1N L0TTERY". You add a rule. They adapt. You add another rule. This is an arms race you fight manually forever. After 2 years you have 4,000 rules, half of them contradict each other, and nobody remembers why rule #1734 exists.

*Approach 2 — ML:* You collect 100,000 emails that humans already labeled as "spam" or "not spam". You feed these to a learning algorithm. The algorithm adjusts millions of internal numbers (called **weights**) until its outputs match the human labels as closely as possible. The result: a function that predicts spam-ness for emails it has NEVER seen before. When spammers change tactics, you collect new examples, retrain, done — no manual rule editing.

That adjustment process is called **training**, and the resulting "function with tuned weights" is called a **model**.

**The key differences, one by one (interview gold):**

| Aspect | Traditional Software | ML System |
|---|---|---|
| Where does the logic live? | In the code you write | In the weights, learned from data |
| Behaviour on same input | Always same output | Usually same, but not guaranteed (sampling, versions) |
| What can go wrong | Code bugs | Data bugs, bad labels, distribution changes, model limitations |
| How do you test it? | Test cases: exact expected outputs | Statistical evaluation: "96.4% accuracy on a held-out test set" |
| Does it degrade over time? | No (code doesn't rust) | YES — the world changes, the model doesn't (drift — §1.8, §1.9) |
| Debugging a wrong output | Read the code path | Hard! Trace inputs, check data, check version, check features |
| Changes require | Code change + deploy | Data collection + retraining + evaluation + deployment |

**The deepest point (say this to impress):** In traditional software, correctness is *verifiable by reading code*. In ML, correctness can only be *estimated statistically*. You can never prove an ML system is correct — you can only measure how often it was correct on data you've seen, and hope that generalizes to data you haven't seen. This single fact changes testing, monitoring, deployment, and even product decisions (whether ML is acceptable for a task at all).

**Common misconception to avoid:** "ML replaces rules entirely." In production, the best systems are usually HYBRID: deterministic rules for the clear cases (amount > ₹10 lakh → always human review), ML for the fuzzy middle. Saying this in an interview shows real-world maturity.

---

### 1.2 Training vs Inference

**Plain words:**
- **Training** = *studying*. The model reads millions of examples and adjusts its weights. Slow, expensive, done occasionally.
- **Inference** = *writing the exam*. The model applies learned weights to new inputs and produces outputs. Fast, cheap-ish, done millions of times a day.

**Full explanation:**

During training, the system repeatedly does this loop:
1. Take a batch of examples (say 256 emails).
2. Run the model on them → get predictions.
3. Compare predictions to the true labels → compute a **loss** (a single number measuring "how wrong we were").
4. Adjust every weight slightly in the direction that reduces the loss (via **gradient descent** — think of a hiker walking downhill in fog, one small step at a time; the mountain height is the loss).
5. Repeat with the next batch — millions of times, for days or weeks, across hundreds of GPUs.

For a modern LLM, training costs range from thousands to millions of dollars and takes weeks. That's why nobody retrains casually.

During inference, there is no learning. The weights are **frozen**. A request comes in, the model does forward-pass math (matrix multiplications), and out comes a prediction. One forward pass. No weight updates.

**Why this matters enormously for system design (interview gold):**

Training and inference have completely opposite hardware needs:

| | Training | Inference |
|---|---|---|
| Runs | Occasionally (weekly/monthly/yearly) | 24×7 forever |
| Duration per run | Days to weeks | Milliseconds to seconds per request |
| Hardware | HUGE GPU clusters, all talking to each other (gradients must be shared) | Individual replicas, independent, behind a load balancer |
| Failure impact | Lose a run → restart from checkpoint (annoying, costly) | One request fails → retry (normal ops) |
| Optimization goal | Total time to finish training | Latency per request + throughput |

This is why the infrastructure topics are split (distributed training concepts in File 03 Part 5; everything else is serving).

**The follow-up you WILL get: "Why can't you just retrain often to fix problems?"**
Full answer: Three reasons. (1) **Cost** — training runs are expensive; you don't trigger them casually. (2) **Data** — retraining needs fresh, clean, correctly-labeled data; collecting and validating that takes time. (3) **Risk** — a new model might fix one thing and break another; every new model must be evaluated against golden datasets (Part 3) and rolled out gradually (canary — §2.7) before replacing the current one. So mature teams retrain on a *schedule* (e.g., weekly) or when *monitoring* shows degradation — the decision is data-driven, not vibes.

---

### 1.3 Batch vs Online Inference

**Plain words:**
- **Batch inference** = answering ALL questions at once, on a schedule. Like a teacher checking 500 answer sheets over the weekend.
- **Online inference** = answering ONE question immediately when asked. Like a student answering on the spot in class.

**Full explanation with examples:**

*Batch example — Netflix recommendations (simplified):* At 2 AM every night, a job runs over all 200M users and pre-computes "recommended rows" for each one, writing results to a cache. When you open the app, your recommendations are already sitting there — no model runs at request time. Cost-efficient: you can use big, expensive models (latency doesn't matter at 2 AM), you can batch optimally (GPUs love batches — File 02 §2.5), and you can plan capacity exactly.

*Online example — fraud detection at card swipe:* The card is swiped; you have ~100–300 milliseconds to decide approve/decline, because the payment terminal is waiting. The model must run at request time, with low-latency replicas, autoscaling, the works. Every millisecond and every GPU is on the critical path.

*Online example — Uber ETA:* You open the app; ETA must appear in a second or two.

**How to decide (the framework to recite):** Ask three questions:
1. **Does the user need it right now?** (Yes → online; no → batch is cheaper)
2. **How stale can answers be?** Batch answers are as old as the last run. Recommendations from last night are fine; fraud decisions must be real-time.
3. **What's the volume pattern?** Predictable, huge volumes (score all 50M accounts nightly) suit batch. Spiky, interactive traffic suits online with autoscaling.

**The subtle production pattern (mention for seniority): many systems are BOTH.** Fraud: online model for the instant decision + nightly batch model that re-scores everything for analyst review queues. Or recommendations: nightly batch for the base rows + online model for "because you watched X" fresh signals. This "offline precompute + online real-time" blend is extremely common in production.

---

### 1.4 Offline vs Real-Time Systems

**Plain words:** This is the same idea as batch vs online, but zoomed out to the whole SYSTEM — including how data flows in.

**Full explanation:**

An **offline system** works on stored, historical data that has already been collected. Data arrives via pipelines that run periodically (hourly/nightly ETL jobs). Everything in the system — features, predictions, reports — operates on data that is, by definition, from the past.

A **real-time system** reacts to events as they happen. Data arrives as a stream (Kafka topics): "transaction happened", "user clicked", "ticket created". The system processes each event within seconds or milliseconds of occurrence.

**Concrete contrast:**
- Detecting fraud: real-time (block the transaction at swipe time).
- Analyzing last month's fraud patterns to train next week's model: offline.
- Recommending what to show a user RIGHT NOW: real-time features (clicked 2 items in "electronics" in this session) + offline features (their average spend over 6 months).

The engineering takeaway: real-time systems need streaming infrastructure (Kafka, stream processors), lower per-event latency, and careful handling of late/missing data. Offline systems tolerate slower, heavier processing (Spark-style batch jobs). Most ML systems in production have BOTH halves, feeding the same model — which is exactly why **feature stores** exist (§2.4): to compute the same feature once and share it between both.

---

### 1.5 Deterministic vs Probabilistic Systems

**Plain words:**
- **Deterministic** = guaranteed. 2+2 is always 4. Given the same input, the output is always the same.
- **Probabilistic** = best guess with uncertainty. "70% chance of rain." Might not rain.

**Full explanation:**

Traditional software is deterministic: `processPayment(amount=500)` behaves identically every single time, forever. That's why we can write unit tests: input X → assert output Y, always.

ML systems are probabilistic in three distinct ways (understanding all three separates you from other candidates):

1. **The model itself gives probabilities.** A spam model doesn't say "spam" — it says "0.93 probability of spam." Somewhere you (the engineer!) choose a **threshold**: above 0.9 → spam. That threshold is a business decision, not a technical one — lower threshold catches more spam but flags more legit mail (precision/recall — §3.1).

2. **Correctness is statistical, not absolute.** "The model is 96% accurate" means: on data like our test set, it was right 96% of the time. The next single prediction could be wrong. There's no way to know WHICH ones are wrong without ground truth.

3. **Sometimes even the SAME input gives different outputs.** LLMs have a **temperature** parameter (File 02 §1.1) — with temperature > 0, the same prompt can produce different completions on different runs. That's why LLM testing can't be exact-match assertions.

**Why this changes EVERYTHING about system design — the four ripple effects:**

1. **Testing changes.** No unit-test-style exact assertions. Instead: golden datasets, accuracy/precision/recall metrics, and evaluation pipelines.

2. **Monitoring changes.** In deterministic software, if the service is up and errors are 0, output is correct. In ML, the service can be up, errors 0, and 100% of answers garbage. You must monitor OUTPUT QUALITY over time (§2.10) — a concept that simply doesn't exist in traditional SRE.

3. **Rollbacks change.** In normal software you roll back a bad deploy. In ML you must also be able to roll back a bad MODEL or bad DATA (bad labels corrupting training) — which is why model registries and versioning exist (§2.6).

4. **Product decisions change.** ML's error rate must be acceptable to the business. A spell-checker at 98% accuracy: fine. A payment ledger at 98%: catastrophic (2% of millions of transactions = disaster). Part of YOUR job as an AI engineer is advising WHERE probabilistic systems are acceptable.

---

### 1.6 Model Quality vs System Quality

**Plain words:** A great model inside a badly built system still gives a bad product — and a mediocre model in a great system can still win.

**Full explanation via a story:**

Suppose a data scientist delivers a fraud model with 99% accuracy — excellent. The engineering team deploys it. Six months later, fraud losses are UP. Post-mortem findings (each one is a system-quality failure, not a model failure):

- The model takes 900ms to respond. The payment flow only allows 300ms, so callers used a timeout fallback that approved everything → system-quality failure (latency).
- Real-time features were computed differently in production than during training (a timezone bug in "hours since last transaction") → the model received garbage inputs at serving time → system-quality failure (training/serving skew — §2.4).
- The model went live for ALL traffic on day one. A bad interaction with one merchant category caused a wave of false declines before anyone noticed → system-quality failure (no canary rollout — §2.7).
- Nobody monitored output quality. The model drifted for 4 months before anyone looked → system-quality failure (monitoring — §2.10).

The lesson, and the two-sided checklist to recite in interviews:

**Model quality** (data science): accuracy, precision, recall, AUC on held-out data; measured offline; improved with better data, features, architectures.

**System quality** (engineering — YOUR side): latency, throughput, availability, cost per prediction, training/serving consistency, safe rollout, monitoring, rollback capability, graceful degradation.

**One-liner to remember:** *"A model is only as good as the system serving it."* Interviewers at AI labs specifically probe this because most candidates only talk model-side.

---

### 1.7 The Latency vs Accuracy vs Cost Triangle

**The most important mental model in this whole track.** Every ML deployment decision balances three forces, and improving one usually hurts the others.

**Full explanation of each corner and its pull:**

**Latency:** how fast the answer comes. Driven by model size (a 70B model is inherently slower than an 8B one), prompt length (longer prompt = longer prefill — File 02 §2.3), infrastructure (GPU vs CPU, batching), and network hops.

**Accuracy:** how correct the answer is. Driven by model size/capability, data quality, feature quality, and freshness.

**Cost:** money. For hosted APIs: per-token pricing. For self-hosted: GPU hours. Bigger models cost more per request, period.

**Why they conflict — concrete examples of the tug-of-war:**

- Double the model size → accuracy up (usually), latency up, cost up substantially.
- Quantize to 4-bit (File 02 §3.1) → cost and latency down a lot, accuracy down slightly.
- Cache yesterday's answers → latency nearly zero, cost nearly zero, but accuracy decays as the world moves (stale answers).
- Add more retrieved context (RAG) → accuracy up, but latency (bigger prefill) and cost (more input tokens) up.
- Batch more requests together → cost per request down and GPU efficiency up, but each request waits for batch formation → per-request latency slightly up.

**The key insight that makes you sound senior:** there is no universally right balance — the USE CASE sets it. Walk through these three:

1. **Fraud detection:** accuracy (specifically recall — missing fraud costs ₹50,000 per miss) dominates. Accept 500ms and high compute. A false decline annoys a customer; a missed fraud loses money.
2. **Search autocomplete:** latency dominates. A user typing sees suggestions appear in <100ms; at 2 seconds they've typed the whole query and the feature is worthless. A slightly worse suggestion delivered instantly beats a perfect one delivered late.
3. **Nightly credit-risk re-scoring (batch):** cost and accuracy dominate, latency is irrelevant (it runs at 2 AM for 6 hours).

**Interview answer template:** "I'd start by asking what the use case punishes most — wrong answers, slow answers, or expensive answers — then optimize for that dimension and measure the other two as guardrails, not goals." Then give one of the three examples above.

**Memory hook:** *Fast, Right, Cheap — pick two.*

---

### 1.8 Model Drift

**Plain words:** The world changes, your model doesn't — so it slowly becomes wrong.

**Full explanation with the classic story:**

A fraud model was trained in 2022, reaching 97% accuracy. It runs, unchanged, for two years. Accuracy quietly falls to 89%. Nothing crashed. No code changed. No data pipeline broke. What happened?

Fraudsters ADAPTED. They watched what got flagged, changed their patterns — smaller amounts, new merchant categories, new device farms. The model's learned patterns ("late-night high-value electronics purchases are suspicious") became outdated because the underlying reality moved.

**The taxonomy (know both names):**

- **Concept drift:** the relationship between inputs and correct output changed. What "suspicious behavior" MEANS has shifted. Example: during COVID, "sudden spike in online transactions" became normal for everyone — a model trained pre-COVID would have flagged half the country.
- **Data drift:** the input distribution changed — even if the input→output rule is the same. Example: you launch in a new geography and suddenly 30% of transactions come from a region never seen in training. The model's knowledge simply doesn't cover that region.

**How drift manifests in metrics (what you'd actually see on dashboards):**
- Prediction distribution shifts (model used to flag 2% of transactions; now flags 8%).
- Input feature distributions shift (average transaction amount climbs).
- When delayed ground truth arrives (§2.10), accuracy/precision/recall are down vs. a stable baseline.

**How you catch it:** monitoring (§2.10) — compare input distributions and prediction rates against a reference window (last week vs. 3 months ago), alert on significant shifts, and schedule periodic accuracy recomputation against fresh labeled data.

**How you fix it:** retrain on recent data (most common), sometimes add features (new signals the old model lacked), sometimes redesign (when the concept shifted so hard that the old feature set is obsolete).

**Analogy:** a GPS map from 2015. It still *functions*, but new roads exist and old ones closed. It needs map updates. Software doesn't rust; models do.

---

### 1.9 Data Drift

**Plain words:** The input data itself changes over time, even if the model's logic is still fine.

**Full explanation with numbers:**

A house-price model trained in 2020, when the average transaction in its training data was ₹50 lakh. By 2026, prices doubled: typical inputs are now ₹1 crore. Two failure modes:

1. The model has never SEEN ₹1-crore inputs — it extrapolates poorly, producing nonsense for exactly the most common cases now.
2. Even where it interpolates, its learned "adjustment factors" (location premium, size premium) were calibrated to a price level that no longer exists.

**Data drift vs concept drift — the clean way to tell them apart in an interview:**
- **Data drift** = the QUESTIONS changed (inputs moved).
- **Concept drift** = the correct ANSWERS for the same questions changed (the input→output relationship moved).

Example that shows both: a food-delivery ETA model. City adds a metro line (travel patterns change → data drift: inputs differ). Simultaneously, a new traffic rule halves speeds on major roads (same input — same distance, same hour — now has a different correct ETA → concept drift). Both require retraining; diagnosing which one (or both) tells you what data to prioritize.

**The general principle:** drift detection IS a monitoring problem. You cannot fix what you don't measure. Alert on input-distribution shifts (cheap, real-time — no labels needed) as an EARLY WARNING, and label-based accuracy drops as the confirmation. Data drift is detectable before concept drift because it doesn't need ground-truth labels — pure input statistics suffice.

---

## Part 2: Production ML Architecture (The Full Lifecycle)

The complete lifecycle. Memorize this assembly line, because every ML system design answer walks along it:

```
Data Collection → Data Validation → Feature Engineering → Feature Store
      → Training Pipeline → Model Registry → Deployment
      → Online/Batch Inference → Monitoring → Retraining (loop back to top!)
```

Note it's a LOOP, not a line: monitoring feeds retraining, retraining feeds deployment. ML systems are never "done" — they're maintained, like machines, not shipped, like apps.

---

### 2.1 Data Collection

**Plain words:** Gathering the raw examples your model will learn from.

**Full explanation:**

Sources, in practice: user activity logs (clicks, transactions, searches), application databases, event streams (Kafka), third-party datasets, and human labeling (annotation teams or ops teams labeling edge cases).

**The single most important fact about data collection (interview gold):** *data quality beats model sophistication.* A simple logistic regression on clean, well-labeled, representative data beats a deep neural network on messy data — this has been proven repeatedly in industry and academia. When someone proposes "let's try a bigger model," the senior engineer first asks "what's wrong with our data?"

**The failure modes of collection (each one is a war story you can cite):**

1. **Duplicate data** — the same example appears 10 times; the model over-weights it.
2. **Missing values** — a sensor was down for a week; 8% of rows have nulls.
3. **Biased samples** — trained fraud data from urban users only; deployed nationally; rural fraud patterns are invisible to the model.
4. **Label noise** — humans labeling "was this transaction fraudulent?" disagree with each other 15% of the time. Your "ground truth" is 15% wrong.
5. **Leakage** — a feature that accidentally contains the answer (a "disputed_by_user" flag present for fraud cases only — the model learns to detect disputes, not fraud; looks amazing in training, useless in production).

All of these are silent: training runs "successfully" and produces a model anyway. That's why the NEXT stage exists.

---

### 2.2 Data Validation

**Plain words:** Checking the collected data BEFORE training — like checking ingredients before cooking.

**Full explanation:**

**The one rotten tomato principle:** one bad data batch can silently corrupt a model. Since training doesn't crash on bad data — it just learns wrong patterns — validation must catch what training can't.

What validation checks, concretely:

1. **Schema conformance:** do fields have expected types and ranges? `age` must be 0–120; a value of 300 or -1 means an upstream bug.
2. **Completeness:** % of nulls per column. A feature that jumped from 0.1% nulls to 40% nulls this week means a producer broke.
3. **Distribution sanity vs. reference:** compare this batch's statistics to last month's. Mean transaction amount suddenly 3× higher? Either the world changed (real) or the currency unit changed in some pipeline (bug). Automation should flag it either way; a human decides.
4. **Duplicates and label quality:** deduplication stats; sample-based label audits (humans re-check 500 random labels to estimate noise rate).

**The process point that impresses:** validation should be an automated GATE in the training pipeline, not a notebook someone runs when things look weird. Batch fails checks → pipeline stops → alert. Data quality gates are CI/CD for ML.

(TFX — Google's production ML framework — became famous largely for making "data validation as a pipeline stage" mainstream. Name-drop if asked.)

---

### 2.3 Feature Engineering

**Plain words:** Converting raw data into meaningful numbers the model can learn from. Models eat numbers, not meaning.

**Full explanation with a worked example:**

Raw record: `{time: "2026-09-26T23:41:03", amount: 12499, merchant: "ELECTRONICS_HUB", device_age_days: 47}`

Engineered features the model can actually use:
- `hour_of_day = 23`, `is_weekend = 0`, `is_night = 1` (cyclical time encoded as useful flags)
- `amount_vs_user_avg = 4.2` (this transaction is 4.2× this user's average — a RATIO is far more informative than the raw amount)
- `merchant_category_risk_score = 0.31` (from historical fraud rates by category)
- `hours_since_last_txn = 0.2` (12 minutes — rapid succession is a signal)

**Why this step decides outcomes:** the model can only learn from patterns REPRESENTED in its features. "amount_vs_user_avg" lets it learn "transactions far above a user's normal are riskier" — a raw amount column can't express that cleanly. Good features encode domain knowledge (what fraud analysts know matters).

**Rules of thumb to quote:**
- Ratios and relative values usually beat raw values.
- Domain knowledge should drive feature ideas, not blind enumeration.
- Fewer, well-designed, validated features often beat hundreds of noisy ones.
- For deep learning on text/images, the model learns internal representations — feature engineering matters less there (this nuance shows you understand WHERE classic ML vs deep learning differ).

---

### 2.4 Feature Stores

**Plain words:** A central, shared store of ready-to-use features, used by BOTH training and serving.

**Full explanation — built around the problem it solves, which is the interview gold:**

**The problem: training/serving skew.**

During training (offline, batch), a data scientist computes `hours_since_last_txn` from nightly logs with a certain SQL script. During serving (online, real-time), an engineer implements the SAME feature in the low-latency service — but in different code, written months later, by a different person.

Subtle differences creep in:
- Training version uses transaction time in UTC; serving version uses the server's local timezone. One hour off.
- Training version includes ALL transactions; serving version can only see what's in its cache.
- Training uses end-of-day state; serving uses point-in-time state mid-stream.

The model was trained on feature values that differ systematically from what it receives at serving time. Accuracy drops in production while offline scores look fine — one of the nastiest bugs in ML, because NOTHING is visibly broken.

**The solution:** compute each feature ONCE, in ONE definition, store it centrally; both the training pipeline and the online service READ from the same store. One definition → no skew. Additional wins:
- **Feature reuse across teams:** fraud team and credit team share `user_avg_txn_30d` instead of each computing (and each subtly buggering up) their own.
- **Point-in-time correctness:** the store can return feature values AS OF a past moment, so training uses the values that would have existed then — no leakage from the future.

**Analogy:** central water tank vs. every house digging its own well. One source of truth, consistent pressure everywhere.

---

### 2.5 Training Pipelines

**Plain words:** Automated, repeatable machinery that takes data in and produces a trained model out.

**Full explanation:**

Why "pipeline" and not "a script someone runs": over a model's life you will train hundreds of versions — every retrain cycle (§1.8–1.9), every experiment, every A/B candidate. That requires:

- **Reproducibility:** any past training run can be recreated exactly — same data snapshot, same code commit, same hyperparameters. When someone asks "why did the March model behave differently?", you must be able to rebuild and diff.
- **Automation & scheduling:** triggered by schedule (weekly), by event (data drift alarm), or by human (experiment). Tools: Kubeflow Pipelines, MLflow, Airflow, Metaflow.
- **Versioned inputs:** the pipeline snapshots its input data (or records its exact provenance), so "trained on data as of Sept 1, 2:00 AM" is a queryable fact.
- **Structured outputs:** each run emits evaluation metrics + artifacts (the model file) + metadata to the model registry (next).

**The senior-engineer framing:** training pipelines are CI/CD for models — versioned, automated, test-gated (evaluation thresholds block bad models from registry promotion), observable. Say that sentence, and the concept is fully communicated.

---

### 2.6 Model Registry

**Plain words:** A locker room for models — a central catalog of every trained model with full metadata.

**Full explanation:**

What's stored per model: version number, which training run produced it (code commit, data snapshot, hyperparameters), evaluation scores (on which datasets), who trained it, when — and its LIFECYCLE STATUS: `staging` (candidate) → `production` (serving traffic) → `archived` (previous version, kept warm).

**Why it exists — the two questions it answers:**

1. *"Something's wrong in production — which model is live, what was it trained on, and what did we change?"* Without a registry, the answer is "ask Rajesh, he ran the notebook in March." With a registry, it's a query.
2. *"Roll back!"* When a new model misbehaves, you repoint serving to the previous version — which is still sitting in the registry, tested and ready.

**Why rollback without a registry can force RETRAINING (the failure chain, step by step):**
1. **The old weights may simply not exist anymore.** A registry systematically stores model ARTIFACTS (the trained weights as files, often hundreds of GB for big models) + METADATA (which data version, which code, which training run, what metrics). Without one, nobody systematically keeps old versions — the new deploy may have overwritten the only copy, and the previous version's files are gone.
2. **Even if a weights file exists somewhere, you can't trust it.** Is it really the v4.2 that scored 96.1%? Which data was it trained on? Without lineage (the registry's metadata), deploying that file is a gamble, not a rollback.
3. **Reproduce = retrain.** If you can't locate or trust the old artifacts, the only way to get a KNOWN-GOOD model is to re-run the training pipeline with the recorded data and code versions — hours to days, while production serves a broken model.

**The analogy:** deployments without version control. If you only ever copy the "latest" build over the previous one and it breaks, you can't go back — you must rebuild from source. Git is the registry for code; the model registry is git for weights. So the sentence's real meaning: *registry = the old version is guaranteed stored, versioned, tested, and one repoint away. No registry = you're hunting for files and hoping — and in the worst case, retraining.*

**Real registries:** MLflow Model Registry, SageMaker Model Registry, Vertex AI Model Registry.

---

### 2.7 Model Deployment

**Plain words:** Making the new model available to serve real requests — safely.

**Full explanation — the four strategies, when to use each:**

**1. Big-bang (replace for everyone at once):** simplest, fastest. Riskiest: a bad model hits 100% of users before you can react. Only acceptable for low-stakes models or when the new model's offline evaluation is overwhelmingly better AND you have instant rollback ready.

**2. Canary:** the new model serves 1% of traffic (or 1% of users — sticky per user!) while the old model serves 99%. Watch quality + latency + cost metrics for an hour/a day. Healthy → ramp to 5% → 25% → 100%. Unhealthy → automatic rollback to the old model (which is why the registry keeps it warm — §2.6). The sticky-per-user part matters: a user shouldn't get model A on one request and model B on the next (consistency of experience, cleaner A/B reads).

**3. Shadow (dark launch):** the new model runs in PARALLEL on the same live traffic — but its answers are logged and compared, NEVER shown to users. Users still get the old model's answers. Zero user risk, full production realism. Cost: you pay double compute during shadowing. Use when: model change is risky, you don't trust offline evaluation, or you need production data to compare behaviors (e.g., "how often do new and old models disagree, and who's right when they do?").

**4. Blue-green:** two full environments (blue = current, green = new). Switch all traffic at once — but keep blue running warm — so rollback is an instant traffic switch back. Useful when you can't do gradual (stateful systems, protocol changes) and need fast undo.

**Why ML specifically needs canary/shadow (the deep reason):** offline evaluation uses offline datasets — however good, they never fully match production's distribution (new query types, adversarial inputs, weird edge cases). You CANNOT fully test an ML model offline; the last stage of testing must happen on real traffic, safely. Shadow = test on real traffic with zero user exposure; canary = expose real traffic in a controlled dose. Both exist because "test in staging" is structurally insufficient for ML.

---

### 2.8 Online Inference (production view)

The model answers requests in real time via an API. What serving entails, concretely: a low-latency service hosting the model (or calling a hosted API), replicated and load-balanced, with autoscaling on request volume, per-request timeouts, fallback behavior on overload (queue or reject), and per-request logging for monitoring. Latency budget matters: total time = network + feature fetch + model inference + post-processing — the model is often only part of the budget, and the feature-fetch hop (feature store lookup) is a real component of request latency.

### 2.9 Batch Inference (production view)

A scheduled job scores thousands/millions of records and writes results to a database or cache. Cheaper per prediction (optimal batching, off-peak hours, bigger models allowed), but answers are stale between runs. The freshness requirement (§1.3) decides if this is acceptable.

### 2.10 Monitoring (ML-specific)

**Plain words:** Watching the model's health in production — not just "is the service up" but "is the model still GOOD?"

**Full explanation — the four layers, each catches different problems:**

**Layer 1 — System metrics (classic SRE):** latency (p50/p95/p99), error rates, throughput, CPU/GPU utilization, memory. Catches: infrastructure failures. Same as any service.

**Layer 2 — Data metrics (input health):** distributions of input features vs. a reference window; null rates; out-of-range rates; prediction-score distributions. Catches: DATA DRIFT (§1.9) — often the earliest warning, and needs NO ground-truth labels (cheap, real-time). If the model's score distribution suddenly shifts, either the world changed or a producer broke.

**Layer 3 — Model quality metrics (output health):** accuracy/precision/recall — but computed against GROUND TRUTH, which arrives LATE. For fraud: you learn a flagged transaction was truly fraudulent only weeks later (investigation concludes). For recommendations: did the user click? (immediate but weak signal). This layer catches CONCEPT DRIFT (§1.8) and silent model degradation.

**Layer 4 — Business metrics (the ultimate truth):** fraud losses per month, conversion rate, revenue per session, CSAT. What the business cares about; what ultimately justifies the model. A model can hold 97% accuracy while the business metric degrades (the drift hit the 3% that mattered most).

**The monitoring loop that ties it together:** Layers 1–2 watch in real time (cheap, immediate, no labels). Layer 3 recomputes quality when labels arrive (delayed but precise). Layer 4 validates continuously. Any anomaly at any layer → investigate → likely retrain (feeding the lifecycle loop).

**Monitoring before deployment vs after:** before launch, define the baselines and alert thresholds from offline evaluation + shadow runs; after launch, track deviation from baseline. "You can't detect drift without a reference point" — a monitoring design starts by capturing what NORMAL looks like.

### 2.11 Retraining

**Plain words:** When monitoring shows drift, or a schedule fires, train a fresh model on recent data, evaluate it, and deploy via canary.

**Full explanation:**

Triggers for retraining, in order of sophistication:
1. **Scheduled** (simplest): weekly/monthly, whether needed or not. Wasteful but safe and predictable.
2. **Metric-triggered** (monitoring-driven): accuracy fell below threshold, or input drift alarm fired — retrain NOW.
3. **Continuous training** (advanced, e.g., Netflix/Uber style): pipelines automatically retrain and re-evaluate, promoting new models only if they beat the current one on golden datasets.

**Retraining vs fine-tuning — the distinction interviewers check:**
- **Retrain (from scratch):** fresh random start, learn everything from new data. Thorough but expensive. Common for classic ML models (trees, regressors) where full retraining is cheap.
- **Fine-tune (from existing weights):** start from the current model's weights, continue training on fresh data. Cheap, fast. Common for deep learning and LLMs.

**"Train on WHAT data?" — the follow-up you WILL get (old model was trained on X; new data collected since is Y — which do you feed?):**

**The answer: almost always X+Y (full accumulated history), not Y alone.** Retraining from scratch on all accumulated data is the default; training only on fresh data is "continual/online learning," which exists but has problems. The reasoning, point by point:
1. **Y alone is too narrow.** If the last 3 months of data is all you train on, the model forgets everything else — a fraud model trained only on last quarter's fraud patterns loses the long tail of older fraud types that still occur. Models need volume; Y alone is usually a fraction of X.
2. **Old patterns still matter.** Drift means the distribution SHIFTED, not that the past became irrelevant. Diwali spikes, year-end tax scams — seasonal patterns from years ago repeat; you need that history.
3. **Catastrophic forgetting.** Fine-tuning repeatedly on only-new data makes neural networks literally forget old knowledge — a well-documented failure mode. Retraining from scratch on X+Y sidesteps it entirely.

**Practical refinements on the X+Y theme (each solves a real problem with "literally all data ever"):**
- **Rolling window:** X+Y where "X" is a trailing window (e.g., last 24 months), not literally everything — balances recency against stale data and storage cost.
- **Recency weighting:** keep all of X but weight recent rows more heavily (e.g., 0.9 decay per quarter) — recency without amnesia.

**When Y-only (continual learning) IS used:**
- **Online learning** (streaming fraud detection): tiny frequent updates on fresh data only — but these are INCREMENTAL updates to the existing model, not fresh training, and they're paired with drift monitors precisely because they can drift badly.
- **Warm-start fine-tune:** take the existing model and fine-tune briefly on Y — cheap, but risks forgetting; usually needs rehearsal data (a replay mix of old samples + Y) to prevent it.

**The one-liner to memorize:** *"Retraining re-runs the full pipeline on accumulated data — typically a rolling window of all history, sometimes recency-weighted — because training on only-new data makes the model forget the long tail. Pure new-data-only training is continual learning: cheaper, but it needs replay buffers and drift guards."*

And note the vocabulary subtlety: for LLMs, "fine-tuning" usually means adapting a PRE-TRAINED model to a specific task/style (File 02 §4.3) — different context, same word. Clarify which meaning is intended when asked.

**The deployment-safe rule:** retrained model → full evaluation on golden datasets → canary rollout (never big-bang) → monitor → promote or rollback. The loop closes.

## Part 3: ML Metrics

### 3.1 Offline Metrics (measured BEFORE deployment, on held-out test data)

**The setup you must explain first:** when training, data is split into three parts:
- **Training set** (~70–80%): the model LEARNS from this.
- **Validation set** (~10–15%): used DURING development to tune hyperparameters and compare candidate models.
- **Test set** (~10–15%): touched exactly ONCE, at the end, to estimate real-world performance. If you reuse the test set to pick models, you've "contaminated" it — the model has effectively seen it, and your score is optimistically biased. This mistake is called **overfitting to the test set**, and mentioning it is interview gold.

**Accuracy — and why it lies (the most instructive example in all of ML metrics):**

Suppose only 1% of transactions are fraudulent. Build a "model" that says "not fraud" for EVERYTHING. Accuracy = 99%! Useless model. This is the **class imbalance problem** — when one class dominates, accuracy is meaningless, and precision/recall must be used instead.

**Precision vs Recall — the classic question, fully explained:**

Fraud model, 100 transactions, 10 actually fraudulent.
- Model flags 12 as fraud. Of those 12, 8 are truly fraud → **Precision = 8/12 = 67%.** Plain words: "When I raise the alarm, how often am I right?"
- Of the 10 real frauds, the model caught 8 → **Recall = 8/10 = 80%.** Plain words: "Of all the actual fraud, how much did I catch?"

The costs, concretely:
- **Miss a fraud (low recall)** → money lost directly.
- **Flag a genuine customer (low precision)** → angry customer, blocked payment, support cost.

**The trade-off knob (understand this mechanism):** the model outputs a fraud PROBABILITY. You choose the threshold:
- Threshold low (flag at 0.3): catch almost all fraud (recall ↑) but many false alarms (precision ↓).
- Threshold high (flag at 0.9): alarms are trustworthy (precision ↑) but fraud slips through (recall ↓).
You cannot move both in the same direction for free — the threshold IS the business trade-off dial. Where you set it depends on costs: for a $2 promotional abuse, tolerate misses (high threshold); for a ₹50 lakh wire transfer, tolerate false alarms (low threshold + human review).

**F1 score:** the harmonic mean of precision and recall — a single number that punishes either one being terrible. Use when you need one number and classes are imbalanced. (Harmonic mean, not average: precision 100% + recall 0% → F1 = 0, correctly signaling uselessness.)

**Confusion matrix (the underlying bookkeeping):** a 2×2 table — True Positives (caught fraud), False Positives (false alarms), False Negatives (missed fraud), True Negatives (correctly passed). Every metric above is arithmetic on these four cells. Drawing this in an interview instantly organizes the discussion.

**AUC-ROC (know it conceptually):** the model's ability to RANK frauds above non-frauds across ALL thresholds. 0.5 = coin flip, 1.0 = perfect ranking. Useful because it's threshold-independent — it evaluates the model before you've chosen the business dial.

### 3.2 Online Metrics (measured AFTER deployment, on real users)

Click-through rate, fraud actually caught (confirmed by investigators), real conversions, live latency percentiles. Ground truth from the real world.

**Why both offline AND online:** a model scoring 97% offline can behave differently in production — drift between test set and reality, feedback loops (users react to model outputs, changing future data), and interaction effects. Offline = dress rehearsal; online = opening night. Both needed, and when they disagree, believe online (and investigate the gap).

### 3.3 Business Metrics

Revenue, fraud losses prevented, retention, CSAT — what the company actually cares about. The senior move: CONNECT the chain. "Model metric: recall +3% → Business metric: ₹40 lakh/month fewer fraud losses → justifies the 20% compute increase." Engineers who can walk this chain get trusted with model decisions.

### 3.4 A/B Testing for ML

**Plain words:** Show old model (A) to half the users, new model (B) to the other half, compare results fairly.

**Full mechanics:**
1. **Random split:** each user randomly assigned to A or B — randomness kills selection bias (power users landing in one group by chance would skew results).
2. **Statistical significance:** if B is 1% better, is that real or luck? Run with enough users for enough time; compute a p-value (or use a proper sequential testing framework). Small differences need BIG samples. Stopping early when the numbers "look good" is the classic A/B testing sin (peeking).
3. **One change at a time:** if you deploy a new model AND a new UI together, you can't attribute the improvement.

**ML-specific traps (this is where you earn points):**
- **Non-determinism:** same model, same user, different days → different outputs. Adds variance; requires larger samples than deterministic-feature A/B tests.
- **User stickiness:** keep a user in ONE group for the experiment's duration — flipping users between experiences mid-test poisons both measurement and UX.
- **Feedback loops:** recommender A changes what users click, which changes future data — can slowly self-reinforce (rich-get-richer). Long-run effects can diverge from short-run A/B reads.
- **Proxy metric trap:** optimizing CTR can degrade long-term satisfaction (clickbait wins A/B, users churn later). Pair short-term proxies with guardrail metrics (retention, complaints).

---

## Part 4: Case Study — Design Uber ETA Prediction

**Follow this 9-step structure in ANY ML design question.**

**1. Clarify requirements:**
- Which ETA? Pickup ETA (driver to you) vs trip ETA (drop-off) vs both? Both, but start with pickup.
- Latency: user staring at a screen after tapping — needs sub-second response, real-time (online inference).
- Accuracy target: typical MAE tolerance is 1–2 minutes; being systematically wrong (always 5 min late) destroys trust more than random error.

**2. Prioritize the triangle:** latency and accuracy both critical (screen-waiting user; broken promise = angry user). Cost: millions of ETA requests/day favors efficient models — and per-city models rather than one global model (traffic in Mumbai ≠ Jakarta ≠ SF).

**3. Training data (historical):** GPS traces of completed trips (the ACTUAL durations — ground truth), timestamps, pickup/drop coordinates, traffic speeds per road segment, weather, day/time, driver history.

**4. Features:** distance, hour-of-day, day-of-week, historical average speed on that route segment at that hour, weather, city, number of traffic signals. Note the blend: static features (route distance — precomputable) + real-time features (live traffic — fetched at request time). That split is a system-design decision as much as an ML one.

**5. Model:** gradient-boosted trees (XGBoost-style) on tabular features — the strong, classic, cheap baseline for this data shape. Deep learning only if using raw map sequences. Interview tip: naming the simple strong baseline FIRST and justifying it shows maturity; jumping to "transformers" shows the opposite.

**6. Serving:** online inference service, model replicas behind a load balancer, low-latency feature fetches — feature store territory (§2.4): training and serving must compute "avg speed on segment X at hour Y" identically.

**7. Evaluation:** offline MAE (mean absolute error in minutes — explainable to business) on held-out completed trips → shadow deploy against current ETA model → canary → A/B test vs old model (user cancellation rate, app-abandonment as guardrails).

**8. Monitoring & retraining:** ground truth arrives when trips COMPLETE (delayed labels — Layer 3 monitoring, §2.10). New roads, monsoon season, new traffic rules = drift → scheduled retraining + drift alarms on input features (rain spikes = input shift).

**9. Failure scenarios (volunteer these):**
- Driver GPS off → garbage trace data → data validation catches it (§2.2).
- New city launch → no history = cold start → fall back to map-distance ÷ speed-limit estimate until data accumulates.
- Concert/strike/monsoon → traffic the model never saw → real-time features must carry the signal; cap absurd predictions with sanity rules (ETA > 2× normal → flag).

---

## Tough Interview Follow-ups (with full answers)

**Q: "Your fraud model's accuracy dropped from 97% to 90% in 2 months. Debug it."**
A: First, segment: WHERE did it drop — one geography, one merchant category, one amount range? The segment localizes the cause. Then check the three usual suspects in order:
1. **Data drift:** did inputs change? (new region launch, new product type) → check input distributions vs. 3 months ago (Layer 2 monitoring).
2. **Concept drift:** did fraudsters adapt? → look at the fraud cases you MISSED; are they a new pattern? (Layer 3, using delayed labels.)
3. **Pipeline bug:** did a feature computation change upstream? (schema change, timezone, unit change) → compare feature distributions and spot-check raw values.
Fix depends on cause: retrain on recent data (drift), add features for the new pattern (concept), or fix the pipeline (bug). The discipline: diagnose BEFORE retraining — blind retraining on shifted data can bake the problem in.

**Q: "How do you monitor a model when ground truth is delayed?"**
A: Use proxy signals until truth arrives. Fraud model: monitor the SCORE distribution, alert rate, user complaints, analyst overturn rates — all real-time. When actual labels arrive weeks later, recompute true precision/recall (Layer 3). Additionally monitor input distributions continuously (Layer 2) since drift is detectable before accuracy drops are confirmable. Two-speed monitoring: fast proxies + slow truth.

**Q: "When would you NOT use ML?"**
A: When rules suffice. If a domain expert can write 20 rules covering 95% of cases, start there — cheaper, explainable, debuggable, instant to change. Use ML when the pattern is too complex or evolving for rules (fraud, recommendations, language). Also avoid ML where errors are catastrophic AND unexplainable behavior is unacceptable (core payment ledger). The mature answer includes the hybrid point: rules for the extremes, ML for the fuzzy middle, rules as guardrails around ML output (amount > ₹10L → human review regardless of score).

**Q: "What's the first thing you'd build for a new ML feature: the model or the pipeline?"**
A: The pipeline — data collection, validation, feature computation, and MONITORING. Because: (1) data quality determines model ceiling (§2.1); (2) you need the pipeline to even evaluate models properly; (3) without monitoring you can't trust production behavior later. Model iteration on top of a solid pipeline is fast; the reverse is a science fair project.

---

## Rapid-Fire Flashcards

1. Training vs inference? → Studying (adjust weights, expensive, rare) vs exam (frozen weights, fast, constant).
2. Batch vs online inference? → Cook everything at 2 AM vs cook to order. Decided by freshness need + volume pattern.
3. Deterministic vs probabilistic? → Guaranteed identical output vs probability + threshold; testing/monitoring/rollback all change.
4. Latency/accuracy/cost? → Fast, Right, Cheap — pick two; the USE CASE picks which.
5. Data drift vs concept drift? → Questions changed vs correct answers changed; drift detected via input stats (no labels needed) vs label-based accuracy.
6. Feature store solves? → Training/serving skew — one definition, one source of truth, both read from it.
7. Model registry? → Git for models: versions, lineage, lifecycle states, instant rollback.
8. Canary? → 1% traffic, sticky per user, auto-rollback; ramp 1→5→25→100.
9. Shadow? → New model runs parallel on live traffic, logged not served; zero risk, double cost.
10. Precision vs recall? → When I alarm, how often right (8/12) vs of all real fraud, how much caught (8/10).
11. Threshold = ? → The business trade-off dial between precision and recall.
12. Why accuracy lies? → Class imbalance: "always not-fraud" scores 99%.
13. F1? → Harmonmonic mean; punishes either being zero.
14. Offline vs online metrics? → Dress rehearsal vs opening night; when they disagree, believe online.
15. A/B testing rules? → Random split, significance (no peeking), one change, sticky users.
16. Monitoring layers? → System, data (drift, no labels), model quality (delayed labels), business (truth).
17. Retraining triggers? → Scheduled, metric-triggered, continuous.
18. Retrain vs fine-tune? → From scratch (thorough, costly) vs from existing weights (cheap, fast).

---
