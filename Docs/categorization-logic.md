# Categorization Logic: Two-Layer Theme Discovery

> **For the coding agent.** This spec defines how scraped Play Store reviews are turned into exploration-framed themes. It sits between the clustering stage and the insight/reporting stage in `architecture.md`. Implement exactly as described — the two layers must stay separate.

---

## Why two layers

Raw clustering alone produces generic operational buckets ("late delivery", "app crashes") that don't explain **why users don't explore new categories** — which is the growth goal. So categorization runs in two layers:

- **Layer 1 — open, data-driven topic clusters:** *what the review is about.* No predefined categories.
- **Layer 2 — a fixed barrier lens applied over those clusters:** *why the topic blocks category exploration.*

Rule: **open clusters, fixed lens.** Predefine only Layer 2. If both layers are predefined you only find what you went looking for; if neither is, you get topics that don't answer the exploration question.

---

## Layer 1 — Topic clusters (already in the pipeline)

Reuse the existing flow, no changes:

1. Embed each review (local/free embedding model).
2. kNN similarity graph over embeddings → `SIMILAR` edges.
3. Louvain community detection → topic communities.
4. Each community is a raw topic (e.g. stockouts, delivery fees, product quality, support failures, search/discovery friction).

Output per cluster: `cluster_id`, member review IDs, size.

---

## Layer 2 — Barrier mapping (new)

Each Layer-1 cluster is tagged with exactly **one primary barrier** from this **fixed, closed taxonomy**. Define it before running the mapping; do not let the model invent new labels.

| Barrier label | Definition (what it means for exploration) |
|---|---|
| `trust_risk` | Won't try an unfamiliar product because quality/authenticity is unpredictable — near-expiry, opened seal, wrong/damaged item, counterfeit fear. |
| `economic` | Delivery fees / free-delivery thresholds / price make trying a non-essential or new-category item not worth it. |
| `reliability` | Late, missing, or cancelled orders push users back to safe, known staples. |
| `discovery` | Search and browse surface the same known items; new categories never get seen (out-of-stock items buried below ads, no relevant recommendations). |
| `recovery` | Weak support / refund friction after a failure kills willingness to experiment again. |
| `habit_load` | Reordering the usual is effortless; exploring costs cognitive effort, so users default to the familiar basket. |

Notes:
- A review can touch several barriers; the **cluster** gets one *dominant* label for counting. Keep secondary barriers in a `secondary_barriers` list for richer analysis, but count on primary only.
- If a cluster genuinely fits none (e.g. pure app-performance bug unrelated to exploration), tag `out_of_scope` — don't force-fit. These are excluded from theme counts but kept for the appendix.

---

## The mapping step (LLM, per cluster)

When the LLM summarizes each Layer-1 cluster, it emits the barrier mapping in the **same call**. This keeps the mapping auditable and produces the citations the chat page needs.

Input to the model per cluster:
- `cluster_id`
- A representative sample of member reviews (e.g. 15–25, sampled to span the rating range and review lengths in the cluster, not just the top-upvoted ones).
- The fixed barrier taxonomy above (paste it into the prompt verbatim).

### Prompt template

```
You are analyzing clusters of Blinkit (Indian quick-commerce app) Play Store
reviews. The business goal is to understand what stops users from exploring
NEW product categories on the app.

You will be given one cluster of related reviews. Do the following and return
ONLY valid JSON, no preamble, no markdown fences.

1. theme_name: a short (<=6 word) human-readable name for what this cluster is about.
2. summary: one sentence, plain and specific, describing the shared complaint or topic.
3. primary_barrier: the SINGLE label from the fixed list below that best explains
   how this topic blocks users from trying new categories.
4. secondary_barriers: array of 0-2 other labels from the list that also apply (may be empty).
5. barrier_justification: one sentence explaining why you chose the primary barrier,
   grounded in the reviews (not generic reasoning).
6. representative_quotes: array of 2-3 verbatim review snippets (exact text, each with
   its review_id) that best illustrate the theme. Do not paraphrase.
7. confidence: "high" | "medium" | "low" — how cleanly the cluster maps to one barrier.

FIXED BARRIER LIST (choose only from these exact labels):
- trust_risk: won't try unfamiliar products because quality/authenticity is unpredictable
- economic: fees/thresholds/price make trying a non-essential or new item not worth it
- reliability: late/missing/cancelled orders push users back to safe staples
- discovery: search/browse surface only known items; new categories never get seen
- recovery: weak support/refunds after a failure kills willingness to experiment again
- habit_load: reordering the usual is effortless; exploring costs effort, so users default to the familiar
- out_of_scope: does not relate to category exploration at all

If the cluster fits none of the exploration barriers, set primary_barrier to
"out_of_scope". Never invent a label outside this list.

CLUSTER REVIEWS:
{sampled_reviews_with_ids}
```

### Expected output schema (per cluster)

```json
{
  "cluster_id": "c_017",
  "theme_name": "Missing items on delivery",
  "summary": "Users repeatedly receive orders with items missing and no proactive fix.",
  "primary_barrier": "trust_risk",
  "secondary_barriers": ["recovery"],
  "barrier_justification": "Repeated missing items make users doubt reliability of anything beyond their proven staples.",
  "representative_quotes": [
    {"review_id": "r_88213", "text": "..."},
    {"review_id": "r_90441", "text": "..."}
  ],
  "confidence": "high"
}
```

---

## Aggregation → theme table

After every cluster is mapped:

1. Group clusters by `primary_barrier`.
2. Sum member review counts per barrier → the support count for the page-2 narrative (e.g. `trust_risk — 4,873`).
3. Within each barrier, keep the individual `theme_name`s as sub-themes so the report can drill down.
4. Carry `representative_quotes` up so every headline number is backed by real verbatims (feeds the RAG/synthesis chat page's citations).

This is what produces lines like "Order issues kill trust in unfamiliar products — 4,873" instead of a flat topic tally.

---

## Guardrails

- **Taxonomy is frozen at mapping time.** No new barrier labels mid-run. If a clear pattern doesn't fit, log it for a taxonomy review *before* the next full run — never silently.
- **Sample, don't cherry-pick.** Cluster samples fed to the LLM must span the rating range, or trust/reliability themes will be over-weighted by 1-star reviews.
- **Count primary only.** Secondary barriers enrich analysis but double-counting inflates every theme.
- **Keep `out_of_scope` visible.** Don't delete it — showing what was excluded is part of the validation story for the deck.
- **Every count is auditable.** Each barrier total must trace to cluster IDs → review IDs → verbatims. No number appears in the report without that chain.
