# Architecture: AI-Powered Discovery Engine

> **Purpose:** Project plan and architecture for the Blinkit review-analysis pipeline. Maps the requirements in `problemstatement.md` and the context in `context.md` to concrete modules, data flow, and file layout. Build-order item 1 of 6.

---

## 1. Design Principles

Every decision traces back to the spec constraints:

- **Zero cost / fully local** — no paid APIs or models. Local embeddings, local/free LLM (Groq), free scraping lib.
- **Single source (amended §12)** — Google Play Store reviews only, originally. §12 documents a deliberate, user-directed exception adding one specific second source (Mouthshut) - not a general opening to arbitrary connectors.
- **Question-driven** — the pipeline exists to answer the 8 research questions; the insight layer maps every theme back to them.
- **Whole-funnel** — operational/pricing complaints are categorized, not filtered out.
- **Reproducible** — deterministic where possible (fixed seeds, pinned deps, cached artifacts); a fresh environment can run it end-to-end.
- **Artifact-based stages** — each stage reads a file and writes a file, so stages are independently re-runnable and debuggable.

---

## 2. High-Level Data Flow

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                    DISCOVERY ENGINE                        │
                    └──────────────────────────────────────────────────────────┘

  [Google Play]        Stage 1                               Stage 2             Stage 3
      reviews   ──►  Scrape/Ingest  ──┐                 ──►  Normalize   ──►  Unit Extraction  ──►
                    raw_reviews.jsonl  │                    reviews.jsonl      units.jsonl
  [Mouthshut CSV,                     │  Stage 1b (§12,
   optional §12]  ───────────────────┘   optional/no-op if
                                          CSV absent)
                                        raw_mouthshut.jsonl

     Stage 4             Stage 5              Stage 6              Stage 7
  ──► Embed      ──►  Similarity Graph ──► Community Detect ──► LLM Summarize ──►
     embeddings.npy    graph.gpickle       communities.json    themes.json

     Stage 8                    Stage 9                Stage 10
  ──► Category Graph +    ──►  Validation      ──►  UI / Notebook
      Insight Mapping          Report               (browse themes,
      insights.json            validation.json       answer 8 Qs)
```

Data flows one direction; each stage's output is a durable artifact in `data/`. Re-running a stage only requires its upstream artifact.

---

## 3. Pipeline Stages → Modules

| # | Stage | Module | Input | Output | Key libs |
|---|---|---|---|---|---|
| 1 | Scrape | `src/scrape.py` | Play Store (app `com.grofers.customerapp`) | `data/raw_reviews.jsonl` | `google-play-scraper` |
| 1b | Ingest Mouthshut (§12, optional, second source) | `src/scrape_mouthshut.py` | `data/Mouthshut_reviews.csv` (if present) | `data/raw_mouthshut.jsonl` | stdlib `csv` |
| 2 | Normalize | `src/normalize.py` | `raw_reviews.jsonl` (+ `raw_mouthshut.jsonl` if present) | `data/reviews.jsonl` | stdlib |
| 3 | Unit extraction | `src/units.py` | `reviews.jsonl` | `data/units.jsonl` | LLM (Groq) / rules |
| 4 | Embed | `src/embed.py` | `units.jsonl` | `data/embeddings.npy` + `data/unit_index.json` | `sentence-transformers` |
| 5 | Similarity graph | `src/graph.py` | embeddings | `data/graph.gpickle` | `faiss`/`sklearn`, `networkx` |
| 6 | Community detection | `src/cluster.py` | `graph.gpickle` | `data/communities.json` | `python-louvain` |
| 7 | LLM summarization | `src/summarize.py` | communities + units | `data/themes.json` | Groq |
| 8 | Insight mapping | `src/insights.py` | themes + units | `data/insights.json` | LLM + stdlib |
| 9 | Validation | `src/validate.py` | themes, communities, embeddings | `data/validation.json` | `sklearn`, stdlib |
| 10 | UI | `app.py` / `notebook.ipynb` | `insights.json`, `themes.json`, `validation.json` | browsable output | `streamlit` |

Shared code lives in `src/schema.py` (dataclasses + JSON I/O helpers) and `src/config.py` (paths, model names, tunables).

---

## 4. Module Responsibilities

### Stage 1 — `scrape.py`
- Collection target is a **rolling lookback window** (`scrape.lookback_months`, default 4) rather than a fixed review count — the goal is "all reviews from the last N months", not "N reviews".
- Pull per (sort × rating-band) bucket across ratings 1–5; `lang='en'`, `country='in'`. For the chronological `newest` sort, pagination stops exactly at the window boundary (exact/complete); non-chronological sorts (e.g. `relevance`) are best-effort only (edgecases.md S1-12).
- `scrape.max_per_bucket` is a safety cap per bucket in case a rating band has more reviews than expected within the window (edgecases.md S1-13).
- Handle pagination/continuation tokens; deduplicate by review id.
- Persist raw payloads verbatim to `raw_reviews.jsonl` (one JSON object per line) so re-normalization never requires re-scraping.
- Cache-friendly: skip re-fetch if raw file exists unless `--refresh` is passed; an interrupted run resumes using the originally-computed cutoff, not a freshly recomputed one (edgecases.md S1-14).

### Stage 1b — `scrape_mouthshut.py` (§12, optional, second source)
- Reads a pre-scraped CSV (`data/Mouthshut_reviews.csv`) rather than calling any network API - this stage's own job is only "raw CSV → raw JSONL, tagged with `source: mouthshut`", the same shape Stage 2 already expects from Stage 1.
- A no-op (logs and returns) if the CSV isn't present, so the pipeline behaves identically to before §12 for anyone who never adds that file.

### Stage 2 — `normalize.py`
- Map raw Play Store fields → canonical schema (see §5).
- Coerce dates to ISO 8601, ratings to int, capture `thumbs_up`, `app_version`, `developer_reply`, `lang`.
- Basic cleaning: strip control chars, drop empty/whitespace-only text, keep emojis (signal).
- **§12 addendum:** also normalizes `raw_mouthshut.jsonl` if present (title+body concatenation, absolute/relative date parsing, `source: "mouthshut"`) and merges both sources into one `reviews.jsonl`.

### Stage 3 — `units.py`
- Split multi-topic reviews into atomic complaint/insight units.
- Primary path: local LLM prompt that returns a list of atomic statements. Fallback: sentence/clause splitting rule-based, for zero-LLM runs.
- Each unit keeps a `review_id` back-reference for verbatim traceability.

### Stage 4 — `embed.py`
- Encode each unit with a local sentence-transformer (default `all-MiniLM-L6-v2`).
- Normalize vectors (cosine ready). Save as `embeddings.npy` aligned to `unit_index.json` (row → unit id).
- Batch + cache; deterministic (no network).

### Stage 5 — `graph.py`
- Build kNN over embeddings (cosine). Add `SIMILAR` edges above a similarity threshold or top-k per node.
- Node = unit; edge weight = similarity. Save as `networkx` graph.

### Stage 6 — `cluster.py`
- Run Louvain community detection on the weighted graph.
- Assign each unit a `community_id`; record community sizes; drop/flag micro-communities.

### Stage 7 — `summarize.py`
- For each community: select representative units (e.g., highest intra-community centrality / medoid), send to local LLM.
- Produce a one-sentence theme label + short description + 2–3 representative verbatims + member count + dominant sentiment.
- Output `themes.json`.

### Stage 8 — `insights.py`
- Map each theme → one or more of the 8 research questions (LLM-assisted classification + rules).
- Quantify per theme: frequency, rating-band spread, time-cohort spread, sentiment.
- Build category–category similarity edges (averaged member similarities) for a navigable map.
- Assemble `insights.json`: for each of the 8 questions, the supporting themes, counts, and verbatims.

### Stage 9 — `validate.py`
- **Cluster coherence:** intra- vs inter-cluster similarity, silhouette-style score.
- **Manual spot-check sample:** export a stratified sample for human labeling; compute agreement if labels provided.
- **Cross-segment triangulation:** verify each theme appears across rating bands, time periods, and review-length/recency cohorts.
- Output `validation.json` + a human-readable summary.

### Stage 10 — `app.py` / notebook
- Browse themes, drill into verbatims, and view each research question's evidence-backed answer.
- Streamlit app (shareable) or notebook fallback.

---

## 5. Canonical Schemas

**Review (`reviews.jsonl`):**

```json
{
  "id": "string",
  "source": "google_play | mouthshut",
  "text": "string",
  "rating": 1,
  "date": "2026-01-01T00:00:00Z",
  "url": null,
  "metadata": {
    "thumbs_up": 0,
    "app_version": "string",
    "developer_reply": null,
    "lang": "en"
  }
}
```

**Unit (`units.jsonl`):**

```json
{
  "unit_id": "string",
  "review_id": "string",
  "text": "atomic statement",
  "rating": 1,
  "date": "2026-01-01T00:00:00Z",
  "lang": "en",
  "source": "google_play | mouthshut"
}
```

**`themes.json` (top-level shape, revised during Stage 7 implementation):** a single object with two separate sections, not one flat list - see edgecases.md S6-03/S7-01 for why below-`min_community_size` communities are deliberately kept out of `themes` rather than forced into the `Theme` schema.

```json
{
  "themes": [
    {
      "theme_id": "string",
      "community_id": 0,
      "label": "one-sentence theme",
      "description": "short summary",
      "representative_quotes": ["...", "..."],
      "member_count": 0,
      "sentiment": "negative | neutral | positive",
      "questions": [2, 6]
    }
  ],
  "emerging_signals": [
    {
      "signal_id": "string",
      "community_id": 0,
      "label": "short excerpt-derived label",
      "description": "short templated summary",
      "representative_quotes": ["...", "..."],
      "support_count": 1,
      "confidence": "very_low | low",
      "avg_rating": 5.0,
      "lang_counts": {"en": 1}
    }
  ],
  "summary": {"num_themes": 0, "num_emerging_signals": 0, "schema_rejected": 0, "use_llm": false}
}
```

`themes` entries conform to the `Theme` dataclass (`src/schema.py`) and come only from communities at/above `clustering.min_community_size`. `emerging_signals` entries are plain dicts (not schema-validated `Theme`/`QuestionInsight` records) from every below-min-size community, each carrying its own `support_count`/`confidence` rather than being forced into a full theme's shape; Stage 8 (`insights.py`) may treat them as supplementary evidence only, not as first-class themes.

**`insights.json` (top-level shape, added during Stage 8 implementation):**

```json
{
  "questions": [
    {
      "question_id": 1,
      "query": "topic description embedded for matching, not the literal RQ text",
      "coverage": "sufficient | insufficient",
      "theme_ids": ["theme-0000", "..."],
      "theme_similarities": {"theme-0000": 0.44},
      "total_count": 0,
      "top_verbatims": ["...", "..."],
      "signal_ids": ["signal-0040", "..."],
      "signal_support_total": 0,
      "signal_confidence": "low | null"
    }
  ],
  "uncategorized": {"theme_ids": ["..."], "signal_ids": ["..."]},
  "top_themes": [{"theme_id": "...", "label": "...", "member_count": 0, "sentiment": "..."}],
  "theme_segment_stats": {
    "theme-0000": {
      "rating_distribution": {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0},
      "time_cohort_distribution": {"2026-03": 0, "2026-04": 0}
    }
  },
  "category_graph": [{"theme_a": "theme-0000", "theme_b": "theme-0003", "similarity": 0.75}],
  "summary": {"num_themes": 0, "num_emerging_signals": 0, "similarity_threshold": 0.3, "num_questions_sufficient": 0, "num_questions_insufficient": 0}
}
```

`questions[i]` conforms to (a superset of) the `QuestionInsight` dataclass - `theme_ids`/`total_count`/`top_verbatims`/`coverage` are schema-validated via `QuestionInsight(...)` before being merged with the extra non-schema fields (`query`, `theme_similarities`, `signal_*`) at write time, same pattern as Stage 7's `Theme` construction. `category_graph` is a theme-to-theme similarity graph over our *own discovered themes* (problemstatement.md §7's "category-level graph"), not an external Blinkit product-category taxonomy - built from centroid-embedding similarity, top-5 neighbors per theme.

---

## 6. Repository Layout

```
Discovery Engine/
├── problemstatement.md         # spec (source of truth)
├── context.md                  # persistent project context
├── architecture.md             # this file
├── README.md                   # setup + run steps (to be created)
├── requirements.txt            # pinned deps (to be created)
├── config.yaml                 # tunables (app id, models, thresholds)
├── src/
│   ├── config.py
│   ├── schema.py
│   ├── scrape.py               # Stage 1
│   ├── scrape_mouthshut.py     # Stage 1b (§12, optional, second source)
│   ├── normalize.py            # Stage 2
│   ├── units.py                # Stage 3
│   ├── embed.py                # Stage 4
│   ├── graph.py                # Stage 5
│   ├── cluster.py              # Stage 6
│   ├── summarize.py            # Stage 7
│   ├── insights.py             # Stage 8
│   ├── validate.py             # Stage 9
│   └── pipeline.py             # orchestrates 1→9
├── app.py                      # Stage 10 (Streamlit)
├── notebook.ipynb              # Stage 10 (fallback)
└── data/                       # all artifacts (gitignored)
    ├── raw_reviews.jsonl
    ├── Mouthshut_reviews.csv   # §12, optional, dropped in by hand/separate process
    ├── raw_mouthshut.jsonl     # §12, optional
    ├── reviews.jsonl
    ├── units.jsonl
    ├── embeddings.npy
    ├── unit_index.json
    ├── graph.gpickle
    ├── communities.json
    ├── themes.json
    ├── insights.json
    └── validation.json
```

---

## 7. Orchestration & Reproducibility

- `src/pipeline.py` runs stages 1→9 in order; each stage is skippable if its artifact exists (`--force` to rebuild).
- CLI per stage for isolated debugging (e.g., `python -m src.embed`).
- `config.yaml` centralizes: app id, scrape lookback window (months) + per-bucket safety cap, embedding model, kNN `k`/threshold, Louvain resolution, LLM model name, random seed.
- Determinism: fixed seeds, pinned `requirements.txt`, cached embeddings; LLM steps are the only non-deterministic parts (mitigated with low temperature + cached outputs).

---

## 8. Tech Stack (Zero-Cost)

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.10+ | |
| Scraping | `google-play-scraper` | free, no key |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | local, CPU-friendly |
| kNN | `faiss-cpu` or `scikit-learn` | fallback to sklearn if faiss unavailable |
| Graph | `networkx` | |
| Community detection | `python-louvain` | |
| LLM (units, summaries, mapping) | Groq local model (e.g., `llama3`/`qwen`) | zero-cost; rule-based fallback for units |
| Validation | `scikit-learn`, `numpy`, `pandas` | |
| UI | `streamlit` | shareable link; notebook fallback |

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Play Store rate-limits / partial pulls | Persist raw incrementally; resumable scrape; cache raw file. |
| Local LLM slow/unavailable | Rule-based fallback for unit extraction; batch + cache summaries; keep summarization on representative units only. |
| Hinglish/code-mixed text hurts embeddings | Multilingual embedding option; keep language tag; report per-lang coverage. |
| Louvain instability / too many micro-clusters | Tune resolution; merge/prune small communities; report cluster coherence in validation. |
| Single-source bias | Cross-segment triangulation (rating bands, time cohorts) as the validation substitute. |

---

## 10. Build Sequence (Remaining)

1. ✅ `architecture.md` (this doc).
2. `requirements.txt` + `src/config.py` + `src/schema.py` (foundations).
3. `src/scrape.py` → `src/normalize.py` (Stages 1–2).
4. `src/embed.py` → `src/graph.py` → `src/cluster.py` (Stages 4–6; Stage 3 units in parallel).
5. `src/summarize.py` → `src/insights.py` (Stages 7–8).
6. `src/validate.py` (Stage 9).
7. `app.py` / notebook (Stage 10) + `README.md`.

---

## 11. Stage 11 (Addendum, Post-Phase-7) — Deep Pattern Synthesis via Groq

**Not part of the original 10-stage plan (§2/§3) or the project's core Definition of Done** -
added afterward as an explicit, user-directed extension once Phases 0-7 were already
complete and verified. Documented here for the same traceability reason every other module
is documented: `src/llm_synthesis.py` reads `themes.json`/`insights.json`/`communities.json`/
`units.jsonl` and writes `data/llm_insights.json`, consumed by `app.py`'s Research Questions
page as an additional, clearly-labeled section - it does not feed back into or alter any
S1-S9 artifact.

**Motivating gap:** Stage 7's TF-IDF labels and Stage 8's embedding-similarity mapping can
only ever surface literal recurring vocabulary in review text - by construction, neither
method can name an abstract behavioral driver that is never phrased that way in any single
review but is implied across many differently-worded ones (e.g. "choice overload",
"cognitive load when facing unfamiliar categories"). This was verified empirically, not
assumed: Q2 ("what prevents users from exploring new categories?") had **zero** supporting
themes in the real `insights.json` run, and all 16 "uncategorized" themes were inspected and
confirmed to be generic praise/rating-noise/competitor-mention content, not a hidden
exploration-barrier signal the mapping missed by threshold alone.

**Design, deviating deliberately from §1's "fully local" principle:**

- Uses the Groq API (OpenAI-compatible, free tier, no credit card) instead of local Groq -
  zero-cost is preserved, "fully local"/offline is not. This is why it is a separate,
  opt-in script, never folded into `python -m src.pipeline`.
- Sends a bounded, prioritized, deterministically-seeded *sample* of raw review excerpts
  (not the full corpus - free-tier rate limits make that impractical), prioritizing exactly
  the units Stage 8 already flagged `"uncategorized"`.
- One batched LLM call is asked to infer, per excerpt, an indirect/implicit behavioral
  pattern against all 8 research questions at once - explicitly instructed (with a worked
  example) not to just restate the question categories verbatim, after empirically observing
  a naive prompt caused exactly that lazy-copy failure mode in initial testing.
- Raw per-batch pattern phrases are aggregated into canonical patterns via the *same local*
  embedding model used everywhere else in the pipeline (no extra API calls, deterministic
  clustering step) - the LLM infers, the deterministic local pipeline aggregates.
- Every pattern remains traceable to real verbatim quotes/unit ids - only the
  *interpretation* is model-generated, never the evidence itself.

See `README.md` §12 for setup, the measured model comparison (`llama-3.1-8b-instant` vs.
`openai/gpt-oss-120b`) that justified the default model choice, and rate-limit economics.

---

## 12. Stage 1b (Addendum, Post-Phase-8) — Second Data Source: Mouthshut

Unlike §11's Stage 11 (a separate opt-in script), this addendum changes the **core**
`python -m src.pipeline` (§1-§9): a second source, ingested at Stage 1b and merged into the
same `reviews.jsonl`/`units.jsonl` every downstream stage already reads. It directly amends
§1's "single source" principle - flagged explicitly to the user before any code was written
(the source, Mouthshut.com, is literally the "complaint boards" category problemstatement.md
§5 names as out-of-scope), and the user chose to amend the constraint rather than keep this
data separate or skip it.

**What changed, concretely:**

- `src/scrape_mouthshut.py` (Stage 1b): reads `data/Mouthshut_reviews.csv` (a pre-scraped
  export, not fetched by this stage - no network call, unlike Stage 1) and writes
  `data/raw_mouthshut.jsonl`, one raw record per review tagged `source: "mouthshut"`. A no-op
  if the CSV is absent, so nothing changes for a fresh clone without that file.
- `src/schema.py`: `Review.source`/`Unit.source` now validate against
  `VALID_SOURCES = {"google_play", "mouthshut"}` instead of a hard-coded single value; `Unit`
  gained a `source` field (default `"google_play"`, for backward-compat with pre-addendum
  `units.jsonl`) so provenance survives Stage 3 onward.
- `src/normalize.py` (Stage 2): normalizes both raw sources and merges them into one
  `reviews.jsonl`. Mouthshut-specific handling: `title`+`body` concatenated into one `text`
  (Play Store's `content` is already single-field); dates parsed in both shapes actually
  present in the file - absolute (`"Jun 20, 2026 05:15 PM"`) and relative (`"N days ago"`,
  resolved against the CSV's own file-mtime as a documented best-effort reference anchor);
  no `thumbs_up`/`app_version`/`developer_reply` equivalent exists in this export, so those
  stay at schema defaults rather than being invented.
- Stages 4-7 (`embed.py`/`graph.py`/`cluster.py`/`summarize.py`) are **unchanged** - they
  operate on `Unit`/embeddings generically and never branched on source, so the merged
  corpus flows through them with zero code changes.
- `src/insights.py`/`src/validate.py` (Stages 8-9): `_theme_segment_stats`/`_triangulation`
  now also report a per-theme `source_distribution` (and corpus-level
  `corpus_sources_observed`) - purely informational, deliberately **excluded** from the
  cross-segment `stability` verdict for the same reason rating bands are (S9-05): Mouthshut
  is ~4.4% of the merged unit corpus, so near-universal `google_play` dominance per theme is
  an expected size-imbalance artifact, not evidence of instability.
- `app.py` (Stage 10): Overview gets a third "Source split" chart; the Theme Explorer's
  per-theme detail view gets a 4th "Source distribution" chart alongside rating/time/length.

**Verified on the real merged corpus** (`python -m src.pipeline`, re-run from `normalize`
onward - `raw_reviews.jsonl`/`raw_mouthshut.jsonl` reused, no re-scrape): 157,219 reviews
(156,219 Google Play + 1,000 Mouthshut, all 1,000 Mouthshut rows normalized with zero
rejections) → 93,393 units (89,295 + 4,098) → 5 themes + 851 emerging signals → same 4/8
questions "sufficient" → validation modularity 0.8366, 4/5 cross-segment stable.
Mouthshut units landed in all 5 themes, confirming they blend topically with the existing
corpus rather than forming an isolated island.

`problemstatement.md` itself is left unmodified (treated as the immutable original spec,
same precedent as §11's Groq addendum) - this file, `context.md`, `Implementation-plan.md`,
and `README.md` carry the addendum instead.
