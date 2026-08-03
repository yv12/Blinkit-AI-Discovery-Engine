# Project Context: AI-Powered Discovery Engine

> **Purpose:** Persistent, agent-facing context for this project. Read this first at the start of any session for fast grounding. It distills `problemstatement.md` into the durable facts, decisions, constraints, and current state. When decisions change or milestones complete, update this file. The full spec in `problemstatement.md` remains the source of truth for requirements.

---

## 1. One-Line Summary

Build a **zero-cost, local AI pipeline** that scrapes **Google Play Store reviews of Blinkit**, clusters them into themes via an embedding + similarity-graph + community-detection approach, and produces **evidence-backed answers to 8 research questions** about why users don't explore new product categories.

---

## 2. Mission & Success Definition

- **Business goal (simulated):** As a Blinkit Growth PM, increase the % of Monthly Active Customers who buy from **at least one new category per month**.
- **What this codebase does:** A discovery engine that gathers and analyzes user feedback **at scale** and surfaces insights on category-exploration behavior across the **whole funnel** (discovery → engagement → add-to-cart → conversion).
- **Done when:**
  - Pipeline runs end-to-end (scrape → browsable insight output) on a free/local stack.
  - Each of the 8 research questions has an evidence-backed answer (theme names, counts, representative verbatims).
  - Theme identification + validation methodology is documented and demonstrable.
  - Output is presentable via a link (hosted workflow, notebook, or lightweight app).
  - Reproducible from a fresh environment with documented steps.

---

## 3. The 8 Research Questions (Decision Criteria)

Every analysis-method choice is judged against whether it helps answer these:

1. Why do users repeatedly buy from the same categories?
2. What prevents users from exploring new categories?
3. How do users discover products today?
4. What role do habits play in shopping behavior?
5. What information do users need before trying a new category?
6. What frustrations emerge repeatedly?
7. Which user segments are more likely to experiment?
8. What unmet needs emerge consistently across discussions?

**Note:** Operational/pricing complaints (stockouts, price, delivery time) must be **captured and categorized**, not filtered out — they are part of the funnel diagnosis.

---

## 4. Hard Constraints (Non-Negotiable)

| Constraint | Rule |
|---|---|
| **Zero cost** | No paid APIs or paid models at any stage. Local embeddings, local/free-tier LLMs, free scraping libs only. |
| **Code-only collection** | All data gathered programmatically. No manual copy-paste. |
| **Single data source** | **Google Play Store reviews only** (originally). Reddit, Apple App Store, YouTube, Quora, forums, X/Twitter, Instagram remain **out of scope** — do not build connectors for them. **Amended in §11 Phase 9**, with the user, to also accept a Mouthshut review-forum CSV as a second, explicitly-tagged source (`schema.py: VALID_SOURCES`) — a deliberate, documented exception for that one additional source, not a general opening of scope. |
| **Target app** | **Blinkit.** Competitor mentions (Zepto/Instamart) are useful for contrast but Blinkit is the subject. |

---

## 5. Data Source Details

- **Library:** `google-play-scraper` (Python, free, no API key).
- **App ID:** `com.grofers.customerapp` (Blinkit — verify at build time).
- **Corpus goal:** All reviews across all ratings (1–5 stars) within a **rolling lookback window** (default: last 4 calendar months, `config.yaml` `scrape.lookback_months`), collected via the chronological `newest` sort per rating band for exact/complete window coverage; `lang='en'`, `country='in'` (English + Hindi/Hinglish where available). A per-bucket safety cap (`scrape.max_per_bucket`) guards against runaway pulls on high-volume rating bands.
- **Per-review metadata to capture:** rating, date, thumbs-up count, app version, developer reply (if present).
- **Validation twist:** Single source means "cross-source triangulation" becomes **cross-segment triangulation** — verify themes hold across rating bands, time periods, and review length/recency cohorts.

---

## 6. Analysis Approach (Graph-Based Theme Discovery)

Preserve this shape; refine details as needed:

1. **Ingest & normalize** into common schema (see §7).
2. **Extract complaint/insight units** — split multi-topic reviews into atomic statements.
3. **Embed** each unit with a free/local embedding model.
4. **Similarity graph** — each unit is a node; kNN over embeddings creates `SIMILAR` edges.
5. **Community detection** (Louvain) → theme communities.
6. **LLM summarization** (free/local) → one-sentence theme node per community, with representative quotes + counts.
7. **Category-level graph** — category–category edges from averaged member similarities → navigable insight map.
8. **Insight layer** — map themes to the 8 questions; quantify frequency, source spread, sentiment.
9. **Validation layer** — cluster coherence, manual spot-check sample, cross-segment triangulation report.

---

## 7. Canonical Data Schema

Normalize every scraped item into:

```json
{
  "id": "string",
  "source": "google_play | mouthshut",
  "text": "string",
  "rating": "int (1-5, optional)",
  "date": "ISO 8601 datetime",
  "url": "string (optional)",
  "metadata": {
    "thumbs_up": "int",
    "app_version": "string",
    "developer_reply": "string | null",
    "lang": "en | hi | ..."
  }
}
```

---

## 8. Suggested Free/Local Stack

> These are defaults consistent with the zero-cost constraint — confirm/adjust at build time.

- **Language:** Python 3.
- **Scraping:** `google-play-scraper`.
- **Embeddings:** local `sentence-transformers` (e.g., `all-MiniLM-L6-v2`).
- **Graph + clustering:** `networkx` + `python-louvain` (community detection); `scikit-learn`/`faiss` for kNN.
- **LLM summarization:** local/free-tier model (e.g., Groq-hosted model) — must remain zero-cost.
- **UI/output:** notebook or lightweight app (e.g., Streamlit) for browsing themes and question answers.

---

## 9. Deliverables (Context, Not All Code Targets)

- Shareable/testable link to the review-analysis workflow.
- A 1-slide workflow explainer inside a 10-slide PDF deck.
- Findings later validated via 5–6 user interviews (**outside this codebase's scope**).

---

## 10. Build Order (What to Generate Next)

Per the spec, generate in this order:

1. `architecture.md` — project plan / architecture doc mapping pipeline stages to modules.
2. Play Store scraper module with the normalized output schema (§7).
3. Embedding + graph + clustering pipeline.
4. LLM summarization + insight-mapping layer.
5. Validation report generator.
6. Minimal UI or notebook to browse themes and answer the 8 questions.

---

## 11. Current State

- **Repo contents:** `Docs/` (`problemstatement.md`, `context.md`, `architecture.md`, `Implementation-plan.md`, `edgecases.md`); project root: `requirements.txt`, `config.yaml`, `.gitignore`, `src/` (`__init__.py`, `config.py`, `schema.py`, `pipeline.py`, `scrape.py`, `normalize.py`, `units.py`, `embed.py`, `graph.py`, `cluster.py`, `summarize.py`, `insights.py`, `validate.py`).
- **Progress:** Planning docs complete. **Phases 0-5 are all done and verified on the real corpus** - the full pipeline (`scrape → normalize → units → embed → graph → cluster → summarize → insights → validate`) runs end-to-end cleanly via `python -m src.pipeline`. **Phase 6 (UI/Presentation) is next**, the last phase:
  - Phase 0: `config.py` loads/validates `config.yaml` (edgecases.md X-08); `schema.py` provides validated Review/Unit/Theme/QuestionInsight dataclasses with atomic JSONL/JSON I/O and corrupt-artifact detection (X-04/X-05/X-06); `pipeline.py` orchestrates stages S1–S9 with skip/force/only controls.
  - Phase 1: `scrape.py` collects a **rolling lookback window** (`scrape.lookback_months`, default 4) rather than a fixed count — pulls per (sort × rating-band) bucket with retry/backoff, pagination-loop detection, chronological early-stop exactly at the window boundary (verified with zero out-of-window leakage), a per-bucket safety cap (`max_per_bucket`), and crash-resumable checkpointing that preserves the original cutoff across restarts (S1-01/02/03/04/06/09/10/12/13/14). Verified live against the real Blinkit app (`com.grofers.customerapp`, title "Blinkit: Groceries & more"). `normalize.py` cleans/validates text, normalizes dates to UTC ISO 8601, coerces ratings, tags language (Devanagari heuristic confirmed working on real Hindi review text), and flags near-duplicates without dropping them (S2-01/02/03/05/06/07/08). Both stages tested end-to-end via `python -m src.pipeline` and confirmed to skip cleanly on re-run.
- **Real corpus collected** (`data/raw_reviews.jsonl` 86 MB, `data/reviews.jsonl` 62 MB, both gitignored): **156,219 unique reviews**, window 2026-03-22 to 2026-07-22. Rating split: 1★ 27,181 / 2★ 4,191 / 3★ 7,096 / 4★ 17,751 / 5★ 100,000. 1★–4★ fully cover the 4-month window; 5★ hit the `max_per_bucket=100000` safety cap and only reaches back to 2026-04-03 (~90% of the window covered) — accepted as-is per user decision, since the missing slice is the oldest, lowest-signal 5★ praise. Language split after normalization: 155,647 `en` / 572 `hi` (Devanagari-script heuristic; Hinglish in Latin script is not distinguished and counts as `en` — a known simplification). 0 reviews dropped during normalization; 3,906 near-duplicate text groups flagged (not dropped).
- **Unit extraction done** (`data/units.jsonl`, gitignored): two filtering/method decisions were made explicitly with the user rather than left as defaults —
  1. **Low-signal filter (edgecases.md S3-09, revised):** reviews with <4 words (e.g. "good", "nice", "super app") are **fully excluded** — no unit produced, not counted in any downstream artifact or stat. This is a **pure length cutoff** (not a generic-phrase/stoplist filter), a deliberate simplicity-over-precision trade-off that also drops some genuinely short complaints (e.g. "no delivery"). Real impact: 96,688 / 156,219 reviews (61.9%) excluded.
  2. **Splitting method (edgecases.md S3-01, revised):** rule-based sentence/conjunction splitting is the **primary/only** method (`units.use_llm: false`) — a per-review local LLM call is impractical at this corpus scale; the LLM is reserved for Stage 7 (one call per cluster, not per review).
  - Result: **89,295 units** from 59,531 contributing reviews (avg 1.50 units/review). Spot-checked and confirmed the splitter correctly isolates specific complaints and separates distinct topics within a review.
- **Embedding done** (`data/embeddings.npy` 131 MB, `data/unit_index.json`, both gitignored): all 89,295 units encoded with local `sentence-transformers/all-MiniLM-L6-v2` (384-dim), L2-normalized at encode time (verified row norm ≈ 1.0, so cosine similarity == dot product for Stage 5), batch size 128 (`models.embed_batch_size`), ~7 minutes / ~212 units/sec on CPU. Row count strictly asserted == unit count == `unit_index.json`'s `unit_ids` length (S4-03 alignment contract) before saving; both artifacts written atomically. Re-run confirmed idempotent.
  - **Real environment issue hit and fixed (edgecases.md R-07):** on this Windows machine, `torch`'s native DLLs (`c10.dll`) failed to initialize — `OSError: [WinError 1114]` — whenever `sentence-transformers`/`transformers` imported `torch` transitively (a plain top-level `import torch` on its own worked fine). Root-caused to a combination of (a) `torch` and `numpy`/`transformers` having landed in different site-packages roots (system-wide vs. per-user) from earlier ad-hoc installs, and (b) `transformers`' own import chain triggering the conflict before any of our code touched `torch` directly. Fixed by reinstalling `torch` into the same site-packages root as the rest (`pip install --user torch --index-url https://download.pytorch.org/whl/cpu`) and having `embed.py` explicitly `import torch` before `import sentence_transformers`. `requirements.txt` now pins `torch==2.13.0` via `--extra-index-url https://download.pytorch.org/whl/cpu` and documents this gotcha inline.
- **Similarity graph done** (`data/graph.gpickle`, gitignored): exact (brute-force) cosine kNN built over all 89,295 unit embeddings via scikit-learn's `NearestNeighbors` (`faiss` genuinely absent on this machine, so the S5-05 fallback path ran for real, not just in theory), `graph.knn_k=15`, `graph.similarity_threshold=0.5`, "any-kNN" union construction (an edge exists if either endpoint has the other in its top-15, not requiring mutual membership). Result: **89,295 nodes, 1,054,505 edges**; degree min=0 / median=20 / mean=23.6 / max=169; **794 isolated singleton nodes (0.9%)**, kept rather than dropped (S5-04) — Stage 6 will give each its own micro-community. Edge weights verified clamped into `[threshold, 1.0]` (float-drift guard, S5-06) — a nonzero number of edges landed at exactly weight 1.0, i.e. real duplicate/near-duplicate unit text in the corpus. kNN search over the real corpus took ~5 minutes on CPU (`NearestNeighbors(algorithm="brute", metric="cosine")`, benchmarked to scale ~O(n²): 5k units in <1s, 20k in ~16s, full 89k in ~305s).
  - **Real environment issue hit and fixed (edgecases.md R-08):** `networkx==3.3` (pinned) has no `write_gpickle`/`read_gpickle` — that API was removed in networkx 3.0. `graph.py` instead uses plain `pickle.dump`/`pickle.load` on the `nx.Graph` object (networkx's own documented replacement); the `.gpickle` filename is kept as a naming convention only, not a format guarantee.
- **Community detection done** (`data/communities.json` 4.7 MB, gitignored): Louvain (`clustering.louvain_resolution=1.0`, `random_state=config.seed`) over the 89,295-node graph produced **859 communities** across 825 connected components (S6-04, informational). Verified an exact partition — every unit id appears in exactly one community, no duplicates or gaps. Distribution is long-tailed by design: 819 communities are below `clustering.min_community_size=3` and flagged `"below_min_size": true` rather than dropped (S6-03 revision — Stage 6 flags, Stage 7 decides whether to skip/merge them); 794 of those are singletons, matching Stage 5's isolated-node count exactly (a good cross-stage consistency check, S6-05). The remaining **5 communities hold 88,451 units** (sizes ~40 to 8,609), with meaningfully different average ratings across the top ones (1.5-4.6 range) — i.e. distinct, non-trivial themes, and a manageable count for Stage 7's per-cluster LLM calls. No giant-community warning (largest is 9.6% of the corpus, well under the 50% advisory threshold, S6-02).
- **Summarization done** (`data/themes.json` 366 KB, gitignored): two decisions made explicitly with the user before implementing —
  1. **Long-tail handling (edgecases.md S6-03/S6-05, revised):** the 819 below-`min_community_size` communities are *not* run through full per-community theme summarization (representative selection, TF-IDF, optional LLM call) - that would be noise, not signal, at that granularity. Instead they're processed in **one dedicated batched pass** and surfaced as **`emerging_signals`** - plain dicts (deliberately *not* `Theme` records) tagged with `support_count` (= community size) and a `confidence` tier (`"very_low"` for singletons, `"low"` for pairs). They live in their own `themes.json` key, separate from `themes`; Stage 8 may treat them as supplementary evidence only.
  2. **LLM usage (edgecases.md S7-01, revised):** the local TF-IDF + rating-derived extractive method is the **primary** path (`summarize.use_llm: false` by default), not a fallback-of-last-resort - confirmed live that Groq is unreachable on this machine, so this is also what actually ran. Sentiment always comes from the rating distribution, never from an LLM (S7-06). Setting `summarize.use_llm: true` (with Groq running locally) additionally tries an LLM label/description per theme (and a batched LLM pass for the long tail), degrading silently back to the extractive result on any failure.
  - Representative units per theme are selected by cosine similarity to the community's centroid embedding (cheap O(n) proxy for medoid, avoids O(n²) on communities up to 8,609 members); quotes are therefore always real member text, never LLM-generated (S7-05 satisfied by construction).
  - Result: **5 themes** (from the 5 qualifying communities, `member_count` 40-8,609) + **819 emerging signals**. Theme sentiment split 15 positive / 14 negative / 11 neutral. Spot-checked and labels are topically coherent even without an LLM - e.g. `theme-0002` "customer / service / support / care" (avg rating 1.5, negative) with verbatim quotes like "Very poor service by the customer support team"; `theme-0004` "app / good / best / nice" (positive). `Theme.questions` is left empty (`[]`) - mapping to the 8 research questions is Stage 8's job.
- **Insight mapping done** (`data/insights.json` 57 KB, gitignored): two decisions made explicitly with the user before implementing —
  1. **Theme creation stays fully bottom-up.** The 8 research questions are never an input to clustering (Stage 6) or summarization (Stage 7); this stage is a strictly post-hoc tagging pass over `themes.json`. Mapping method is **embedding similarity**, not keywords or an LLM call: each theme's `label + description` is embedded with the same local model as Stage 4 and compared via cosine similarity against a short topic description per question (`insights.question_queries`, config.yaml) - keeps the zero-cost/no-LLM property that was already the actual runtime path in Stages 3 and 7.
  2. **The 819 `emerging_signals` are mapped the same way**, run against Stage 7's already-batched long-tail summaries (not re-run against raw units), but tracked in a *separate* `signal_ids`/`signal_support_total` key per question, tagged `signal_confidence: "low"` - they never flip a question's `coverage` to `"sufficient"` on their own, only real themes count as strong evidence.
  - **Real bug found and fixed empirically:** embedding similarity alone matches on shared topic nouns regardless of polarity - a clearly *positive* theme ("app / delivery / fast / best") scored 0.575 against the "frustrations" question, higher than most genuinely negative themes, because MiniLM embeddings of short phrases are dominated by noun overlap ("delivery", "app") not sentiment framing. Fixed with `insights.question_required_sentiment` - Q6 is gated on the theme's/signal's own rating-derived sentiment (never LLM-derived, same S7-06 principle); this dropped Q6 from 34 to 11 themes, all genuinely negative.
  - **Final real result:** 4/8 questions `"sufficient"` (Q1 repeat-category habits, Q3 discovery, Q4 habitual behavior, Q6 frustrations), 4 `"insufficient"` (Q2 exploration barriers, Q5 pre-trial info needs, Q7 experimenter segments, Q8 unmet needs) - reported honestly rather than force-matched; this Blinkit corpus genuinely skews toward operational complaints/praise over discovery-behavior commentary, which is itself a real finding. 2/5 themes and 780/819 signals are `"uncategorized"` (kept, not dropped). Q1 has known residual imprecision (matches on shared word-roots like "order"/"items" regardless of habit-vs-complaint framing) - documented as a v1 limitation, not further chased.
  - Also implemented `theme_segment_stats` (per-theme rating-band + time-cohort distribution, sourced from `communities.json` + `units.jsonl`) and `category_graph` (5 themes × top-5 similarity edges via centroid embedding) - both were in architecture.md's/problemstatement.md's original Stage 8 spec but missing from the first implementation pass; `category_graph` is a graph over our *own discovered themes* (problemstatement.md §7), not an external Blinkit product-category taxonomy, since Google Play reviews carry no such label.
- **Validation done** (`data/validation.json` 36 KB + `data/validation_summary.md`, both gitignored except the doc references here; `data/spot_check_sample.json` 74 KB also gitignored): three independent checks, only over the 5 qualifying `themes` (the 819 `emerging_signals` stay out of scope, consistent with their supplementary-evidence status since Stage 7/8) —
  1. **Coherence:** graph **modularity of the actual Louvain partition = 0.8394** (very high - strong community structure) as the primary, methodologically-correct metric for a graph-clustering method; a per-theme centroid-based "silhouette-style" score (embedding space, `[-1,1]`, not literal per-point silhouette which is infeasible at up to 8,609 members/theme) came out with a near-zero mean (**-0.001**). This pairing is itself an honest, informative finding, not a contradiction: the graph resolves fine sub-topic distinctions via precise unit-to-unit kNN edges (hence very high modularity), while several distinct themes share enough surface vocabulary (e.g. `theme-0002` "customer/service/support" vs `theme-0003` "delivery/service/location") that their coarse centroid vectors sit close together. The smallest themes (3-4 members) show the highest silhouette scores - a known small-N artifact, documented as such rather than read as "better clustering."
  2. **Cross-segment triangulation:** recomputed independently from `communities.json`+`units.jsonl` (not read back from `insights.json`, for a genuinely separate check). **4/5 themes are "cross_segment" stable**, 1/5 "segment_specific" (mostly concentrated in one review-length bucket - e.g. short one-line praise themes, which is expected given the content, not necessarily a flaw). Rating-band concentration is deliberately excluded from the stability verdict (a sentiment-driven theme skewing one rating band is a healthy signal, not instability) but still reported per-theme for context.
  3. **Spot-check sample:** a stratified random sample (5 units/theme, 192 rows) written once to `data/spot_check_sample.json` with `human_agrees: null` per row, for manual review. Verified live (hand-edited one row, re-ran `--refresh`) that the file is **never auto-regenerated once it exists**, so manual labels always survive re-runs; `agreement_rate` computes correctly over whatever fraction is actually labeled. Delivered clean/unlabeled, ready for real review.
  - `data/validation_summary.md` (human-readable) lists every theme sorted worst-to-best by coherence and every segment-specific theme with its reason.
- **UI done (Phase 6):** `app.py` (Streamlit, 5 pages: Overview, Research Questions, Theme Explorer, Category Graph, Validation & Methodology) + `notebook.ipynb` (pandas/stdlib fallback, no Streamlit) both read the real Phase 4/5 artifacts (5 themes, 819 emerging signals, 8 questions, 5 validated themes) live - no hardcoded numbers. Verified with Streamlit's headless `AppTest` harness across all pages/filters/drill-downs and a real `streamlit run app.py` smoke test; notebook executed end-to-end via `nbclient` with zero error cells.
- **Reproducibility & docs done (Phase 7, final phase):** `src/pipeline.py` (already built incrementally since Phase 0) re-verified live - orchestrates all 9 batch stages in order, skip-if-artifact-exists / `--force` / `--only <stage>` all confirmed working against the real `data/`. `README.md` written: setup (incl. the real Windows torch-DLL gotcha), optional Groq install steps (pipeline is LLM-free by default and that's the verified path), config reference, run steps, expected artifact shapes/sizes, UI/notebook viewing instructions, determinism section, troubleshooting table, the 8 questions. **Real gap found and fixed while documenting LLM non-determinism mitigations:** `summarize.py`'s Groq call never actually set `temperature`/`seed` despite architecture.md §7 always specifying this mitigation - fixed by adding `options: {temperature: 0, seed: config.seed}` to both Groq call sites (per-theme label + batched long-tail label), only affecting the optional `summarize.use_llm: true` path.
- **Core project complete.** All 8 phases (0-7) done and verified against the real, full-scale Blinkit corpus (156,219 reviews → 89,295 units → 5 themes + 819 emerging signals → 8 research questions → validation report → browsable UI). See `Docs/Implementation-plan.md` for the full phase-by-phase verification log.
- **Phase 8 (addendum, post-Phase-7, user-directed, not part of the core Definition of Done):** built `src/llm_synthesis.py` (Stage 11) to close a gap surfaced by direct user question - the deterministic pipeline (TF-IDF labels + embedding-similarity question-mapping) can only find literal recurring vocabulary, never an indirectly-implied abstract behavioral driver (e.g. "choice overload"); verified concretely that Q2 had zero supporting themes and all 16 "uncategorized" themes were generic praise, not a missed signal. Uses the user-provided Groq API key (free tier, `.env`-gitignored) - a deliberate, explicit deviation from "fully local" (zero-cost is preserved, offline is not), kept fully separate from `python -m src.pipeline`. Real findings while building it: a naive prompt made the model lazily copy the 8 question descriptions verbatim as "patterns" - fixed with explicit anti-copying instructions + a worked example, verified fixed on a live batch; three models measured head-to-head on identical real data (`llama-3.1-8b-instant` chosen as default for its ~500K-token/day budget vs. `openai/gpt-oss-120b`'s deeper-but-hidden-reasoning-token-heavy ~130K effective daily budget); real rate-limit ceiling measured via live `usage.total_tokens` (~4-5 calls/min sustainable, not the nominal 30 RPM). Output (`data/llm_insights.json`) surfaces in `app.py`'s Research Questions page as a clearly-labeled, separate "LLM-Inferred Patterns" section. See `Docs/Implementation-plan.md` Phase 8 and `README.md` §12 for full details.

- **Phase 9 (addendum, post-Phase-8, user-directed, not part of the core Definition of Done):** integrated a second data source, a pre-scraped **Mouthshut** review-forum CSV (`data/Mouthshut_reviews.csv`, ~1,000 rows), into the core pipeline (not a side/optional stage like Phase 8's Groq synthesis). This directly relaxes §4's original "single data source: Google Play only" constraint — flagged explicitly to the user before doing any work (Mouthshut is literally the "complaint boards" category problemstatement.md §5 lists as out-of-scope), and the user chose to amend the constraint and integrate it as a real second source rather than keep it separate or skip it.
  - **Design:** a new Stage 1b (`src/scrape_mouthshut.py`) ingests the CSV into `data/raw_mouthshut.jsonl` (raw, source-tagged, one record per review) — a no-op if the CSV isn't present, so the pipeline behaves identically to before for anyone who never adds that file. `normalize.py` then normalizes both raw sources into one unified `reviews.jsonl`, each `Review.source` tagged `"google_play"` or `"mouthshut"` (`schema.py: VALID_SOURCES = {"google_play", "mouthshut"}`, relaxed from the old hard-coded single-value check). `Unit.source` (new field, default `"google_play"` for pre-existing artifacts) propagates this through Stage 3 so provenance survives all the way to validation - added specifically so a theme's source mix is auditable, not lost after normalization.
  - **Mouthshut-specific normalization decisions:** (1) `title` + `body` are concatenated into one `text` field (Mouthshut splits them; Play Store's `content` is already a single field) so Stage 3's unit splitter sees the whole complaint. (2) Dates come in two real shapes in the actual file - absolute (`"Jun 20, 2026 05:15 PM"`, the vast majority) and relative (`"N days ago"`, only the most recent ~20 rows, max 30 days) - the relative ones are resolved against the CSV file's own last-modified time as a documented best-effort reference anchor, cross-checked against the data itself (the newest absolute date is 2026-06-20 and the oldest relative phrasing is "30 days ago" from a same-day ingest reference of 2026-07-23 → resolves to ~2026-06-23, consistent). (3) No `thumbs_up`/`app_version`/`developer_reply` equivalent exists in this export - left at their schema defaults (0/None/None) rather than invented. Verified live: all 1,000 rows normalized with zero rejections/unparseable dates.
  - **Cross-segment triangulation extended, not replaced:** `insights.py`'s `_theme_segment_stats` and `validate.py`'s `_triangulation` now also compute a `source_distribution` per theme (and a corpus-level `corpus_sources_observed`) - reported for every theme, exactly like rating bands, but **deliberately excluded from the `stability`/dominant-share verdict** for the same reason rating bands are (S9-05): Mouthshut is ~4.4% of the merged unit corpus (4,098 of 93,393), so a theme reading as ~95%+ google_play is an expected size-imbalance artifact, not evidence of instability - folding it into the stability judgment would flag almost every theme as trivially "segment_specific" on that axis alone, which would be misleading, not informative.
  - **Re-ran the full pipeline end-to-end** (`python -m src.pipeline`, `reviews.jsonl` onward regenerated; `raw_reviews.jsonl`/`raw_mouthshut.jsonl` reused, no re-scrape) against the merged corpus: **157,219 reviews** (156,219 Google Play + 1,000 Mouthshut) → **93,393 units** (89,295 + 4,098) → **5 themes** + 851 emerging signals (up slightly from 5/819 pre-merge - the extra Mouthshut volume shifted a few communities across the `min_community_size` boundary) → same 4/8 questions `"sufficient"` (Q1/Q3/Q4/Q6) → validation modularity 0.8366, 4/5 cross-segment stable. Confirmed live that Mouthshut units landed in all 5 themes (not siloed into their own cluster), i.e. they're topically blending with the existing Google Play corpus rather than forming an isolated island.
  - `app.py` updated to surface the new dimension without overclaiming: Overview gets a third "Source split" chart (with a caption noting the size imbalance), and the Theme Explorer's per-theme detail view gets a 4th "Source distribution" chart alongside rating/time/length. `problemstatement.md` itself is left untouched (treated as the immutable original spec, same precedent as Phase 8) - this file, `architecture.md`, `Implementation-plan.md`, and `README.md` carry the addendum instead.

> **Maintenance:** Update §11 as milestones complete and record any deviations from the defaults in §8.
