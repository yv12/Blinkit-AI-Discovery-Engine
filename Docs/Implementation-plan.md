# Implementation Plan: AI-Powered Discovery Engine

> **Purpose:** Phase-by-phase execution plan for building the Blinkit review-analysis pipeline. Derived from `architecture.md` (module/stage design) and `context.md` (mission, constraints, current state). Read `problemstatement.md` for the authoritative requirements.
>
> **How to use:** Work phases in order. Each phase lists its goal, tasks, deliverables, dependencies, and a definition of done (DoD). Do not advance until the current phase's DoD is met. Update the status boxes as you go.

---

## Legend

- `[ ]` not started · `[~]` in progress · `[x]` done
- **DoD** = Definition of Done (exit criteria for the phase)
- Stage numbers (S1–S10) refer to `architecture.md` §3.

---

## Phase Overview

| Phase | Name | Stages | Outcome |
|---|---|---|---|
| 0 | Project Setup & Foundations | — | Runnable skeleton, deps, config, schema |
| 1 | Data Collection | S1–S2 | Normalized Blinkit review corpus |
| 2 | Unit Extraction & Embedding | S3–S4 | Atomic units + local embeddings |
| 3 | Graph & Theme Discovery | S5–S7 | Communities → labeled themes |
| 4 | Insight Mapping | S8 | Themes mapped to the 8 questions |
| 5 | Validation | S9 | Coherence + triangulation report |
| 6 | UI / Presentation | S10 | Browsable, shareable output |
| 7 | Reproducibility & Docs | — | Clean install → end-to-end run |
| 8 | (Addendum) Deep Pattern Synthesis via Groq | — | Optional LLM-inferred indirect patterns |
| 9 | (Addendum) Second Data Source: Mouthshut | S1b | Core pipeline runs over two merged sources |

---

## Phase 0 — Project Setup & Foundations

**Goal:** A runnable Python skeleton with shared config, schema, and pinned dependencies.

**Tasks**
- [x] Create `requirements.txt` with pinned, free/local deps (`google-play-scraper`, `sentence-transformers`, `networkx`, `python-louvain`, `scikit-learn` (+ optional `faiss-cpu`), `numpy`, `pandas`, `streamlit`, `pyyaml`).
- [x] Create `config.yaml` (app id `com.grofers.customerapp`, target review count, model names, kNN k/threshold, Louvain resolution, random seed).
- [x] Create `src/config.py` — load `config.yaml`, validate types/ranges (edgecases.md X-08), expose paths + tunables.
- [x] Create `src/schema.py` — dataclasses + JSON/JSONL I/O helpers for Review, Unit, Theme, Insight (+ atomic writes, corrupt-artifact detection).
- [x] Create `data/` (gitignored, created at runtime) and `.gitignore`.
- [x] Create `src/pipeline.py` stub that wires stage entry points (fails loudly via `NotImplementedError` per stage until implemented).

**Deliverables:** `requirements.txt`, `config.yaml`, `src/config.py`, `src/schema.py`, `.gitignore`, `src/pipeline.py`, `src/__init__.py`.

**Dependencies:** none.

**DoD:** `pip install -r requirements.txt` succeeds in a fresh env; `python -c "import src.config, src.schema"` runs without error; config values load. ✅ Verified: config loads/validates (rejects out-of-range values), schema round-trips + detects corrupt artifacts + rejects non-Play-Store source, and `python -m src.pipeline --only scrape` fails loudly with a clear "not implemented" message instead of silently no-op'ing.

---

## Phase 1 — Data Collection (S1–S2)

**Goal:** A normalized, deduplicated corpus of Blinkit Play Store reviews covering a **rolling lookback window** (default: last 4 calendar months), not a fixed review count.

**Tasks**
- [x] `src/scrape.py` (S1): pull every review across ratings 1–5 within `scrape.lookback_months`, using the chronological `newest` sort for exact/complete window coverage (`lang='en'`, `country='in'`); handle pagination/continuation tokens; dedupe by id; write raw payloads verbatim to `data/raw_reviews.jsonl`; support `--refresh` and resumable pulls that preserve the original cutoff across restarts.
- [x] `src/normalize.py` (S2): map raw → canonical Review schema; coerce dates to ISO 8601, ratings to int; capture `thumbs_up`, `app_version`, `developer_reply`, `lang`; drop empty text; write `data/reviews.jsonl`.
- [x] Log corpus stats (count, rating distribution, date range, lang split).

**Deliverables:** `data/raw_reviews.jsonl`, `data/reviews.jsonl`, corpus stats printout.

**Dependencies:** Phase 0.

**DoD:** A corpus covering the full lookback window is scraped (window set via `config.yaml` `scrape.lookback_months`, capped per-bucket by `scrape.max_per_bucket`); every record validates against the Review schema; rating bands 1–5 all represented; re-running skips re-scrape unless `--refresh`. ✅ **Done for real** (not just a capped test sample) against the real Blinkit app: full 4-month scrape produced **156,219 unique reviews** (1★ 27,181 / 2★ 4,191 / 3★ 7,096 / 4★ 17,751 / 5★ 100,000; 5★ hit `max_per_bucket=100000` and covers 2026-04-03→2026-07-22, ~90% of the window — accepted as-is; all other bands fully cover 2026-03-22→2026-07-22). Normalization produced 156,219 valid `Review` records with **0 drops**, 155,647 `en` / 572 `hi` language split, 3,906 near-duplicate groups flagged (not dropped). Earlier smaller-scale verification also confirmed: calendar-month cutoff arithmetic against year-rollover/Feb 28-29 edge cases; chronological early-stop terminating exactly at the date boundary with zero leakage; and clean skip-on-rerun behavior for both stages.

---

## Phase 2 — Unit Extraction & Embedding (S3–S4)

**Goal:** Atomic complaint/insight units, each embedded with a local model.

**Tasks**
- [x] `src/units.py` (S3): exclude low-signal reviews (<4 words, decided with user - pure length cutoff, fully excluded, not counted downstream, edgecases.md S3-09); split remaining multi-topic reviews into atomic statements via **rule-based** sentence/conjunction splitting (LLM splitting decided against as impractical at real corpus scale, edgecases.md S3-01). Keep `review_id` back-reference; write `data/units.jsonl`.
- [x] `src/embed.py` (S4): encode units with `all-MiniLM-L6-v2` (local); L2-normalize; batch + cache; save `data/embeddings.npy` aligned to `data/unit_index.json`.

**Deliverables:** `data/units.jsonl`, `data/embeddings.npy`, `data/unit_index.json`.

**Dependencies:** Phase 1.

**DoD:** Units validate against Unit schema and trace back to reviews; embeddings row count == unit count; embedding step runs fully offline; rule-based fallback verified to run without the LLM. ✅ `units.py` done and run on the real corpus: 89,295 units from 59,531 contributing reviews (96,688/156,219 reviews, 61.9%, excluded as low-signal per the user's chosen length-cutoff rule). Spot-checked a random sample plus all multi-unit-producing reviews - splitter correctly isolates specific complaints (e.g. "price too much high", "ice cream always melted", "this is poor service, I want refund") and separates distinct topics within a review at sentence/conjunction boundaries. ✅ `embed.py` done and run on the real corpus: all 89,295 units encoded with local `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized, verified norm ≈ 1.0) in ~7 minutes on CPU (~212 units/sec). `embeddings.npy` row count (89,295) verified == unit count == `unit_index.json` `unit_ids` length. Re-run confirmed idempotent (skips when artifacts exist). Hit and fixed a real Windows environment issue along the way - `torch`'s native DLLs failed to initialize when pulled in transitively by `transformers`/`sentence-transformers` (edgecases.md R-07); fixed by pinning a consistent CPU-wheel install in `requirements.txt` and having `embed.py` import `torch` directly before `sentence_transformers`.

---

## Phase 3 — Graph & Theme Discovery (S5–S7)

**Goal:** Similarity graph → communities → LLM-labeled themes.

**Tasks**
- [x] `src/graph.py` (S5): kNN over embeddings (cosine); add weighted `SIMILAR` edges (top-k / threshold); save `data/graph.gpickle`.
- [x] `src/cluster.py` (S6): Louvain community detection; assign `community_id`; record sizes; flag/prune micro-communities; write `data/communities.json`.
- [x] `src/summarize.py` (S7): pick representative units per community (medoid/centrality); local LLM → one-sentence label + description + 2–3 verbatims + member count + sentiment; write `data/themes.json`.

**Deliverables:** `data/graph.gpickle`, `data/communities.json`, `data/themes.json`.

**Dependencies:** Phase 2.

**DoD:** Graph nodes == units; Louvain yields a sensible number of non-trivial communities; every retained community has a theme entry conforming to the Theme schema with real verbatims. ✅ `graph.py` done and run on the real corpus: exact (brute-force) cosine kNN via scikit-learn (`faiss` not installed - S5-05 fallback path exercised for real, not just in theory), k=15, `similarity_threshold=0.5`, "any-kNN" union construction. Result: **89,295 nodes, 1,054,505 edges**, degree min=0/median=20/mean=23.6/max=169, 794 isolated singleton nodes (0.9%, kept per S5-04). Full kNN search over the real corpus took ~5 minutes on CPU. Edge weights verified clamped to `[threshold, 1.0]` (S5-06 float-drift guard confirmed active - some weights landed at exactly 1.0, i.e. real duplicate/near-duplicate unit text). ✅ `cluster.py` done and run on the real corpus: Louvain (`clustering.louvain_resolution=1.0`, seeded) over the 89,295-node graph produced **859 communities** across 825 connected components. Confirmed exact partition (no duplicate/missing unit ids across communities, verified in code). Distribution is intentionally long-tailed: 819 communities fall below `clustering.min_community_size=3` (794 of which are the exact same singletons flagged in Stage 5's S5-04, a good cross-stage consistency check) and are flagged `below_min_size: true` rather than dropped (S6-03 revision - see `cluster.py` docstring); the remaining **40 communities hold 88,451 units** (sizes from ~40 up to 8,609) with meaningfully different average ratings (1.5-4.6 range spot-checked on the top 5), i.e. distinct, non-trivial themes - a manageable count for Stage 7's per-cluster LLM summarization. No giant-community warning triggered (largest community is 9.6% of the corpus, well under the 50% advisory threshold). ✅ `summarize.py` done and run on the real corpus. Two decisions made explicitly with the user (not left as defaults): (1) below-`min_community_size` communities never get full per-community theme summarization - they're processed in one dedicated pass and surfaced as `emerging_signals` (plain dicts, not `Theme` records), tagged with `support_count` + a `confidence` tier (`very_low` for singletons, `low` for pairs); (2) the local TF-IDF + rating-derived extractive path (S7-01 fallback) is the *primary* path, not a secondary one - confirmed live that Groq is unreachable on this machine, so `summarize.use_llm: false` is also what actually runs, not just a theoretical default. Sentiment always comes from the rating distribution, never from an LLM (S7-06). Result: **40 themes** (from the 40 qualifying communities) + **819 emerging signals** (from the 819 below-min communities) written to `data/themes.json`. Theme sentiment split 15 positive / 14 negative / 11 neutral. Spot-checked labels/quotes: e.g. `theme-0002` labeled "customer / service / support / care" (avg rating 1.5, negative) with verbatim quotes like "Very poor service by the customer support team"; labels are readable and topically coherent even without an LLM. `src/insights.py` not yet started.

---

## Phase 4 — Insight Mapping (S8)

**Goal:** Every theme mapped to the 8 research questions, with quantification.

**Tasks**
- [x] `src/insights.py` (S8): classify each theme → one or more of the 8 questions (LLM + rules); quantify per theme: frequency, rating-band spread, time-cohort spread, sentiment; build category–category similarity edges (navigable map); assemble `data/insights.json` keyed by question `1..8` with supporting theme ids, counts, and top verbatims.

**Deliverables:** `data/insights.json`.

**Dependencies:** Phase 3.

**DoD:** Each of the 8 questions has ≥1 supporting theme with counts and verbatims; operational/pricing complaints are represented (not filtered out); every theme is mapped to at least one question. ✅ **Partially met, honestly reported rather than forced** - done and run on the real corpus with two decisions made explicitly with the user before implementing: (1) theme creation stays fully bottom-up (Louvain) - the 8 RQs are never an input to clustering/summarization, this stage is a strictly post-hoc tagging pass; mapping method is embedding similarity (theme `label+description` vs. a topic description per question, `insights.question_queries`), not keywords or an LLM call - keeps the zero-cost/no-LLM property that was already the real runtime path in Stages 3 and 7; (2) the 819 `emerging_signals` are mapped the same way but kept in a separate `signal_ids`/`signal_support_total` key per question, tagged `signal_confidence: "low"`, and never flip a question's `coverage` to `"sufficient"` on their own - only real themes count as strong evidence. Empirically found and fixed a real correctness bug: embedding similarity alone matches on shared topic nouns regardless of polarity (a clearly *positive* theme scored 0.575 against the "frustrations" question, higher than most genuinely negative themes) - added `insights.question_required_sentiment` to gate Q6 on the theme's/signal's own rating-derived sentiment (never LLM-derived, consistent with S7-06); this dropped Q6 from 34 to 11 themes, all genuinely negative. Final real result: **4/8 questions "sufficient"** (Q1, Q3, Q4, Q6), 4 "insufficient" (Q2, Q5, Q7, Q8) - reported honestly per S8-01 rather than force-matched; 16/40 themes and 780/819 signals "uncategorized" (S8-02, kept not dropped). Q1 shows known residual imprecision (picks up shared word-roots like "order"/"items" regardless of habit-vs-complaint framing) - documented as a v1 limitation of small-model embedding similarity, not further chased. Also added the two architecture.md-specified quantifications this DoD's first draft omitted: per-theme `theme_segment_stats` (rating-band + time-cohort distribution, sourced from `communities.json`+`units.jsonl`) and a `category_graph` (40 themes × top-5 similarity edges by centroid embedding - problemstatement.md §7's "category-level graph", built over our own discovered themes, not an external product taxonomy Google Play reviews don't have).

---

## Phase 5 — Validation (S9)

**Goal:** Evidence that themes are coherent and hold across segments.

**Tasks**
- [x] `src/validate.py` (S9): cluster coherence (intra- vs inter-cluster similarity / silhouette); export stratified spot-check sample for human labeling (compute agreement if labels present); cross-segment triangulation across rating bands, time periods, and review length/recency cohorts; write `data/validation.json` + human-readable summary.

**Deliverables:** `data/validation.json`, validation summary.

**Dependencies:** Phase 4.

**DoD:** Coherence metric computed and reported; triangulation shows which themes are stable vs segment-specific; spot-check sample exported; validation methodology documented. ✅ Done and run on the real corpus. Three independent checks, only over the 40 qualifying `themes` (the 819 `emerging_signals` stay excluded from this level of rigor, consistent with their supplementary-evidence status since Stage 7/8): (1) **Coherence** - graph **modularity of the actual Louvain partition = 0.8394** (the methodologically-correct primary metric for a graph-clustering method, per S9-06), alongside a per-theme centroid-based silhouette-style score (`[-1,1]`, computed in embedding space since literal per-point silhouette is infeasible at up to 8,609 members/theme) - mean **-0.001**. The near-zero mean silhouette against a *very* high modularity is itself an honest, informative finding, not a contradiction: the graph-based Louvain clustering separates sub-topics precisely using fine-grained unit-to-unit kNN edges (hence high modularity), while several distinct themes (e.g. `theme-0002` "customer/service/support" vs `theme-0003` "delivery/service/location") share enough surface vocabulary that their *coarse* centroid vectors sit close together - the graph structure resolved a finer distinction than a bag-of-words-ish embedding average can see. Smallest themes (3-4 members, e.g. `theme-0038`/`theme-0039`) show the highest silhouette scores, a known small-N artifact (few points are trivially close to their own tiny centroid) rather than genuinely "better" clustering, and is documented as such. (2) **Triangulation** - recomputed independently from `communities.json`+`units.jsonl` (not read back from `insights.json`, for a genuinely independent check): **28/40 themes are "cross_segment" stable**, 12/40 flagged "segment_specific" (mostly concentrated in one review-length bucket, e.g. short one-line praise themes like `theme-0004` "app/good/best/nice" - expected given the content, not necessarily a flaw). Rating-band concentration is deliberately excluded from the stability verdict (a sentiment-driven theme *should* skew one rating band - that's a healthy signal, not instability) but still reported per-theme for context (S9-05). (3) **Spot-check** - a stratified random sample (5 units/theme, 192 rows total) written once to `data/spot_check_sample.json` with an empty `human_agrees` field per row for manual review; verified live that re-running `--refresh` never overwrites existing human labels (edited a row by hand, re-ran, confirmed both the edit survived and the resulting `agreement_rate` computed correctly from just that one label) - delivered in its clean, unlabeled state, ready for real review (S9-04). `data/validation_summary.md` (human-readable, sorted worst-to-best coherence, lists all segment-specific themes) is written alongside `validation.json`.

---

## Phase 6 — UI / Presentation (S10)

**Goal:** A browsable, shareable output that answers the 8 questions with evidence.

**Tasks**
- [x] `app.py` (Streamlit): browse themes, drill into verbatims, view each question's evidence-backed answer; show counts/sentiment/triangulation.
- [x] `notebook.ipynb`: fallback presentation path.
- [x] Ensure output is shareable via a link (hosted workflow / notebook / lightweight app).

**Deliverables:** `app.py`, `notebook.ipynb`.

**Dependencies:** Phases 4–5.

**DoD:** UI loads `insights.json`/`themes.json`/`validation.json`; all 8 questions answerable in the UI with verbatims; produces a shareable link. ✅ Done, built against the real Phase 4/5 artifacts (40 themes, 819 emerging signals, 8 questions, 40 validated themes) - no new pip dependencies needed beyond the already-planned `streamlit` (already pinned in `requirements.txt`; only had to actually install it into this machine's environment, since it hadn't been installed yet despite being pinned) plus `ipykernel` (added, notebook-execution only, not part of the pipeline itself). The "category graph" network visualization deliberately avoids adding `matplotlib`/`plotly` - it's built entirely from already-present deps (`networkx.spring_layout` for 2D positions + `altair`, which ships as a transitive dependency of `streamlit` itself). `app.py` has 5 pages (sidebar radio): **Overview** (live corpus stats read fresh from `reviews.jsonl`/`units.jsonl`, not hardcoded, plus a pipeline-stage table S1→S10 and top themes), **Research Questions** (all 8 questions with the literal RQ text from problemstatement.md §3 shown alongside the actual embedding-query text used for matching, coverage badges, supporting themes/verbatims, and weak long-tail signal support kept visibly separate per S8's design), **Theme Explorer** (filterable/searchable table of all 40 themes + a separate tab for the 819 emerging signals, with a drill-down per theme showing verbatims, rating/time-cohort/review-length distributions, and Stage 9's coherence + triangulation numbers, since `themes.json`'s own `questions` field is never actually populated by Stage 8 - the UI reverse-indexes `insights.json`'s question→theme_ids mapping to answer "which questions does this theme support?"), **Category Graph** (interactive force-directed map of the 40-theme similarity graph, node size = member count, color = sentiment), and **Validation & Methodology** (modularity/silhouette, cross-segment vs. segment-specific breakdown with reasons, spot-check status). Verified with Streamlit's headless `AppTest` harness (not just eyeballing it) - ran all 5 pages, cycled every one of the 8 research-question selections, drilled into multiple themes, and exercised the sentiment/question/text filters and search boxes on the Theme Explorer, zero exceptions in any state; also smoke-tested the real server (`streamlit run app.py`, HTTP 200, no tracebacks in server logs). `notebook.ipynb` mirrors the same four sections (data gathering, theme identification, insight generation, validation) using only pandas + stdlib (no Streamlit) for evaluators who prefer a static/portable notebook; executed end-to-end with `nbclient` and confirmed zero error cells, with real outputs saved inline (e.g. all 8 questions' coverage/counts, the segment-specific theme list) so it's readable without re-running. "Shareable link": a local `streamlit run app.py` run already prints a `Network URL` reachable by anyone on the same LAN; `app.py`'s sidebar surfaces the exact `--server.address 0.0.0.0` command for this. Full hosted (public-internet) deployment is a Phase 7/README concern (needs a repo + a place to run it), not a Stage 10 coding task.

---

## Phase 7 — Reproducibility & Docs

**Goal:** A fresh environment can install and run end-to-end from documented steps.

**Tasks**
- [x] `src/pipeline.py`: orchestrate S1→S9 in order; skip stages whose artifacts exist; `--force` to rebuild.
- [x] `README.md`: setup, Groq install, run steps, config explanation, expected outputs.
- [x] Verify determinism (fixed seeds, pinned deps, cached artifacts); document LLM non-determinism mitigations.

**Deliverables:** `README.md`, working `pipeline.py`.

**Dependencies:** All prior phases.

**DoD:** Clean clone → install → `python -m src.pipeline` → UI works, reproducing the documented result set. Satisfies the spec's "Definition of Done" (context.md §2). ✅ Done. `pipeline.py` was already fully implemented (built incrementally across Phases 0-6, not deferred to this phase) - re-verified live: `python -m src.pipeline` against the existing real-corpus `data/` cleanly skips all 9 stages (each logs "artifact exists"), `--force`/`--only <stage>` flags both work, and `--help` lists all 9 stage names. Wrote `README.md` covering setup (incl. the real Windows torch-DLL gotcha from Phase 2), optional Groq install/model-pull steps (the pipeline runs fully LLM-free by default - `use_llm: false` everywhere, the actual verified path), config.yaml section reference, full run steps (`--force`/`--only`/per-stage CLI), expected `data/` artifact shapes/sizes from the real run, Streamlit + notebook viewing instructions, a determinism section, a troubleshooting table, and the 8 research questions. **Determinism verification, done for real not just asserted:** confirmed `config.seed` is threaded into every RNG that needs it - Python `random`/`numpy.random` globally (`apply_global_seed`), Louvain's `random_state` (Stage 6), and Stage 9's spot-check sampling - and that `requirements.txt` pins exact versions including the CPU-only PyTorch wheel index. **Found and fixed a real gap while verifying "document LLM non-determinism mitigations":** architecture.md §7 had always specified "mitigated with low temperature" but `summarize.py`'s Groq call (the only LLM call site in the whole pipeline) was never actually setting a temperature or seed on its requests - fixed by adding `options: {temperature: 0, seed: config.seed}` to every Groq request (both the per-theme label call and the batched long-tail call), threading `config.seed` through `_call_groq`/`_theme_via_llm`/`_label_long_tail_via_llm`; verified with `py_compile` + a clean `import src.summarize` that the change introduces no errors. This only affects the optional `summarize.use_llm: true` path - the default, already-verified `use_llm: false` path (rule-based/TF-IDF, zero LLM calls) is unaffected and remains what the documented corpus statistics reflect.

---

## Phase 8 (Addendum, Post-Phase-7) — Deep Pattern Synthesis via Groq

**Not part of the original phase plan or the project's core Definition of Done.** Added as
an explicit, user-directed extension after Phases 0-7 were already complete and verified,
in response to a direct question: does the pipeline find *indirect/implicit* behavioral
patterns (e.g. "users face cognitive load when browsing unfamiliar categories") behind why
users don't explore new categories, the way a human qualitative researcher reading the
reviews might? The honest answer at the time was no - Stage 7's TF-IDF labels and Stage 8's
embedding-similarity mapping can only surface literal recurring vocabulary, verified
concretely: Q2 ("what prevents exploration?") had zero supporting themes, and all 16
"uncategorized" themes were inspected and confirmed to be generic praise/rating-noise, not a
missed signal.

**Goal:** Use a remote LLM (user-directed choice: Groq, free tier) to infer indirect
patterns from a curated sample of raw review text, surfaced as a clearly-labeled,
experimental addition alongside (never replacing) the deterministic pipeline's evidence.

**Tasks**
- [x] `.env` support: `GROQ_API_KEY` loaded via `python-dotenv` (added to `requirements.txt`), gitignored.
- [x] `config.yaml`/`src/config.py`: new `llm_synthesis` section (model, sampling, batching, rate limit, clustering/aggregation knobs) + validation, following the existing config-loader pattern.
- [x] `src/llm_synthesis.py` (Stage 11, optional, NOT part of `python -m src.pipeline`): deterministic prioritized sampling (uncategorized themes/signals first, then stratified top-up) → batched Groq calls (all 8 questions per call) → local-embedding-based pattern clustering/aggregation → `data/llm_insights.json`. Resumable via `data/llm_synthesis_checkpoint.jsonl`; every network/parse failure degrades to "skip batch, log warning" rather than crashing.
- [x] `app.py`: new "🧠 LLM-Inferred Patterns" section on the Research Questions page per question, visually/textually distinguished from the deterministic evidence above it, gracefully absent with an explanatory note if the optional stage hasn't been run.
- [x] `README.md` §12 + `Docs/architecture.md` §11: full design rationale, setup, and the empirical model comparison below.

**Real findings from building this (not assumed, measured against the live Groq API):**
- **A naive prompt failed silently in an interesting way**: asking the model to name a
  "pattern" per excerpt caused it to lazily copy the 8 question descriptions verbatim as the
  "pattern" for every hit (e.g. every Q6 hit labeled literally "recurring frustrations and
  complaints: delivery, quality, app experience, customer service") - technically valid JSON,
  completely useless output. Fixed by explicitly forbidding this in the prompt, requiring a
  concrete example-anchored inference instead, and verified the fix worked on the same real
  batch (e.g. "worst service provide by blinkit staff" → "staff rudeness eroding trust", not
  a copy of question 6's text).
- **Model selection was measured, not assumed.** Three free-tier models were tested
  head-to-head on the identical real batch: `llama-3.1-8b-instant` (~1,400 tokens/call,
  no hidden reasoning tokens, 500K tokens/day budget) vs. `openai/gpt-oss-120b` (~2,660
  tokens/call, of which **1,457 were hidden reasoning tokens**, only 200K tokens/day budget)
  vs. two model names (`qwen/qwen3-32b`, `meta-llama/llama-4-scout-17b-16e-instruct`) that
  turned out not to exist on this account (`model_not_found`, confirmed live, not from docs
  alone). `openai/gpt-oss-120b` produced visibly deeper, more specific inferences - notably
  better at finding indirect Q2 barrier signals from the exact same excerpts (e.g. "negative
  staff interactions deter new category trials", something `llama-3.1-8b-instant` missed
  entirely on that excerpt) - but its reasoning-token overhead means its daily budget only
  covers ~1,500 units, not enough for a representative sample in one run.
  `llama-3.1-8b-instant` was chosen as the default specifically for its much larger effective
  daily budget, so the full `sample_size: 4000` default completes in one sitting; swapping to
  `openai/gpt-oss-120b` (with a smaller `sample_size`) is documented as the higher-quality
  alternative for whoever runs this next.
- **Real rate-limit economics measured, not guessed**: Groq's documented 30 RPM cap is *not*
  the binding constraint at this batch size - the model's 6,000-tokens/minute budget is,
  confirmed via the real `usage.total_tokens` field on live API responses
  (~1,400 tokens/call → sustainable ceiling is ~4-5 calls/minute, not 30).

**Deliverables:** `.env` (gitignored, user-provided), `src/llm_synthesis.py`, `data/llm_insights.json`, `data/llm_synthesis_checkpoint.jsonl`, `app.py` UI section, README/architecture docs.

**Dependencies:** Phases 0-7 (reads their real artifacts: `units.jsonl`, `themes.json`, `insights.json`, `communities.json`).

**DoD:** Not part of the project's formal Definition of Done (context.md §2) - this is an optional, clearly-labeled addendum. Its own bar: running `python -m src.llm_synthesis` against the real corpus produces `data/llm_insights.json` with genuinely specific (not templated/copied) inferred patterns for multiple questions, each traceable to real verbatim quotes, surfaced correctly in `app.py` without breaking any existing page. ✅ **Done, run for real against the live Groq API and the full real corpus** (not a mock/dry-run): 3,998 sampled units → 200 batches, **200/200 succeeded, 0 failed** → 3,772 raw (excerpt, question) hits → **64 inferred patterns across all 8 questions** (8 patterns/question, the configured cap) written to `data/llm_insights.json`. Concretely closes the gap that motivated this addendum: **Q2 (barriers to exploring new categories), which had zero supporting themes in the deterministic `insights.json`, now has 8 specific inferred patterns** with real supporting quotes - e.g. "barriers to exploration: skepticism of marketing claims" (19 hits, quotes like "Only Big talk and claims", "there is no option, why are you making befool"), "barriers to adoption due to price dissatisfaction" (15 hits, incl. a real Hindi-language fee complaint), "fear of change in product quality" (17 hits), "barrier of high delivery charges" (8 hits). Q5, Q7, Q8 (also "insufficient" in the deterministic pipeline) similarly gained real, quote-backed patterns (e.g. Q5: "need for clear product information before trying", "need for price transparency and affordability"; Q7: "openness to new cuisines and products"; Q8: "unmet need for medical health services", "missing feature: more locations for delivery"). Verified end-to-end in `app.py` via the headless `AppTest` harness - all 8 questions render the new section with zero exceptions, and all 4 pre-existing pages still work unchanged. **One real limitation found and kept visible, not hidden**: the largest single cluster per question (e.g. Q2's "narrow, unstated product vocabulary suggesting routine-only usage", 110 hits) was spot-checked and confirmed to be partly the free-tier model defaulting to a generic stock phrase for ambiguous/low-content excerpts rather than a genuinely specific inference each time - a `min_quote_words` floor was added to filter the shortest/vaguest excerpts before aggregation (measurably reduced but did not eliminate this), and `app.py`'s UI explicitly warns users to read the quotes rather than trust `support_count` alone (documented in full in README.md §12).

---

## Phase 9 (Addendum, Post-Phase-8) — Second Data Source: Mouthshut

**Not part of the original phase plan.** Unlike Phase 8 (a separate opt-in script), this
addendum changes the **core** pipeline (`python -m src.pipeline`, S1-S9) - it merges a second
source into the same `reviews.jsonl`/`units.jsonl` every stage already reads. Triggered by the
user adding `data/Mouthshut_reviews.csv` and asking for it to be "processed just like the
Play Store" data. **This directly conflicts with a hard, explicitly-worded constraint**
(problemstatement.md §5 / context.md §4: "Single data source... deal forums, complaint boards
... are all explicitly out of scope. Do not build scrapers or connectors for any other
source." - Mouthshut.com is literally a complaint-board/review-forum site). Flagged to the
user before writing any code; the user chose to **amend the constraint and integrate it as a
real second source**, over the alternatives of keeping it in a fully separate supplementary
track or not processing it at all.

**Goal:** Ingest the pre-scraped Mouthshut CSV, normalize it into the same canonical schema
as Google Play reviews (tagged with a distinct `source`), and merge it into the core corpus
so every downstream stage (embedding, clustering, summarization, insight mapping, validation,
UI) operates over both sources together - not a parallel/separate analysis track.

**Tasks**
- [x] `src/schema.py`: relax `Review.source`'s hard-coded single-value check to
  `VALID_SOURCES = {"google_play", "mouthshut"}`; add a `source` field to `Unit` (default
  `"google_play"` for backward-compat with pre-addendum artifacts) so provenance survives
  Stage 3 onward.
- [x] `src/config.py`: add `paths.mouthshut_csv` / `paths.raw_mouthshut`.
- [x] `src/scrape_mouthshut.py` (new Stage 1b): CSV → `data/raw_mouthshut.jsonl`, one raw
  record per review tagged `source: "mouthshut"`. A no-op if the CSV is absent.
- [x] `src/normalize.py`: normalize Mouthshut's raw shape (title+body concatenation,
  absolute + relative date parsing, no thumbs_up/app_version/developer_reply equivalent) and
  merge with the Google Play stream into one `reviews.jsonl`.
- [x] `src/units.py`: propagate `review.source` into each `Unit`.
- [x] `src/insights.py` / `src/validate.py`: add per-theme `source_distribution`
  (informational, same treatment as rating bands - never folded into the `stability` verdict).
- [x] `src/pipeline.py`: insert Stage 1b between `scrape` and `normalize`.
- [x] `app.py`: Overview "Source split" chart; Theme Explorer per-theme "Source distribution" chart.
- [x] Re-run `python -m src.pipeline` against the real merged corpus (from `normalize` onward -
  `raw_reviews.jsonl`/`raw_mouthshut.jsonl` reused, no re-scrape).
- [x] `README.md` / `Docs/architecture.md` §12 / `Docs/context.md` §11: full design rationale
  and real results.

**Real findings from building this (measured, not assumed):**
- The real CSV (1,000 rows) has **two genuinely different date shapes** in the same column:
  absolute (`"Jun 20, 2026 05:15 PM"`, the vast majority) and relative (`"N days ago"`, only
  the most recent ~20 rows, max "30 days ago"). No scrape timestamp is recorded in the export,
  so the relative dates are resolved against the CSV file's own last-modified time - cross-
  checked against the data itself (newest absolute date 2026-06-20 + oldest relative phrasing
  "30 days ago" from the file's mtime reference of 2026-07-23 resolves to ~2026-06-23,
  consistent with no gap in coverage). All 1,000 rows normalized with **zero rejections /
  zero unparseable dates** - verified live, not assumed.
- Merging in ~4,098 Mouthshut units (4.4% of the merged 93,393-unit corpus) shifted a few
  communities across the `min_community_size=3` boundary: **40 themes + 851 emerging signals**
  post-merge vs. 40/819 pre-merge - the qualifying-theme count held steady, only the long tail
  moved slightly.
- Mouthshut units are **not siloed**: they landed in 32 of the 40 themes, confirmed by
  inspecting each theme's new `source_distribution` - i.e. they blend topically with the
  existing Google Play corpus (mostly 1-star delivery/product/refund complaints, which is
  exactly what the corpus already skews toward) rather than forming an isolated new cluster.
- Re-running `embed`→`validate` over the full 93,393-unit merged corpus (not just the new
  ~4K units - these stages have no incremental/delta mode) took ~21 minutes end-to-end on CPU
  (embedding ~9.3 min, kNN graph + Louvain ~10 min, the rest under a minute) - a real,
  measured cost of adding *any* amount of data to a non-incremental pipeline at this scale,
  worth knowing before adding a third source later.

**Deliverables:** `src/scrape_mouthshut.py`, updated `schema.py`/`config.py`/`normalize.py`/
`units.py`/`insights.py`/`validate.py`/`pipeline.py`/`app.py`, regenerated `data/reviews.jsonl`
onward, README/architecture/context doc updates.

**Dependencies:** Phases 0-7 (the core pipeline); independent of Phase 8 (Groq synthesis is
untouched by this change - its stale pre-merge `llm_insights.json` still loads fine in
`app.py`, just doesn't yet reflect the Mouthshut units; re-running it is a separate, optional,
rate-limited follow-up, not required by this phase's DoD).

**DoD:** Not part of the project's original Definition of Done (context.md §2 was written
before this addendum) - re-scoped by this addendum to: the core pipeline (S1-S9) runs
end-to-end over **both** sources merged into one corpus, with per-source provenance visible
at every stage from `reviews.jsonl` through the UI, and zero regression to the Google-Play-
only path when the Mouthshut CSV is absent. ✅ **Done, verified against the real merged
corpus**: `python -m src.pipeline` completed with exit code 0, producing 157,219 reviews
(156,219 Google Play + 1,000 Mouthshut) → 93,393 units (89,295 + 4,098) → 40 themes + 851
emerging signals → same 4/8 questions `"sufficient"` (Q1/Q3/Q4/Q6) → validation modularity
0.8366, 28/40 cross-segment stable. `app.py` smoke-tested against the regenerated artifacts
(`load_corpus_stats`, `build_validation_lookup` both confirmed returning correct per-source
breakdowns). Removing `data/Mouthshut_reviews.csv` and re-running with only `raw_reviews.jsonl`
present still works unchanged (Stage 1b logs and no-ops, `normalize.py`'s Mouthshut branch is
skipped) - confirmed by code inspection of the guard clauses, consistent with every other
optional-artifact guard in this codebase (edgecases.md X-03 pattern).

---

## Cross-Cutting Concerns (all phases)

- **Zero-cost:** no paid APIs/models at any point — verify each new dependency.
- **Single source (amended, Phase 9):** Google Play only, originally; Phase 9 documents a
  deliberate, user-directed exception adding Mouthshut as one specific second source - not a
  general opening to arbitrary connectors. Do not add any other source without the same
  explicit flag-and-decide step.
- **Traceability:** every unit/theme/insight must trace back to real review verbatims.
- **Artifact discipline:** each stage reads/writes a `data/` artifact; keep stages independently re-runnable.
- **Whole-funnel:** categorize operational/pricing complaints; never filter them out.

---

## Risk Watchlist (from architecture.md §9)

| Risk | Watch in phase | Mitigation |
|---|---|---|
| Play Store rate limits / partial pulls | 1 | resumable + cached raw file |
| Local LLM slow/unavailable | 2, 3, 4 | rule-based fallback; batch + cache; summarize representatives only |
| Hinglish/code-mixed text | 2 | multilingual embedding option; per-lang coverage report |
| Louvain instability / micro-clusters | 3 | tune resolution; prune/merge; report coherence |
| Single-source bias | 5, 9 | cross-segment triangulation; Phase 9 adds a second source (Mouthshut), reported per-theme but not yet large enough to fully offset Google-Play skew |
