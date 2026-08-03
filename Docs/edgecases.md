# Edge Cases & Failure Modes: AI-Powered Discovery Engine

> **Purpose:** Exhaustive catalog of failure modes, edge cases, and unhandled assumptions for every pipeline stage. Derived from `problemstatement.md` (constraints), `architecture.md` (stages/modules), and `Implementation-plan.md` (phases). Each entry has an **ID**, **Trigger** (when it happens), **Expected behavior** (what should be true), and **Handling** (how the code deals with it).
>
> **How to use:** When implementing a stage, treat its edge-case IDs as a checklist. Reference them in code comments and tests (e.g., `# handles S1-04`).

---

## Severity Legend

- **Blocker** — pipeline cannot produce valid output; must fail loudly or halt.
- **Degraded** — pipeline continues with reduced quality; must warn + record.
- **Cosmetic** — minor; log at debug/info level.

---

## Cross-Cutting Assumptions (apply to all stages)

| ID | Assumption / Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| X-01 | **Zero-cost stack** stays zero-cost | A dependency or model silently requires a paid key/tier | No stage ever calls a paid API | Fail fast at config load if any paid endpoint/key is configured; document allowed deps only | Blocker |
| X-02 | **Source allow-list** invariant (amended, Phase 9) | Code path attempts a source outside the allow-list | Originally only `source == "google_play"` existed; **amended with the user (Docs/context.md §11 Phase 9)** to also allow `"mouthshut"` as one specific, explicitly-tagged second source - not a general opening to arbitrary connectors | `schema.py: VALID_SOURCES = {"google_play", "mouthshut"}`; `Review.__post_init__`/`Unit.__post_init__` reject any value outside this set with a `SchemaError` naming the offending value | Blocker |
| X-03 | Upstream artifact missing | A stage runs before its input file exists | Clear message naming the missing artifact + which stage produces it | Check input path on entry; raise actionable error; suggest the command to generate it | Blocker |
| X-04 | Upstream artifact corrupt / partial | Process killed mid-write; malformed JSON line | Corrupt artifact is detected, not silently consumed | Atomic writes (temp file + rename); validate on read; skip+count bad lines, fail if >threshold | Degraded/Blocker |
| X-05 | Disk full / write failure | No space or permission error | No half-written artifact left behind | Write to temp, fsync, atomic rename; on error clean up temp; surface OS error | Blocker |
| X-06 | Schema drift between stages | A field renamed/removed upstream | Downstream validates required fields | Central `schema.py` validation on read/write; version the schema; fail on missing required keys | Blocker |
| X-07 | Non-determinism from LLM steps | Re-run yields different themes/labels | Reruns are reproducible enough to trust | Low temperature; fixed seeds where supported; cache LLM outputs keyed by input hash | Degraded |
| X-08 | Config value out of range | Negative counts, k=0, threshold >1, bad resolution | Invalid config rejected before work starts | Validate `config.yaml` types/ranges at load; fail with the offending key | Blocker |
| X-09 | Encoding issues (emoji, Devanagari, RTL) | Non-UTF-8 bytes or mixed scripts | Text preserved without mojibake | Force UTF-8 read/write; never `errors='ignore'` silently — replace + count | Degraded |
| X-10 | Empty corpus propagates | Zero reviews scraped → every later stage empty | Each stage detects empty input and stops with a clear reason | Guard clause: if input count == 0, exit stage with explicit message, don't crash deep in a library | Blocker |
| X-11 | PII in review text | Names, phone numbers, order IDs in verbatims | Verbatims shown responsibly | Optional PII scrub/redaction pass before display; at minimum flag; never log full PII at info level | Degraded |

---

## Stage 1 — Scrape (`src/scrape.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S1-01 | Wrong / changed app id | `com.grofers.customerapp` invalid or renamed | Fail before pulling garbage | Verify app exists (fetch app metadata) before review pull; error with the id | Blocker |
| S1-02 | Rate limiting / throttling | Play Store returns 429 / empty batches early | Backoff and resume, don't lose progress | Exponential backoff + jitter; persist raw incrementally; resumable via continuation token | Degraded |
| S1-03 | Pagination token expires / loops | Continuation token stale or repeats | No infinite loop, no dup explosion | Cap max pages; detect repeated token; dedupe by review id | Degraded |
| S1-04 | Fewer reviews than expected | App has < `max_per_bucket` reviews within the lookback window for a rating band | Take what exists, report shortfall | Continue; log actual count vs cap per rating band; don't fail | Degraded |
| S1-05 | Duplicate reviews across sorts | Same review from `newest` and `relevance` pulls | Corpus deduplicated | Dedupe by review id (union of sorts) | Cosmetic |
| S1-06 | Network flakiness / timeout | Connection drop mid-pull | Retry, then resume from last saved | Retries with timeout; checkpoint raw file after each batch | Degraded |
| S1-07 | Library API change | `google-play-scraper` field/signature changed | Detect mismatch early | Pin version in `requirements.txt`; validate expected keys in first batch | Blocker |
| S1-08 | Missing optional fields | Review lacks `app_version`/`repliedAt`/`thumbsUp` | Absent fields tolerated | Default to null/0; never assume presence | Cosmetic |
| S1-09 | `--refresh` vs cache confusion | Stale raw file reused unintentionally | Explicit control over re-scrape | Skip if raw exists unless `--refresh`; log which path taken | Cosmetic |
| S1-10 | Rating band skew | Almost all pulled reviews are 5-star | Downstream triangulation still possible | Pull per-rating where supported; report distribution; warn if a band is empty | Degraded |
| S1-11 | Non-English / Hinglish dominance | `lang='en'` still returns code-mixed text | Language captured, not dropped | Store text as-is; detect + tag `lang`; keep for multilingual embedding | Degraded |
| S1-12 | Date-bounded collection with a non-chronological sort | `scrape.sort` includes a mode other than `newest` (e.g. `relevance`) while a `lookback_months` window is active | Window coverage claims are honest about which sorts can guarantee completeness | For `newest` (chronological): pagination stops exactly at the cutoff - exact and complete. For non-chronological sorts: page up to the `max_per_bucket`/`max_pages` safety cap and filter by date post-hoc; log a warning that completeness is not guaranteed for that sort; `_log_summary` also flags any collected review found older than cutoff | Degraded |
| S1-13 | Safety cap reached before the window is exhausted | A rating band has more reviews than `scrape.max_per_bucket` within the lookback window | Truncation is visible, not silent | Track a `hit_cap` flag per bucket; log a warning naming the bucket and suggesting a higher `max_per_bucket`; still write whatever was collected | Degraded |
| S1-14 | Resuming a scrape after the window's "now" has moved | Process interrupted and restarted on a later day, mid multi-bucket lookback-window scrape | The window used for bucket N+1 must match the window used for buckets 1..N, not drift forward each resume | Persist the originally-computed cutoff in `raw_reviews.progress.json`; resume reuses the stored cutoff rather than recomputing "N months ago" from the new `now` | Degraded |

---

## Stage 1b — Ingest Mouthshut (`src/scrape_mouthshut.py`, Addendum, Post-Phase-8)

**Not part of the original 10-stage plan.** Added as Phase 9 (Docs/context.md §11,
Docs/Implementation-plan.md, Docs/architecture.md §12) - a deliberate, user-directed
amendment of the X-02/S1 single-source constraint that merges a second, pre-scraped source
(Mouthshut.com) into the core corpus. Unlike Stage 11 (Groq, a fully separate opt-in script),
this stage sits inline in `python -m src.pipeline`, between Stage 1 and Stage 2.

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S1b-01 | Source CSV absent (the default, for anyone who never adds it) | `data/Mouthshut_reviews.csv` does not exist | Pipeline behaves identically to before this addendum - zero regression | `ingest_mouthshut()` logs and returns without writing anything; `normalize.py` only reads `raw_mouthshut.jsonl` if it exists, so the merge step itself is skipped too | Cosmetic |
| S1b-02 | CSV present but wrong shape | Export missing one of `review_url, title, body, rating, reviewer, review_date` | Fail loudly before ingesting partial/misaligned data, not a confusing downstream `KeyError` | `MouthshutIngestError` raised naming the missing columns and the actual header found | Blocker |
| S1b-03 | No scrape timestamp recorded in the source | Mouthshut's own scrape/export date isn't in the CSV, but some rows use relative dates (`"N days ago"`) that need an anchor | A documented, cross-checked best-effort reference, not a silent guess | The CSV file's own last-modified time is used as the reference date, captured once at ingest time and stored per-row (`reference_date`) so re-running `normalize` later doesn't shift already-ingested rows; cross-checked against the real file's own absolute dates for consistency (see Docs/context.md §11) | Degraded |
| S1b-04 | Metadata fields this source doesn't expose | No `thumbs_up`/`app_version`/`developer_reply` equivalent in a Mouthshut export | Fields present in the schema but genuinely unknown for this source | Left at their `ReviewMetadata` defaults (`0`/`None`/`None`) rather than invented or guessed | Cosmetic |
| S1b-05 | Re-ingesting the same CSV | `--refresh` passed, or CSV re-downloaded with the same content | No duplicate reviews from re-ingestion | `reviewId` is a deterministic hash of the review's own `review_url` (Mouthshut assigns no id in this export), so re-ingesting the same row always produces the same id | Cosmetic |
| S1b-06 | CSV has a header but zero data rows | Empty/truncated export | Fail loudly rather than silently producing an empty raw artifact that looks like "no Mouthshut data" (indistinguishable from S1b-01) | `MouthshutIngestError` raised naming the file | Blocker |

---

## Stage 2 — Normalize (`src/normalize.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S2-01 | Unparseable / missing date | Date null or unexpected format | Record kept, date normalized or flagged | Parse to ISO 8601; if impossible, set null + flag; don't drop the review | Degraded |
| S2-02 | Rating out of 1–5 / null | Missing or 0 rating | Rating coerced or marked unknown | Coerce int; if invalid, set null + exclude from rating-band stats | Cosmetic |
| S2-03 | Empty / whitespace-only text | Review has no textual content | Non-informative rows removed | Drop; count dropped rows in stats | Cosmetic |
| S2-04 | Extremely long review | Multi-thousand-char text | No truncation that loses meaning silently | Keep full text; note length; chunking handled later (units/embed) | Cosmetic |
| S2-05 | Duplicate text, different ids | Copy-paste spam reviews | Duplicates recognized | Optional near-dup flag by normalized text hash; keep but mark | Cosmetic |
| S2-06 | Control chars / HTML entities | `&amp;`, `\u200b`, newlines | Clean, readable text | Unescape entities; strip control chars; preserve emojis + Devanagari | Cosmetic |
| S2-07 | Developer reply present | `replyContent` set | Captured in metadata | Store `developer_reply`; may inform theme context | Cosmetic |
| S2-08 | Timezone ambiguity | Naive vs aware datetimes | Consistent timezone | Normalize to UTC ISO 8601 | Cosmetic |
| S2-09 | Merging two sources with different raw shapes (Addendum, Phase 9) | `data/raw_mouthshut.jsonl` exists alongside `raw_reviews.jsonl` | Both normalize into the *same* `Review` schema, tagged with the correct `source`, with no field silently dropped/faked | A dedicated `_normalize_one_mouthshut()` path: title+body concatenated into one `text` (Mouthshut splits them; Play Store's `content` is already single-field); dates parsed in both real shapes present in the file - absolute (`"Jun 20, 2026 05:15 PM"`) via `strptime`, relative (`"N days ago"`/`"Yesterday"`/`"Today"`) resolved against the ingest-time `reference_date` from S1b-03; unparseable dates keep the review with an empty date string, same S2-01 convention, not a drop. Verified live: 0/1,000 rejections, 0/1,000 unparseable dates on the real file | Degraded |

---

## Stage 3 — Unit Extraction (`src/units.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S3-01 | LLM unavailable / impractical at scale | No local model running, or corpus too large for per-review LLM calls | Pipeline still runs | **Decided (not just fallback): rule-based splitting is the primary/only method**, `units.use_llm` defaults to `false`. At real corpus scale (156k+ reviews, 59.5k after the S3-09 filter), per-review local LLM calls are not practical in a zero-cost/local setup; the LLM is reserved for Stage 7, which needs one call per *cluster*, not per review. If `use_llm=true` is set anyway, `units.py` logs a warning and still uses rule-based splitting (LLM path not implemented) | Degraded |
| S3-02 | LLM returns non-parseable output | Free-form text instead of list | Robust parsing | N/A while rule-based splitting is the only implemented path (see S3-01); revisit if an LLM path is added later | Cosmetic |
| S3-03 | Over-splitting | Single idea split into fragments | Units remain meaningful | `units.min_words_per_unit` (default 3) drops trivial split fragments; if splitting leaves nothing substantial, the whole review text is used as one unit instead of contributing zero units | Cosmetic |
| S3-04 | Under-splitting | Multi-topic review kept whole | Multi-topic captured | Rule-based splitter splits on sentence boundaries (`.!?;` / newline) and topic-shift conjunctions (`but`, `however`, `though`, `although`, `except`, `whereas`) | Degraded |
| S3-05 | Lost traceability | Unit missing `review_id` | Every unit traces to a review | `Unit.review_id` is required and validated by `schema.py`; `units.py` always sets it from the source `Review.id` | Blocker |
| S3-06 | Hinglish / mixed script splitting | Splitter misbehaves on Devanagari | Non-English handled | Splitter is punctuation/conjunction-based (language-agnostic); `Unit.lang` is copied from the source review's already-detected `lang` tag | Degraded |
| S3-07 | LLM hallucinates content | Unit text not present in review | Units must be faithful | N/A - rule-based splitting is purely extractive (fragments are substrings of the source text), so hallucination cannot occur by construction | Cosmetic |
| S3-08 | Explosion of units | Very long reviews → too many units | Bounded output | `units.max_units_per_review` (default 5) caps output; when exceeded, the longest (most informative) fragments are kept, not an arbitrary prefix | Cosmetic |
| S3-09 | Low-signal / non-substantive review | Review text is "good", "nice", "super app", emoji-only, etc. | Decision made explicitly with the user (not left as an unexamined default): **pure word-count cutoff, fully excluded** | **Deviates from the original "still counted in corpus stats" default.** Any review with fewer than `units.min_words` words (default 4, i.e. <=3 words is excluded) produces **zero units** and is **not referenced in any downstream artifact or persisted stat** - not `units.jsonl`, not later aggregate reports. Real corpus measurement: 96,688 / 156,219 reviews (61.9%) excluded this way, leaving 89,295 units from 59,531 contributing reviews. **Known accepted trade-off:** this is a pure length cutoff, not a generic-phrase-based filter, so it also drops some genuinely short complaints (e.g. "no delivery", "app crashed") - the user chose this over a more surgical stoplist+length combination, favoring simplicity | Degraded |
| S3-10 | Provenance lost after normalization (Addendum, Phase 9) | A second source exists (S2-09), but `Unit` originally had no `source` field | Downstream stages (8/9/UI) can still audit which source each unit came from | `Unit.source` (new field, default `"google_play"` for backward-compat with pre-addendum `units.jsonl`) is set from `review.source` at construction time in `units.py`; `_log_summary` also reports a per-source unit count | Cosmetic |

---

## Stage 4 — Embed (`src/embed.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S4-01 | Model download fails / offline | First run needs model weights | Clear guidance | Detect missing model; instruct to pre-download; cache locally after first fetch | Blocker |
| S4-02 | Text exceeds model max tokens | Long unit > model context | No silent truncation loss | Chunk + mean-pool or truncate with note; document choice | Degraded |
| S4-03 | Row/index misalignment | `embeddings.npy` rows != units | Strict alignment | Persist `unit_index.json` mapping row→unit id; assert counts match | Blocker |
| S4-04 | Non-normalized vectors | Cosine assumed but vectors unnormalized | Correct similarity | L2-normalize on save; document metric | Cosmetic |
| S4-05 | OOM on large corpus | Too many units in one batch | Stable memory | Batch encoding; stream to disk; configurable batch size | Degraded |
| S4-06 | Empty units file | No units to embed | Stops gracefully | Guard (see X-10); exit with message | Blocker |
| S4-07 | Mixed-language embedding quality | MiniLM weak on Hinglish | Quality flagged | Offer multilingual model option in config; report per-lang counts | Degraded |
| S4-08 | Determinism across machines | Float differences CPU/GPU | Acceptable reproducibility | Fix seed; pin model version; note minor float variance is expected | Cosmetic |

---

## Stage 5 — Similarity Graph (`src/graph.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S5-01 | k too large for node count | k >= number of units | Valid graph | Clamp k to n-1; warn | Cosmetic |
| S5-02 | Threshold too high → no edges | Similarity cutoff excludes all | Graph not empty | Detect empty edge set; auto-relax or error with guidance | Degraded |
| S5-03 | Threshold too low → hairball | Everything connected | Communities still meaningful | Cap edges per node (top-k); expose threshold in config | Degraded |
| S5-04 | Isolated nodes | Unique units with no neighbors | Not dropped silently | Keep as singletons; report count; handled as own micro-community later | Cosmetic |
| S5-05 | faiss unavailable | `faiss-cpu` not installed | Still builds graph | Fall back to `scikit-learn` NearestNeighbors | Degraded |
| S5-06 | Duplicate embeddings | Identical units → sim 1.0 | No divide-by-zero / degenerate edges | Handle exact dups (collapse or keep with weight 1.0 cap) | Cosmetic |
| S5-07 | Memory blow-up on dense kNN | O(n^2) on huge corpus | Scales | Use approximate/ANN or batched kNN; document limits | Degraded |

---

## Stage 6 — Community Detection (`src/cluster.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S6-01 | Louvain non-determinism | Random init → different partitions | Stable-ish clusters | Fixed random seed; document residual variance | Degraded |
| S6-02 | One giant community | Resolution too low | Useful granularity | Tune resolution (config); optionally recurse/split large communities | Degraded |
| S6-03 | Too many micro-communities | Resolution too high / sparse graph | Manageable theme count | **Revised:** flag (`below_min_size: true`), don't physically drop, communities under `clustering.min_community_size` in `communities.json` (keeps `cluster.py`'s output lossless for Stage 9 validation). `src/summarize.py` (Stage 7) then processes all of them in one dedicated batched pass (not per-community theme summarization) and surfaces them as `emerging_signals` - plain dicts tagged with `support_count` + a `confidence` tier, kept separate from `themes` in `themes.json` rather than forced into the `Theme`/`QuestionInsight` schemas | Degraded |
| S6-04 | Disconnected components | Graph in pieces | Each component clustered | Louvain handles per-component; report component count | Cosmetic |
| S6-05 | Singletons from S5-04 | Isolated nodes | Represented, not dropped | Assign own community; flag as low-support | Cosmetic |
| S6-06 | Empty graph | No edges (see S5-02) | Stops gracefully | Guard; exit with actionable message | Blocker |

---

## Stage 7 — LLM Summarization (`src/summarize.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S7-01 | LLM unavailable | Groq down | Degraded summaries | **Decided (not just fallback): the TF-IDF/rating-derived extractive method is the primary path** (`summarize.use_llm` defaults to `false`); Groq was verified live to be unreachable on this machine, so this is also what actually runs, not a theoretical fallback. Label = top TF-IDF terms; description = templated from community stats; sentiment = rating distribution (never LLM-derived, S7-06). If `use_llm=true` is set and a call fails, degrades to this same extractive result per-community/per-batch and logs a warning | Degraded |
| S7-02 | Hallucinated theme label | Label not supported by members | Faithful labels | Feed only representative member texts; require quotes to be verbatim from members | Degraded |
| S7-03 | Non-JSON LLM output | Free-form summary | Parseable themes | Constrain to JSON schema; retry once; fallback on repeated failure | Degraded |
| S7-04 | Community too large to fit context | Thousands of members | Bounded prompt | Select representatives (medoid/centrality), cap count; summarize sample | Degraded |
| S7-05 | Verbatim not from member | Quote fabricated | Quotes are real | Validate each quote exists in a member unit; drop invalid quotes | Degraded |
| S7-06 | Sentiment mislabel | LLM guesses sentiment | Reasonable sentiment | Cross-check with rating distribution of members as a sanity signal | Cosmetic |
| S7-07 | Duplicate/near-identical themes | Two communities same topic | Deduped view | Optional theme-merge by label/embedding similarity; note merges | Cosmetic |
| S7-08 | Mixed-language members | Hinglish cluster | Coherent English label | Prompt to label in English while quoting original verbatims | Cosmetic |

---

## Stage 8 — Insight Mapping (`src/insights.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S8-01 | Question with no supporting theme | No cluster matches a research question | Gap made explicit | **Decided:** report the question as `coverage: "insufficient"` rather than fabricating; confirmed on the real corpus - 4/8 questions (Q2, Q5, Q7, Q8) are genuinely insufficient at `insights.similarity_threshold=0.30`, an honest finding (this Blinkit review corpus skews heavily operational, not discovery-behavior-focused) not a bug to paper over. `emerging_signals`-only support never flips this to sufficient (see S8-06 revision) | Degraded |
| S8-02 | Theme maps to no question | Off-topic/noise cluster | Nothing dropped silently | **Decided:** bucket under `insights.json`'s `"uncategorized"` key (2/5 themes, 780/819 signals on the real corpus); still fully present in `themes.json`, just not force-mapped | Cosmetic |
| S8-03 | Theme maps to many questions | Broad theme spans questions | Multi-mapping allowed | **Decided:** allow 1..n mapping (any question whose similarity clears `insights.similarity_threshold`); the per-question `theme_similarities` dict records the actual score per mapping instead of a separate confidence label | Cosmetic |
| S8-04 | Operational/pricing complaints filtered | Instinct to drop non-discovery themes | Whole-funnel preserved | Explicitly retain + categorize (stockouts, price, delivery) per spec §3 | Blocker |
| S8-05 | Counts double-counted | Unit in overlapping aggregates | Accurate frequencies | Define counting unit clearly (units vs reviews); dedupe in aggregates | Degraded |
| S8-06 | Sentiment/segment fields missing | Upstream didn't populate | Robust aggregation | Treat missing as unknown bucket; don't crash aggregation. **Extended - a real correctness bug found on the real corpus:** embedding similarity alone matches themes to questions by shared topic *nouns* regardless of polarity - a clearly positive theme ("app / delivery / fast / best") scored 0.575 against the "frustrations" question, higher than most genuinely negative themes. Fixed by adding `insights.question_required_sentiment` (currently only Q6 gated to `"negative"`), cross-checked against the theme's/signal's own rating-derived sentiment - never an LLM guess, same ground-truth-from-ratings principle as S7-06. This dropped Q6 from 34 to 11 themes, all genuinely negative | Degraded |
| S8-07 | Competitor mentions (Zepto/Instamart) | Contrastive references | Used for contrast, not confused with Blinkit | `theme-0020` ("zepto / better / instamart / flipkart") surfaced naturally from Louvain and is left as its own theme (bottom-up clustering, not filtered) rather than specially tagged - future work if competitor-contrast needs its own reporting section | Cosmetic |
| S8-08 | Mapping non-determinism | LLM classification varies | Reproducible-ish mapping | **Decided (not just a cache/temperature mitigation): mapping is embedding similarity against a fixed, documented `insights.question_queries` list, not an LLM call at all** - fully deterministic given the same embedding model and config, consistent with S7-01's decision that the no-LLM path is primary, not a fallback | Degraded |

---

## Stage 9 — Validation (`src/validate.py`)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S9-01 | Single rating band populated | All reviews 5-star | Triangulation still reported | `corpus_rating_bands_observed` in `validation.json` reports every band actually present across the real corpus; not an issue on the real data (all 5 bands present) but the field degrades gracefully to whatever subset exists | Degraded |
| S9-02 | Single time cohort | Reviews clustered in one period | Time triangulation limited | `corpus_time_cohorts_observed` reports every month actually present (5 months on the real corpus, matching the 4-month lookback window); a theme concentrated in one cohort is flagged `"segment_specific"` rather than failing | Degraded |
| S9-03 | Coherence undefined for singletons | Community size 1 | No divide-by-zero | `_coherence()` skips (returns `null` fields, not a crash) any theme with < 2 members with embeddings; doesn't occur among the real 5 qualifying themes (all size >= 40) but guarded defensively | Cosmetic |
| S9-04 | No manual labels provided | Spot-check sample unlabeled | Still produces sample | **Decided:** export once to `data/spot_check_sample.json` (192 rows, 5/theme) with `human_agrees: null`; **never regenerated automatically** once the file exists, so manual edits survive `--refresh` - verified live by hand-editing a row, re-running, and confirming both the edit persisted and `agreement_rate` computed correctly from it | Cosmetic |
| S9-05 | Over-claiming stability | Theme in one segment only | Honest reporting | **Decided:** label every theme `"cross_segment"` or `"segment_specific"` based on time-cohort/review-length concentration (`validation.dominant_share_threshold=0.6`) - real result: 4/5 cross-segment, 1/5 segment-specific. **Rating bands are deliberately excluded from this judgment** - a sentiment-driven theme skewing one rating band is an expected, healthy signal (e.g. a complaints theme *should* be mostly 1-star), not evidence of instability; still reported per-theme, just not judged. **Extended, Phase 9 addendum:** the new per-theme `source_distribution` (S2-09/S3-10's provenance) is reported the same way, for the same reason - Mouthshut is ~4.4% of the merged unit corpus, so near-universal `google_play` dominance per theme is an expected size-imbalance artifact, not instability; folding it into the stability verdict would trivially flag almost every theme "segment_specific" on that axis alone, which would be noise, not signal | Degraded |
| S9-06 | Metric misinterpretation | Silhouette on graph clusters | Metric appropriate to method | **Decided:** report graph **modularity of the actual Louvain partition** (0.8394 on the real corpus) as the primary, methodologically-correct metric for a graph-clustering method, alongside a clearly-labeled centroid-based "silhouette-style" score (mean -0.001) computed in embedding space - not literal silhouette (infeasible at up to 8,609 members/theme). The near-zero mean despite very high modularity is documented as a real, informative finding (the graph resolves finer sub-topic distinctions via precise unit-to-unit edges than a coarse centroid average can see - several themes share surface vocabulary like "delivery"/"service" but were still correctly graph-separated), not a bug | Cosmetic |
| S9-07 | Empty themes to validate | Upstream produced none | Stops gracefully | `validate_pipeline()` raises `ValidateError` with an actionable message if `themes.json` has zero themes, before any computation starts | Blocker |

---

## Stage 10 — UI / Presentation (`app.py` / notebook)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| S10-01 | Artifacts missing at load | Run UI before pipeline | Friendly guidance | Detect missing `insights/themes/validation`; show which stage to run | Blocker |
| S10-02 | Very large theme list | Hundreds of themes | UI stays usable | Pagination/search/filter by question, rating, sentiment | Degraded |
| S10-03 | Long verbatims break layout | Multi-paragraph quotes | Readable display | Truncate with expand; wrap text | Cosmetic |
| S10-04 | Emoji/Devanagari rendering | Non-ASCII in UI | Renders correctly | UTF-8 throughout; font-safe rendering | Cosmetic |
| S10-05 | Question with no evidence (S8-01) | Gap in coverage | Shown honestly | Display "insufficient evidence" state, not blank/fake answer | Degraded |
| S10-06 | Shareable link requirement | Needs external access | Presentable output | Streamlit sharing / notebook export; document how to host | Degraded |
| S10-07 | Stale artifacts | UI shows old run | Freshness visible | Show artifact timestamps / run id | Cosmetic |

---

## Reproducibility / Environment (Phase 7)

| ID | Edge Case | Trigger | Expected behavior | Handling | Severity |
|---|---|---|---|---|---|
| R-01 | Fresh env missing Groq | No local LLM installed | Documented + degradable | README install steps; pipeline runs with fallbacks if absent | Degraded |
| R-02 | Dependency version drift | Unpinned deps update + break | Stable installs | Pin all versions in `requirements.txt` | Blocker |
| R-03 | OS path differences (Windows) | Backslash vs POSIX paths | Cross-platform | Use `pathlib`; no hard-coded separators | Cosmetic |
| R-04 | Partial pipeline re-run | Stage rebuilt, downstream stale | Consistent state | `--force` cascades; detect artifact staleness by upstream timestamp | Degraded |
| R-05 | Large `data/` committed | Artifacts pushed to git | Repo stays clean | `.gitignore` `data/`; document regeneration | Cosmetic |
| R-06 | Seeds not set | Non-reproducible run | Reproducible-as-possible | Central seed in config applied to numpy/sklearn/louvain/LLM | Degraded |
| R-07 | Windows torch DLL init failure | `torch` installed in a different site-packages root than `numpy`/`transformers` (mixing `--user` and system-wide installs), or `transformers`/`sentence-transformers` imports `torch` transitively before any code imports it directly | Confusing `OSError: [WinError 1114] ... c10.dll` even though `pip install` reported success | `requirements.txt` pins a single consistent install (`--extra-index-url` CPU wheels, one `pip install -r requirements.txt` run); `src/embed.py` additionally imports `torch` directly before `sentence_transformers` to avoid the transitive-import DLL conflict (hit and fixed live during Phase 2) | Blocker |
| R-08 | `networkx` removed `write_gpickle`/`read_gpickle` in 3.0 | Pinned `networkx==3.3` has no gpickle API despite `graph.gpickle` being the artifact name throughout the docs | `AttributeError` if code assumes the old API | `src/graph.py` uses plain `pickle.dump`/`pickle.load` on the `nx.Graph` object instead (networkx's own documented replacement); the `.gpickle` filename is kept purely as a naming convention, not a format guarantee | Cosmetic |
