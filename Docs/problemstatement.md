# Problem Statement: AI-Powered Discovery Engine for Blinkit Category Exploration

> **Purpose of this file:** This is the base specification for a coding agent. Read this fully before generating any code, plans, or other project files. All architecture decisions, data-source choices, and analysis methods must trace back to the requirements and constraints defined here.

---

## 1. Business Context

**Role being simulated:** Product Manager on the Growth Team at **Blinkit** (Indian quick-commerce platform).

Quick-commerce platforms have become part of users' weekly routines. Many users place recurring orders for groceries, snacks & beverages, and household essentials. Over time, shopping behavior becomes highly repetitive: users purchase the same set of products repeatedly and rarely explore new categories available on the platform.

**Strategic goal:**

> Increase the percentage of Monthly Active Customers (MAC) who purchase products from **at least one new category every month**.

Examples of the target behavior:
- A user who buys groceries starts buying pet supplies.
- A user who buys snacks starts buying personal care products.
- A user who buys household essentials starts buying baby products.

---

## 2. What Must Be Built

Before any product solution is proposed, an **AI-powered discovery engine** must be built that gathers and analyzes user feedback **at scale** and surfaces insights about category-exploration behavior.

The original assignment lists many possible sources (App Store, Play Store, Reddit, forums, social media). **For this implementation, the corpus is Google Play Store reviews only** (see Section 6) — the analysis must still be at scale and answer all research questions from that corpus.

Any AI-native stack is permitted (LLMs, agents, workflows, RAG, n8n, Zapier, etc.) — but see the hard constraints in Section 5.

---

## 3. Research Questions (Decision Criteria)

The engine's output must be able to answer these 8 questions. **Every analysis-method decision should be evaluated against whether it helps answer them:**

1. Why do users repeatedly buy from the same categories?
2. What prevents users from exploring new categories?
3. How do users discover products today?
4. What role do habits play in shopping behavior?
5. What information do users need before trying a new category?
6. What frustrations emerge repeatedly?
7. Which user segments are more likely to experiment?
8. What unmet needs emerge consistently across discussions?

**Scope note:** The engine diagnoses across the **whole funnel** — discovery, engagement, add-to-cart, conversion — not just top-of-funnel discovery. Reviews naturally surface operational and pricing complaints (stockouts, price, delivery time) alongside pure discovery complaints; these must be captured and categorized, not filtered out.

---

## 4. Demonstration Requirements

The final workflow must clearly demonstrate:

1. **How data is gathered and analyzed** — the pipeline from raw source to structured corpus.
2. **How themes are identified** — the clustering/grouping methodology.
3. **How insights are generated** — how clusters become answers to the 8 research questions.
4. **How insight quality was validated** — e.g., sample-based spot checks, cluster coherence metrics, cross-source triangulation, comparison against a manually labeled subset.

Deliverables the engine feeds into (context, not code targets):
- A shareable/testable link to the review-analysis workflow.
- A 1-slide explainer of the workflow inside a 10-slide PDF deck.
- Findings later validated via 5–6 user interviews (outside this codebase's scope).

---

## 5. Hard Constraints

- **Zero cost:** No paid APIs and no paid models at any stage — data gathering or processing. Fully free/local stack only (local embeddings, local/free-tier LLMs, free scraping libraries).
- **Code-only data collection:** All data must be gathered programmatically. No manual copy-paste collection.
- **Single data source:** Only Google Play Store reviews are in scope. Reddit, Apple App Store, YouTube, Quora, deal forums, complaint boards, X/Twitter, and Instagram are all explicitly out of scope. Do not build scrapers or connectors for any other source.
- **Target app:** Blinkit (competitor mentions like Zepto/Instamart may appear in data and are useful for contrast, but Blinkit is the subject).

---

## 6. Data Source

**Google Play Store reviews of the Blinkit app — the only data source for this project.**

- Library: `google-play-scraper` (Python, free, no API key required).
- Target app ID: `com.grofers.customerapp` (Blinkit on Play Store — verify at build time).
- Pull a large corpus: maximize volume across ratings (1–5 stars), sorted by newest and by relevance, in English and Hindi/Hinglish where available (`lang='en'`, `country='in'`).
- Capture full metadata per review: rating, date, thumbs-up count, app version, and developer reply if present.
- Since this is a single-source pipeline, the "cross-source triangulation" validation step becomes **cross-segment triangulation**: check whether themes hold across rating bands, time periods, and review length/recency cohorts.

---

## 7. Proposed Analysis Approach (Graph-Based Theme Discovery)

The intended methodology — refine details as needed, but preserve the shape:

1. **Ingest & normalize** all scraped items (review/comment/post) into a common schema: `{id, source, text, rating?, date, url?, metadata}`.
2. **Extract complaint/insight units** — split multi-topic reviews into atomic statements where needed.
3. **Embed** each unit with a free/local embedding model.
4. **Build a similarity graph:** each unit is a node; kNN over embeddings creates `SIMILAR` edges.
5. **Community detection** (Louvain) groups nodes into theme communities.
6. **LLM summarization** (free/local model): summarize each community into a one-sentence theme/category node, with representative quotes and counts.
7. **Category-level graph:** create category–category similarity edges from averaged member-complaint similarities, enabling a navigable insight map.
8. **Insight layer:** map themes to the 8 research questions; quantify frequency, source spread, and sentiment per theme.
9. **Validation layer:** cluster coherence checks, manual spot-check sample, cross-segment triangulation report (rating bands, time cohorts).

---

## 8. Definition of Done

- Pipeline runs end-to-end from scraping to a queryable/browsable insight output on a free/local stack.
- Each of the 8 research questions has a supported, evidence-backed answer (theme names, counts, representative verbatims, sources).
- Theme identification and validation methodology are documented and demonstrable.
- Output is presentable via a link (hosted workflow, notebook, or lightweight app) suitable for evaluation.
- Reproducible: a fresh environment can install dependencies and run the pipeline with documented steps.

---

## 9. What the Coding Agent Should Generate Next

From this spec, generate (in order):
1. A project plan / architecture doc (`architecture.md`) mapping the pipeline stages to modules.
2. The Play Store scraper module with the normalized output schema.
3. The embedding + graph + clustering pipeline.
4. The LLM summarization + insight-mapping layer.
5. The validation report generator.
6. A minimal UI or notebook to browse themes and answer the 8 questions.
