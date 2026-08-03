Read problemstatement.md, architecture.md, and categorization-logic.md first.

PROBLEM: The current pipeline clusters the full review corpus, so high-volume
operational themes (delivery, fees, support) dominate the output and bury the
in-app UX friction themes that a product MVP can actually solve.

IMPLEMENT THE FOLLOWING CHANGES — do not modify the existing embed → kNN →
Louvain → LLM-summarize pipeline itself, only add stages around it:

1. NEW STAGE — Addressability classifier (runs BEFORE embedding/clustering):
   Classify every review (or complaint unit) into exactly one label using the
   free/local LLM with this prompt:

   ---
   Classify this Blinkit app review into exactly one label:
   - app_ux: friction with the app interface itself (search results,
     out-of-stock handling, recommendations and personalization — repetitive
     or irrelevant suggestions, recently-viewed items shown instead of new
     products, ads crowding results — category/browse navigation, missing
     product information like expiry or reviews, cart, checkout, bugs,
     crashes, notifications/spam)
   - operational: delivery speed, delivery partners, stockouts,
     missing/wrong/damaged items, refunds, customer support
   - pricing_policy: fees, handling/surge charges, thresholds, prices
   - praise_noise: generic praise or abuse with no specific issue
     ("best app", "worst app ever")
   Return ONLY JSON: {"label": "...", "reason": "<one short phrase>"}
   Review: {review_text}
   ---

   Store the label on each review record. Never invent labels outside these
   four. To keep runtime manageable on a local model, add a cheap keyword/
   heuristic pre-filter that auto-labels obvious praise_noise (very short,
   pure sentiment, no specific issue) and obvious operational reviews
   (delivery/rider/refund vocabulary), sending only the uncertain remainder
   to the LLM.

2. RE-CLUSTER ON THE app_ux SUBSET ONLY:
   Run the existing clustering pipeline on reviews labeled app_ux alone.
   Keep the original full-corpus run and its outputs intact — do not delete
   or overwrite them. The app_ux run is a new, separate output.

3. ADD journey_stage TO CLUSTER SUMMARIZATION:
   In the existing cluster-summary prompt, add one required output field:
   "journey_stage": one of "search" | "browse_discover" | "recommendations" |
   "product_page" | "cart_checkout" | "post_order".
   Update the output JSON schema accordingly.

4. REPORTING:
   - Page-2 narrative now leads with app_ux themes ranked by review count,
     grouped by journey_stage (a friction map of the app journey).
   - Keep operational and pricing_policy as context: report their total
     counts and top themes in a clearly separated "outside product scope"
     section. Do not silently drop them.
   - Report the praise_noise count as excluded noise.

5. VALIDATION:
   - Randomly sample 50 classified reviews across all four labels into a
     CSV (review_text, assigned_label, reason) for manual spot-checking.
   - Add classifier distribution stats (count and % per label) to the
     validation report.

CONSTRAINTS: free models only, no paid APIs. Every theme count must remain traceable: label → cluster_id → review_ids → verbatims.

Before writing code, list any ambiguities in this spec rather than guessing.