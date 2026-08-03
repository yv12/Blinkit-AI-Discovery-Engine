"""Stage: Barrier Mapping (Layer 2)

Applies a fixed barrier lens over the open topic clusters (Layer 1) to understand
why users do not explore new categories.

Design choices:
1. Samples 15-25 reviews per cluster spanning the full rating range.
2. Uses the fixed barrier taxonomy; never invents labels.
3. Groups clusters by primary_barrier, sums member counts, and bubbles up quotes.
4. Uses Groq LLM (via _call_groq).
"""

import json
import logging
import random
from collections import defaultdict
from typing import Any, Dict, List

from src.config import Config, ensure_data_dir, load_config
from src.llm_synthesis import _call_groq, _load_api_key, _RateLimiter
from src.schema import Unit, read_json, read_jsonl, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FIXED_BARRIERS = {
    "trust_risk",
    "economic",
    "reliability",
    "discovery",
    "recovery",
    "habit_load",
    "out_of_scope"
}

PROMPT_TEMPLATE = """You are analyzing clusters of Blinkit (Indian quick-commerce app) Play Store
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
"""

def _sample_reviews(unit_ids: List[str], unit_meta: Dict[str, Unit], sample_size: int = 20, seed: int = 42) -> List[Unit]:
    """Sample reviews to span the rating range."""
    rng = random.Random(seed)
    # Group units by rating
    by_rating: Dict[int, List[Unit]] = defaultdict(list)
    for uid in unit_ids:
        if uid in unit_meta:
            u = unit_meta[uid]
            r = u.rating if u.rating is not None else -1
            by_rating[r].append(u)
    
    # Stratified sampling
    selected: List[Unit] = []
    ratings = sorted(by_rating.keys())
    if not ratings:
        return selected

    budget = min(sample_size, sum(len(units) for units in by_rating.values()))
    
    while len(selected) < budget:
        for r in ratings:
            if by_rating[r]:
                selected.append(by_rating[r].pop(rng.randrange(len(by_rating[r]))))
            if len(selected) >= budget:
                break
                
    return selected

def map_barriers(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    output_path = config.paths.data_dir / "barrier_mapping.json"
    
    if output_path.exists() and not refresh:
        logger.info("'%s' already exists; skipping barrier mapping (pass --refresh to rebuild).", output_path)
        return

    logger.info("Loading communities and units...")
    communities_doc = read_json(config.paths.communities)
    communities = communities_doc.get("communities", [])
    
    # Filter to only communities that are not below min size
    valid_communities = [c for c in communities if not c.get("below_min_size", False)]
    logger.info("Found %d valid communities for mapping.", len(valid_communities))

    unit_meta = {u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)}
    api_key = _load_api_key()
    limiter = _RateLimiter(requests_per_minute=25)

    mapped_clusters = []
    
    for c in valid_communities:
        cluster_id = f"c_{c['community_id']:03d}"
        logger.info("Processing cluster %s (size %d)...", cluster_id, c["size"])
        
        sampled_units = _sample_reviews(c["unit_ids"], unit_meta, sample_size=25, seed=config.seed + c["community_id"])
        
        # Format for prompt
        reviews_text = "\n".join(f"[{u.review_id}] (Rating: {u.rating if u.rating else 'N/A'}) {u.text}" for u in sampled_units)
        prompt = PROMPT_TEMPLATE.replace("{sampled_reviews_with_ids}", reviews_text)
        
        # _call_groq handles retries and JSON parsing internally
        data = _call_groq(
            prompt=prompt,
            model=config.llm_synthesis.model,
            api_key=api_key,
            temperature=0.0,
            seed=config.seed,
            max_retries=config.llm_synthesis.max_retries,
            limiter=limiter,
        )
        
        if not data:
            logger.error("Cluster %s failed completely.", cluster_id)
            continue
            
        # Check for primary_barrier guardrail
        pb = data.get("primary_barrier", "out_of_scope")
        if pb not in FIXED_BARRIERS:
            logger.warning("Cluster %s: LLM invented label '%s'. Forcing to out_of_scope.", cluster_id, pb)
            data["primary_barrier"] = "out_of_scope"
        
        # Filter secondary barriers
        sb = data.get("secondary_barriers", [])
        if isinstance(sb, list):
            data["secondary_barriers"] = [b for b in sb if b in FIXED_BARRIERS]
        else:
            data["secondary_barriers"] = []
        
        # Add cluster metadata
        data["cluster_id"] = cluster_id
        data["member_count"] = c["size"]
        mapped_clusters.append(data)

    # Aggregation Layer
    logger.info("Aggregating clusters by primary barrier...")
    aggregated = {b: {"barrier": b, "total_count": 0, "sub_themes": [], "all_quotes": []} for b in FIXED_BARRIERS}
    
    for mc in mapped_clusters:
        pb = mc["primary_barrier"]
        
        aggregated[pb]["total_count"] += mc["member_count"]
        aggregated[pb]["sub_themes"].append({
            "cluster_id": mc["cluster_id"],
            "theme_name": mc["theme_name"],
            "summary": mc["summary"],
            "member_count": mc["member_count"],
            "confidence": mc.get("confidence", "low")
        })
        
        # Carry quotes up
        quotes = mc.get("representative_quotes", [])
        for q in quotes:
            aggregated[pb]["all_quotes"].append({
                "cluster_id": mc["cluster_id"],
                "review_id": q.get("review_id", ""),
                "text": q.get("text", "")
            })
            
    # Sort and clean up output
    final_output = []
    for b in FIXED_BARRIERS:
        if aggregated[b]["total_count"] > 0 or b == "out_of_scope":
            # Sort sub_themes by size
            aggregated[b]["sub_themes"].sort(key=lambda x: x["member_count"], reverse=True)
            final_output.append(aggregated[b])
            
    final_output.sort(key=lambda x: x["total_count"], reverse=True)
    
    write_json(output_path, {"barrier_aggregation": final_output})
    logger.info("Wrote barrier mapping to %s", output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Force rebuild")
    args = parser.parse_args()
    map_barriers(load_config(), refresh=args.refresh)
