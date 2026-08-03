# Validation Summary (Stage 9)

Themes validated: 15
Graph modularity (Louvain partition quality): 0.7519
Mean theme silhouette-style score (centroid-based, [-1, 1]): -0.121
Cross-segment stable themes: 12 / 15
Segment-specific themes: 3 / 15
Spot-check labels collected so far: 0
Spot-check agreement rate: None

## Coherence (sorted by silhouette-style score, ascending = least coherent first)

| theme_id | label | size | intra_sim | nearest_theme | nearest_sim | silhouette |
|---|---|---|---|---|---|---|
| theme-0010 | items / missing / order / ordered | 708 | 0.625 | theme-0003 | 0.805 | -0.223 |
| theme-0003 | product / products / quality / poor | 1104 | 0.633 | theme-0010 | 0.805 | -0.214 |
| theme-0008 | app / delivery / apps / price | 781 | 0.642 | theme-0001 | 0.806 | -0.204 |
| theme-0002 | delivery / service / bad / partner | 1258 | 0.691 | theme-0007 | 0.864 | -0.201 |
| theme-0001 | app / delivery / order / worst | 1424 | 0.677 | theme-0005 | 0.845 | -0.199 |
| theme-0000 | customer / service / support / care | 1658 | 0.669 | theme-0011 | 0.81 | -0.174 |
| theme-0009 | refund / order / product / return | 719 | 0.664 | theme-0010 | 0.786 | -0.156 |
| theme-0005 | app / worst / bad / use | 1029 | 0.724 | theme-0001 | 0.845 | -0.143 |
| theme-0007 | delivery / time / late / order | 911 | 0.75 | theme-0002 | 0.864 | -0.133 |
| theme-0011 | service / poor / worst / bad | 629 | 0.708 | theme-0000 | 0.81 | -0.126 |
| theme-0012 | delivery / cash / app / option | 176 | 0.731 | theme-0001 | 0.81 | -0.098 |
| theme-0014 | app / groceries / grocery / worst | 133 | 0.687 | theme-0001 | 0.754 | -0.089 |
| theme-0004 | delivery / charges / high / charge | 1043 | 0.73 | theme-0002 | 0.752 | -0.028 |
| theme-0013 | hai / app / delivery / service | 140 | 0.736 | theme-0001 | 0.692 | 0.06 |
| theme-0006 | blinkit / delivery / app / service | 993 | 0.752 | theme-0001 | 0.667 | 0.113 |

## Segment-specific themes (concentrated in one time cohort or length bucket)

- **theme-0006** (blinkit / delivery / app / service): concentrated in a single review-length bucket
- **theme-0012** (delivery / cash / app / option): concentrated in a single review-length bucket
- **theme-0013** (hai / app / delivery / service): concentrated in a single review-length bucket

## Spot-check

No human labels present yet - open data/spot_check_sample.json, fill in `human_agrees` (true/false) per row, then re-run `python -m src.validate --refresh` to compute agreement (S9-04).
