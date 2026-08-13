# Session 09 — Digital Customer Acquisition

Scripts and data for Session 09 of *InsurTech & Digital Risk Solutions* (MBA, Woxsen University). Companion to the session page's §5–§6 Python blocks.

## Files

| File | Description |
|---|---|
| `funnel_data.csv` | Session-level funnel events — 10,000 user sessions, one row per session with 1/0 flags for each funnel stage (`stage_awareness` … `stage_onboarding`) plus `channel` and `product`. Deterministic (seed 42); the funnel profile mirrors the taught pattern (biggest leak at Consideration → Quote, ~72%). |
| `session_09_digital_acquisition.py` | **Complete runnable script** merging the page's §5.1–§6.2 blocks so nothing is left undefined: funnel loading → conversion rates → funnel chart → channel-level CAC/LTV/LTV-CAC → budget allocation → bubble chart. |
| `chapter9_funnel.png` | Expected output — digital insurance acquisition funnel bar chart. |
| `channel_economics_bubble.png` | Expected output — CAC vs LTV bubble chart (bubble size = volume, colour = LTV/CAC). |

## How to run

```bash
# from the repo root (or wherever this folder lives)
python3 -m venv venv
venv/bin/pip install -r requirements.txt   # numpy, pandas, matplotlib
venv/bin/python session-09/session_09_digital_acquisition.py
```

The script reads `data/funnel_data.csv` (relative to the working folder) and falls back to `funnel_data.csv` in the current directory. It prints the funnel table, channel economics table, and budget recommendations, and saves `chapter9_funnel.png` + `channel_economics_bubble.png`.

## Expected console output (abridged)

```
Total sessions tracked: 10,000
Stage-wise user counts:
  stage_awareness          : 10,000
  stage_consideration      :  4,257
  stage_quote              :  1,167
  stage_purchase           :    516
  stage_onboarding         :    415
Overall conversion (Awareness → Onboarding): 4.2%
Biggest drop-off: Consideration → Quote (72.6%)

Channel                 CAC (₹)    LTV (₹)  LTV/CAC   Volume Rating
referral             ₹    250  ₹  18,773   75.1x    200  EXCELLENT
organic_search       ₹    500  ₹  11,900   23.8x    400  EXCELLENT
embedded_partner     ₹    375  ₹   1,870    5.0x    800  GOOD
aggregator           ₹  1,333  ₹   4,463    3.3x    600  GOOD
paid_search          ₹  3,333  ₹   5,571    1.7x    450  MARGINAL
social_media         ₹  3,429  ₹   3,150    0.9x    350  POOR (FIX OR KILL)
```

Data generated with `np.random.default_rng(42)` — reproducible across runs.
