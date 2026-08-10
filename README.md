# Insurtech & Digital Risk Solutions — Integrated Synthetic Insurance Dataset

Synthetic insurance dataset used throughout the **Insurtech & Digital Risk Solutions** MBA course (School of Business, Woxsen University). Created with a fixed random seed so every user generates the **identical** dataset.

## Files

| File | Description |
|------|-------------|
| `customers.csv` | 5,000 customers — demographics, location, occupation, income, credit score, KYC status |
| `policies.csv` | 10,000 policies — type, premium, sum assured, tenure, dates, channel |
| `claims.csv` | 20,000 claims — amount, type, status, fraud flag, days-to-settle |
| `fraud_indicators.csv` | 5,000 indicators linked to claims |
| `cyber_incidents.csv` | 2,000 cyber incidents — breach type, loss, ransomware flag |
| `weather_data.csv` | 3,000 daily records — rainfall, temperature, cyclone, flood index |
| `generate_dataset.py` | The generator script — regenerates all six CSVs (`python generate_dataset.py`) |
| `data_analysis.py` | The cleaning & analysis pipeline used in Session 05 |

## How to use

1. **Download** the six CSV files into a folder named `data/` (or clone this repo).
2. Run the Session 05 exercises in a Jupyter Notebook with `DATA_PATH = 'data/'`.
3. To regenerate the CSVs from scratch: `python generate_dataset.py` (seed = 42, byte-identical output).

## Relationships

```
customers.customer_id  ←  policies.customer_id  ←  claims.customer_id
                            policies.policy_id  ←  claims.policy_id
                                                 ←  fraud_indicators.claim_id
```

## Notes

- Run the generator **once** — do not regenerate mid-course or keys will not match earlier exercises.
- Contains embedded patterns for the course: ~5% fraud flag, missing income (MAR), cyclone events in weather data, and realistic right-skewed claim distributions.

---
*Part of the Insurtech & Digital Risk Solutions MBA course — School of Business, Woxsen University.*
