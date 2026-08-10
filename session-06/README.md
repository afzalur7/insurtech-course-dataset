# Session 06 — Data Visualization for Insurance

Complete runnable script for every chart in Session 06 of the **Insurtech & Digital Risk Solutions** course.

## File

| File | Description |
|------|-------------|
| `visualization_analysis.py` | The complete session script — every chart, organized by section banners |

## Requirements

- `data/insurance_cleaned.csv` — the cleaned dataset produced by Session 05's `data_analysis.py` (in the repository root)
- Python with pandas, numpy, matplotlib, seaborn

## Section map (HTML section → script section)

| Session page section | Script banner |
|----------------------|---------------|
| 2. Matplotlib Foundations | `SECTION 2: MATPLOTLIB FOUNDATIONS` |
| 3. Seaborn Statistical Plots | `SECTION 3: SEABORN STATISTICAL PLOTS` |
| 4. Claims Trend Analysis | `SECTION 4: CLAIMS TREND ANALYSIS` |
| 5. Portfolio Composition | `SECTION 5: PORTFOLIO COMPOSITION` |
| 6. Distribution & Outliers | `SECTION 6: DISTRIBUTION & OUTLIER ANALYSIS` |
| 7. Correlation Analysis | `SECTION 7: CORRELATION & RELATIONSHIP ANALYSIS` |
| 8. Publication-Ready Charts | `SECTION 8: PUBLICATION-READY CHARTS` |
| Hands-On Project (6-chart board brief) | `HANDS-ON: THE 6-CHART BOARD BRIEF` |

## Setup (important — avoids environment errors)

Your machine may have conflicting system packages (system matplotlib/scipy built for
NumPy 1.x while pip installed NumPy 2.x). The reliable fix is a **virtual environment**:

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python visualization_analysis.py
```

## Running

```bash
python visualization_analysis.py        # or open in Jupyter Notebook
```

- **In Jupyter:** every chart displays inline.
- **As a plain script:** every chart is saved into `./charts/` (e.g. `chart_01.png`).



```bash
python visualization_analysis.py        # or open in Jupyter Notebook
```

- **In Jupyter:** every chart displays inline.
- **As a plain script:** every chart is saved into `./charts/` (e.g. `chart_01.png`).



Each chart carries the session's manager's lens — for every plot ask:
**WHAT does it tell us? / SO WHAT does it matter? / NOW WHAT should we do?**
