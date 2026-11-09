# U.S. Consumer Finance Segmentation

An original, reproducible unsupervised-learning project using the public 2022 Survey of Consumer Finances (SCF). The analysis follows the broad learning pipeline from WQU Project 6—EDA, subgroup filtering, two-feature K-means, multi-feature K-means, model comparison, PCA visualization, and an interactive dashboard—using original code and a public dataset.

## Research question

Within U.S. households who feared credit denial and had net worth below $2 million, what financial profiles emerge from clustering debt, housing, assets, income, and net worth?

## Data and sampling

- **Official source:** Board of Governors of the Federal Reserve System, [2022 Survey of Consumer Finances](https://www.federalreserve.gov/econres/scf_2022.htm).
- **CSV mirror used by the downloader:** [SCFP2022.csv](https://raw.githubusercontent.com/vaibhavselkar/Consumer-finance-in-USA-2022/main/SCFP2022.csv), a public GitHub copy of the Federal Reserve public-use data. The Federal Reserve landing page and codebook are the authoritative documentation.
- The public file has 22,975 records for 4,595 families, with five multiply-imputed records per family. The pipeline derives implicate number from `Y1` and `YY1` as specified in the Federal Reserve codebook and keeps implicate 1 so each family appears once.
- The project filters `TURNFEAR == 1` and `NETWORTH < 2,000,000`, then retains households with complete, finite values for selected financial measures and positive survey weights.
- The 2022 survey's `TURNFEAR` item identifies respondents who considered applying for credit during the previous five years but changed their mind because they thought they might be denied.

The CSV downloads on first run into `data/` and is excluded from Git. The script verifies required columns and fails with a useful message if the source schema changes.

## Pipeline

1. Inspect dimensions, missing values, subgroup sizes, and financial-variable distributions.
2. Build a two-feature baseline from `DEBT` and `HOUSES`.
3. Apply signed `log1p` transforms for skewed dollar measures, then standardize features.
4. Compare k=2 through k=12 separately for both feature sets using inertia and silhouette; fit a reproducible four-cluster model for consistent segment comparison.
5. Use a fixed five-feature financial set, display its 10%-trimmed variance ranking, fit the multi-feature model, and project it to two dimensions with PCA. Ranking all five candidates is descriptive, not feature selection.
6. Compare assignments and repeat fits with multiple seeds to examine stability. K-means is fit without weights to match the educational workflow; `WGT` is used for descriptive cluster profiles, not population-level inference.
7. Explore segments and feature sets in a Streamlit dashboard.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
python analyze.py
streamlit run app.py
```

The analysis writes `outputs/summary.json` and `outputs/cluster_assignments.csv`. The dashboard downloads and prepares data automatically on first launch.

## Interpretation limits

Clusters are algorithmic summaries, not real or fixed consumer types. Results depend on feature set, transformations, k, and sample definition. The SCF uses a complex sample design and multiple imputation; this project selects one implicate for unique-family clustering and reports survey-weighted descriptive profiles, but does not compute population-level inferential standard errors. Do not use clusters to make credit, eligibility, or lending decisions. Results describe 2022 data and should not be read as current credit-market conditions.

## References

- Board of Governors of the Federal Reserve System. *2022 Survey of Consumer Finances: Codebook and Public Data*. [SCF data page](https://www.federalreserve.gov/econres/scf_2022.htm) · [Codebook](https://www.federalreserve.gov/econres/files/codebk2022.txt).
- Aladangady, A., et al. (2023). “Changes in U.S. Family Finances from 2019 to 2022: Evidence from the Survey of Consumer Finances.” *Federal Reserve Bulletin*.

## Findings and validation

- The first-implicate subgroup contains **732 households**, from **4,595 source families**. Two-feature silhouette is 0.810; five-feature silhouette is 0.459. These are within-space separation measures, not prediction accuracy or directly comparable model-selection scores across feature spaces.
- Two-feature labels closely reproduce four debt/home-presence categories (**ARI 0.972**). The high silhouette largely reflects this simple structure, so the rule benchmark is reported explicitly.
- Five-feature assignments are stable to 20 household bootstrap refits (mean ARI **0.980**, minimum **0.933**) and show ARI **0.929–0.943** against other implicates on shared eligible households. These are exploratory sensitivity checks, not complex-survey uncertainty estimates.

Four clusters are retained for comparison, not claimed as the optimal number: multi-feature silhouette rises from 0.459 at k=4 to 0.474 at k=6 (rounded). Inspect the full sweep in `reports/summary.json` and the dashboard.

Descriptive profiles include unweighted medians and counts, weighted means and subgroup shares. At k=4 the reported mean profiles describe: a no-home-value group with positive net worth; a lower-debt group; a negative-net-worth group; and a higher-housing-value group. These are descriptions of group averages, not labels that describe every household. IDs are arbitrary and can change after refitting.

## Engineering and verification

- The dashboard caches diagnostics and fitted models, distinguishes source families from the filtered sample, and requires at least two features for PCA.
- `data_provenance.json` pins reviewed source bytes with a SHA-256 checksum. A changed download fails clearly until the source is reviewed.
- `requirements-lock.txt` records the tested Python 3.12 Linux dependency environment.
- Run `python -m unittest discover -s tests -v` and `python tests/dashboard_smoke.py`. The latter checks default startup, empty/one/multiple feature selections, and cluster-count endpoints. GitHub Actions also runs the full analysis.
- `reports/summary.json` is the full reviewed result. `results_summary.json` retains the earlier compact summary. Regenerate with `python analyze.py` and review outputs before refreshing published snapshots.
