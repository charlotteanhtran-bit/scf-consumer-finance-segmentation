# U.S. Consumer Finance Segmentation

This project looks at the finances of families who held back from applying for credit because they expected to be turned down. It uses the 2022 Survey of Consumer Finances (SCF), with a further restriction to net worth below $2 million.

The analysis compares a simple debt-and-housing model with a five-variable K-means model. A Streamlit dashboard lets you change the number of clusters and the financial variables used.

## What the results show

The filtered sample contains **732 families**, drawn from **4,595 families** in the source file.

The debt-and-housing model produces a high silhouette score of **0.810** at four clusters. But a simple rule based on whether debt and home value are positive gives almost the same grouping: the adjusted Rand index (ARI) is **0.972**. Much of the apparent separation comes from that basic distinction.

Adding assets, income, and net worth gives a broader view of the families' finances. At four clusters, this model has a silhouette score of **0.459**. The two scores describe separation in different feature spaces, so they are not a fair ranking of which model is better.

The four mean profiles roughly describe a group with no home value and positive net worth, a lower-debt group, a negative-net-worth group, and a higher-home-value group. These are summaries of the groups, not descriptions that fit every family. The dashboard includes medians as well as means so large values do not tell the whole story.

### How stable are the groups?

| Check on the five-variable model | Result |
| --- | --- |
| Refit on 20 bootstrap samples | Mean ARI 0.980; lowest ARI 0.933 |
| Refit using the other imputed versions of the data | ARI 0.929–0.943 on shared eligible families |
| Increase the cluster count from four to six | Silhouette rises from 0.459 to 0.474 |

ARI measures agreement between assignments; a value of 1 means the assignments match, even if the cluster numbers differ.

Four clusters are kept as a common setting for the comparisons. They are not presented as the best possible choice. The full sweep from two to twelve clusters is in [reports/summary.json](reports/summary.json) and the dashboard.

## Data preparation and model choices

The source has 22,975 rows because each family appears in five imputed versions of the data. The main analysis keeps version 1, using `Y1 - 10 * YY1` to identify it. This avoids treating five records for one family as five separate families. The other versions are used in the sensitivity check.

The sample keeps `TURNFEAR == 1` and `NETWORTH < 2,000,000`, then removes rows with missing or infinite financial values or nonpositive survey weights. The `TURNFEAR` question covers people who considered applying for credit in the previous five years but changed their mind because they expected denial.

Both models use the same preparation:

1. Transform dollar values with signed `log1p`. This reduces the influence of very large values while keeping zero and negative values, including negative net worth.
2. Standardize the transformed variables before fitting K-means.
3. Check inertia and silhouette across cluster counts, then compare the four-cluster assignments.

The baseline uses `DEBT` and `HOUSES`. The second model adds `ASSET`, `INCOME`, and `NETWORTH`. A trimmed-variance table describes these five candidates; it does not select a smaller set. PCA is used only to display the fitted groups in two dimensions.

K-means is fitted without survey weights. The weights are used afterward to calculate descriptive means and shares within the filtered sample.

## Explore the dashboard

```bash
python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
python analyze.py
streamlit run app.py
```

Use Python 3.12 with the pinned dependencies. The first run downloads the CSV into `data/`, which is excluded from Git. The dashboard also prepares the data automatically when opened.

The dashboard includes debt-and-housing clusters, a PCA view of the selected variables, group profiles, and cluster-count diagnostics. You can choose two to five variables and between two and eight clusters.

The analysis writes `outputs/summary.json` and `outputs/cluster_assignments.csv`. The full published summary is [reports/summary.json](reports/summary.json); `results_summary.json` is an older, shorter snapshot. Review a new run before replacing either published file.

## Limits

The SCF is a complex survey with multiple imputation. Selecting one imputed version and weighting the profiles does not provide full survey-based uncertainty estimates. The bootstrap and alternative-imputation checks show how the assignments change; they are not population confidence intervals.

Clusters also depend on the variables, transformations, and number of groups chosen. They summarize this part of the 2022 sample and have not been validated as credit-risk categories or lending tools.

## Checks

```bash
python -m unittest discover -s tests -v
python tests/dashboard_smoke.py
```

The unit tests cover one-record-per-family selection, duplicate-family rejection, and the signed log transform. The dashboard check tries default startup, empty and single-variable selections, multiple variables, and both ends of the cluster slider. GitHub Actions is configured to run these checks and the full analysis.

The downloader checks the file against the SHA-256 in `data_provenance.json`. If the upstream file changes, the script stops so the new data can be reviewed before use.

## Sources and background

The workflow builds on the clustering topics in WQU Project 6: subgroup preparation, K-means, model comparison, PCA, and a dashboard.

- Federal Reserve Board: [2022 SCF data page](https://www.federalreserve.gov/econres/scf_2022.htm) and [codebook](https://www.federalreserve.gov/econres/files/codebk2022.txt).
- The downloader uses this [public CSV mirror](https://raw.githubusercontent.com/vaibhavselkar/Consumer-finance-in-USA-2022/main/SCFP2022.csv). The Federal Reserve documentation is the reference for variable definitions.
- Aladangady, A., et al. (2023). “Changes in U.S. Family Finances from 2019 to 2022: Evidence from the Survey of Consumer Finances.” *Federal Reserve Bulletin*.
