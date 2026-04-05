# Inference With Fuzzy-Joined Data

Data products built on fuzzy joins ship a single canonical linkage, hiding the many-to-many candidate-match graph from which it was constructed. When analysts get the graph, the two obvious things to do with it—expand it into a regression dataset or average covariates across candidates—both produce biased estimates. The expanded join attenuates. The averaged-covariate approach can amplify or attenuate, and you can't tell which without ground truth.

This repo contains the paper, replication code, and a proposed relational schema for shipping fuzzy-joined data products.

## What's in the paper

- **Proposition 1.** The expanded join (one row per candidate pair) adds a within-candidate-set variance penalty to the OLS denominator, shrinking the coefficient toward zero relative to the score-averaged estimator.

- **Proposition 2.** The collapsed estimator (one row per firm, covariate averaged across candidates) has a nonclassical errors-in-variables structure. The bias depends on Cov(X*, u), which is generically nonzero when mismatches are confined to string-similar neighbors. The sign is indeterminate.

- **Multiple imputation** over candidate assignments avoids both pathologies. Each draw is a single-valued firm-to-village assignment; Rubin's rules propagate matching uncertainty. The between-imputation share of variance, λ, tells you whether matching uncertainty matters.

- **Regression calibration** via a validation sample is simpler when available. The key requirement: the golden set must be a random sample of the *full population*, including records the canonical policy dropped. A sample of matched records only lets you correct for false positives but leaves selection bias from false negatives invisible.

## Repo structure

```
├── ms/
│   ├── fuzzy_joins_inference.tex   # Main paper
│   ├── fuzzy_joins_inference.pdf   # Compiled PDF
│   ├── fuzzy_joins_SI.tex          # Supplementary: database schemas
│   └── references.bib
├── scripts/
│   └── replicate.py                # Generates all tables
├── tabs/                           # LaTeX tables (generated)
│   ├── example_candidates.tex
│   ├── example_summary.tex
│   ├── estimators_toy.tex
│   ├── estimators_main.tex
│   ├── mi_diagnostics.tex
│   └── variance_decomp.tex
└── README.md
```

## Replication

```bash
pip install numpy pandas statsmodels
python scripts/replicate.py
cd ms && pdflatex fuzzy_joins_inference && bibtex fuzzy_joins_inference && pdflatex fuzzy_joins_inference && pdflatex fuzzy_joins_inference
```

`scripts/replicate.py` runs two simulations (toy: 10 villages, 51 firms; medium: 40 villages, 600 firms), computes all estimators (oracle, canonical, expanded, collapsed, MI), and writes six `.tex` tables to `tabs/`.

## Key result

| Method | β̂₁ | SE | N |
|---|---|---|---|
| True DGP | 5.00 | — | — |
| Oracle | 6.32 | 1.23 | 600 |
| Canonical | 6.05 | 1.17 | 576 |
| Expanded | 3.65 | 0.33 | 1892 |
| **Collapsed** | **9.21** | **1.56** | **598** |
| MI (Rubin) | 6.01 | 1.15 | — |

The collapsed estimator overshoots by 46%. The expanded undershoots by 42%. MI recovers the oracle to within 5%.

## Schema

The SI proposes five relational tables for shipping fuzzy-joined data products:

1. **`canonical_linkage`** — one row per source record (left join, not inner join), with `match_status`, `score_gap`, `n_candidates`, `component_id`
2. **`candidate_matches`** — one row per (source, candidate) pair above screening threshold
3. **`validation_sample`** — verified true links for a random sample of the *full population*
4. **`components`** — connected component diagnostics (`n_ambiguous`, covariate range)
5. **Metadata** — algorithm version, thresholds, score interpretation

## Citation

```bibtex
@article{sood2025fuzzyjoin,
  author  = {Sood, Gaurav},
  title   = {From Data Products to Inference: The Econometrics of Fuzzy Joins},
  year    = {2025}
}
```
