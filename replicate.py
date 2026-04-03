"""
Replication script for "From Data Products to Inference: The Econometrics of Fuzzy Joins"

Produces:
  - tables/example_candidates.tex   (Section 3: candidate-match table)
  - tables/example_summary.tex      (Section 3: match outcome summary)
  - tables/estimators_toy.tex       (Section 4: toy example estimator comparison)
  - tables/estimators_main.tex      (Section 7: medium-scale estimator comparison)
  - tables/variance_decomp.tex      (Section 7: bootstrap variance decomposition)
  - tables/mi_diagnostics.tex       (Section 7: MI diagnostics via Rubin's rules)

Usage: python replicate.py
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.makedirs("tables", exist_ok=True)
np.random.seed(73)

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def lev_sim(s1, s2):
    """Normalized Levenshtein similarity in [0,1]."""
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return 0.0
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (0 if s1[i - 1] == s2[j - 1] else 1))
    return round(1.0 - dp[n][m] / max(n, m), 4)


def build_candidates(firms, villages, tau):
    """Build candidate-match table from firms and villages DataFrames."""
    vnames = villages["vname"].values
    vids = villages["vid"].values
    unique_reported = firms["reported_name"].unique()

    # Precompute scores per unique reported name
    name_cands = {}
    for rn in unique_reported:
        cands = []
        for j in range(len(vnames)):
            s = lev_sim(rn, vnames[j])
            if s >= tau:
                cands.append((vids[j], vnames[j], s))
        name_cands[rn] = cands

    pairs = []
    for _, f in firms.iterrows():
        rn = f["reported_name"]
        for (vid, target, s) in name_cands.get(rn, []):
            pairs.append({
                "fid": f["fid"], "vid": vid,
                "reported": rn, "target": target,
                "score": s, "true_vid": f["true_vid"],
                "is_true": f["true_vid"] == vid
            })
    pairs = pd.DataFrame(pairs)
    if len(pairs) > 0:
        pairs["rank"] = pairs.groupby("fid")["score"].rank(
            ascending=False, method="first").astype(int)
    return pairs


def build_csets(pairs, firms, tau, delta):
    """Build candidate set summaries with match status and outcomes."""
    firm_true_vid = firms.set_index("fid")["true_vid"].to_dict()
    csets = []
    for fid, grp in pairs.groupby("fid"):
        gs = grp.sort_values("score", ascending=False)
        scores = gs["score"].values
        vids_g = gs["vid"].values
        n = len(scores)
        best = scores[0] if n > 0 else 0
        second = scores[1] if n > 1 else 0
        gap = best - second

        if best < tau:
            status = "no_match"
        elif n == 1 or gap >= delta:
            status = "unambiguous"
        else:
            status = "ambiguous"

        assigned = int(vids_g[0]) if status == "unambiguous" else None
        true_vid = firm_true_vid[fid]

        if assigned is not None:
            outcome = "TP" if assigned == true_vid else "FP"
        else:
            outcome = "FN"

        csets.append({
            "fid": fid, "true_vid": true_vid, "n_cands": n,
            "best_score": best, "second_score": second, "gap": round(gap, 4),
            "status": status, "assigned_vid": assigned, "outcome": outcome,
            "cand_vids": list(vids_g[:8].astype(int)),
            "cand_scores": list(scores[:8].round(4)),
        })

    csets = pd.DataFrame(csets)

    # Add firms with zero candidates
    matched_fids = set(csets["fid"])
    missing = []
    for fid in firms["fid"]:
        if fid not in matched_fids:
            missing.append({
                "fid": fid, "true_vid": firm_true_vid[fid], "n_cands": 0,
                "best_score": 0, "second_score": 0, "gap": 0,
                "status": "no_match", "assigned_vid": None, "outcome": "FN",
                "cand_vids": [], "cand_scores": [],
            })
    if missing:
        csets = pd.concat([csets, pd.DataFrame(missing)], ignore_index=True)

    return csets.sort_values("fid").reset_index(drop=True)


def run_estimators(firms, villages, pairs, csets, tau, delta):
    """Run all estimators and return results dict."""
    village_lit = villages.set_index("vid")["literacy"].to_dict()
    firm_rev = firms.set_index("fid")["revenue"].to_dict()
    csets_list = csets.to_dict("records")
    csets_dict = {cs["fid"]: cs for cs in csets_list}
    results = {}

    # --- Oracle ---
    oracle = firms.merge(villages[["vid", "literacy"]], left_on="true_vid", right_on="vid")
    X = sm.add_constant(oracle["literacy"])
    mod = sm.OLS(oracle["revenue"], X).fit(
        cov_type="cluster", cov_kwds={"groups": oracle["true_vid"]})
    results["oracle"] = {
        "beta": mod.params["literacy"], "se": mod.bse["literacy"],
        "n_rows": len(oracle), "n_firms": len(oracle), "n_cl": oracle["true_vid"].nunique()
    }

    # --- Canonical (drop ambiguous) ---
    default_links = csets[csets["status"] == "unambiguous"][["fid", "assigned_vid"]].copy()
    default_links["assigned_vid"] = default_links["assigned_vid"].astype(int)
    dj = firms.merge(default_links, on="fid").merge(
        villages[["vid", "literacy"]], left_on="assigned_vid", right_on="vid")
    X = sm.add_constant(dj["literacy"])
    mod = sm.OLS(dj["revenue"], X).fit(
        cov_type="cluster", cov_kwds={"groups": dj["assigned_vid"]})
    results["canonical"] = {
        "beta": mod.params["literacy"], "se": mod.bse["literacy"],
        "n_rows": len(dj), "n_firms": len(dj), "n_cl": dj["assigned_vid"].nunique()
    }

    # --- Expanded full join ---
    fj = pairs[pairs["score"] >= tau].merge(firms[["fid", "revenue"]], on="fid")
    fj = fj.merge(villages[["vid", "literacy"]], on="vid")
    fj["w"] = fj.groupby("fid")["score"].transform(lambda x: x / x.sum())
    X = sm.add_constant(fj["literacy"])
    mod = sm.WLS(fj["revenue"], X, weights=fj["w"]).fit(
        cov_type="cluster", cov_kwds={"groups": fj["fid"]})
    results["expanded"] = {
        "beta": mod.params["literacy"], "se": mod.bse["literacy"],
        "n_rows": len(fj), "n_firms": fj["fid"].nunique(), "n_cl": fj["fid"].nunique()
    }

    # --- Collapsed (score-weighted avg X) ---
    collapsed_rows = []
    for fid, grp in fj.groupby("fid"):
        ws = grp["score"] / grp["score"].sum()
        x_avg = np.average(grp["literacy"], weights=ws)
        rev = grp["revenue"].iloc[0]
        best_vid = grp.loc[grp["score"].idxmax(), "vid"]
        collapsed_rows.append({
            "fid": fid, "revenue": rev, "literacy_avg": x_avg,
            "best_vid": int(best_vid),
        })
    collapsed = pd.DataFrame(collapsed_rows)
    X = sm.add_constant(collapsed["literacy_avg"])
    mod = sm.OLS(collapsed["revenue"], X).fit(
        cov_type="cluster", cov_kwds={"groups": collapsed["best_vid"]})
    results["collapsed"] = {
        "beta": mod.params["literacy_avg"], "se": mod.bse["literacy_avg"],
        "n_rows": len(collapsed), "n_firms": len(collapsed),
        "n_cl": collapsed["best_vid"].nunique()
    }

    # --- MI (Rubin's rules) ---
    M = 200
    mi_betas = []
    mi_within_vars = []

    for m in range(M):
        draw_rows = []
        for cs in csets_list:
            fid = cs["fid"]
            rev = firm_rev[fid]
            if cs["status"] == "unambiguous":
                vid = int(cs["assigned_vid"])
            elif cs["n_cands"] > 0:
                scores = np.array(cs["cand_scores"][:cs["n_cands"]])
                vids_arr = np.array(cs["cand_vids"][:cs["n_cands"]])
                mask = scores >= tau
                if not mask.any():
                    continue
                scores = scores[mask]
                vids_arr = vids_arr[mask]
                probs = scores / scores.sum()
                vid = int(np.random.choice(vids_arr, p=probs))
            else:
                continue
            draw_rows.append({
                "fid": fid, "vid": vid, "revenue": rev,
                "literacy": village_lit[vid]
            })
        dd = pd.DataFrame(draw_rows)
        if len(dd) < 20:
            continue
        X = sm.add_constant(dd["literacy"])
        mod = sm.OLS(dd["revenue"], X).fit(
            cov_type="cluster", cov_kwds={"groups": dd["vid"]})
        mi_betas.append(mod.params["literacy"])
        mi_within_vars.append(mod.bse["literacy"] ** 2)

    mi_betas = np.array(mi_betas)
    mi_within_vars = np.array(mi_within_vars)
    Q_bar = np.mean(mi_betas)
    U_bar = np.mean(mi_within_vars)
    B_var = np.var(mi_betas, ddof=1)
    T_var = U_bar + (1 + 1 / M) * B_var
    lam = (1 + 1 / M) * B_var / T_var

    results["mi"] = {
        "beta": Q_bar, "se": np.sqrt(T_var),
        "n_rows": "---", "n_firms": len(csets_list), "n_cl": "---",
        "U_bar": U_bar, "B_var": B_var, "T_var": T_var, "lambda": lam, "M": M,
    }

    # --- Bootstrap variance decomposition ---
    B_boot = 200
    N_F = len(firms)
    oracle_arr = oracle[["literacy", "revenue"]].values

    boot_sampling = []
    boot_matching = []
    boot_joint = []

    for b in range(B_boot):
        # Sampling only
        idx = np.random.choice(N_F, size=N_F, replace=True)
        bo = oracle.iloc[idx]
        Xb = sm.add_constant(bo["literacy"])
        mod = sm.OLS(bo["revenue"], Xb).fit()
        boot_sampling.append(mod.params["literacy"])

        # Matching only
        draw_rows = []
        for cs in csets_list:
            fid = cs["fid"]
            rev = firm_rev[fid]
            if cs["status"] == "unambiguous":
                vid = int(cs["assigned_vid"])
            elif cs["n_cands"] > 0:
                scores = np.array(cs["cand_scores"][:cs["n_cands"]])
                vids_arr = np.array(cs["cand_vids"][:cs["n_cands"]])
                mask = scores >= tau
                if not mask.any():
                    continue
                scores = scores[mask]
                vids_arr = vids_arr[mask]
                vid = int(np.random.choice(vids_arr, p=scores / scores.sum()))
            else:
                continue
            draw_rows.append({
                "fid": fid, "vid": vid, "revenue": rev,
                "literacy": village_lit[vid]
            })
        dd = pd.DataFrame(draw_rows)
        Xb = sm.add_constant(dd["literacy"])
        mod = sm.OLS(dd["revenue"], Xb).fit()
        boot_matching.append(mod.params["literacy"])

        # Joint
        boot_fids = np.random.choice(N_F, size=N_F, replace=True)
        draw_rows_j = []
        for fid in boot_fids:
            cs = csets_dict[fid]
            rev = firm_rev[fid]
            if cs["status"] == "unambiguous":
                vid = int(cs["assigned_vid"])
            elif cs["n_cands"] > 0:
                scores = np.array(cs["cand_scores"][:cs["n_cands"]])
                vids_arr = np.array(cs["cand_vids"][:cs["n_cands"]])
                mask = scores >= tau
                if not mask.any():
                    continue
                scores = scores[mask]
                vids_arr = vids_arr[mask]
                vid = int(np.random.choice(vids_arr, p=scores / scores.sum()))
            else:
                continue
            draw_rows_j.append({
                "fid": fid, "vid": vid, "revenue": rev,
                "literacy": village_lit[vid]
            })
        dd_j = pd.DataFrame(draw_rows_j)
        Xb = sm.add_constant(dd_j["literacy"])
        mod = sm.OLS(dd_j["revenue"], Xb).fit()
        boot_joint.append(mod.params["literacy"])

    boot_sampling = np.array(boot_sampling)
    boot_matching = np.array(boot_matching)
    boot_joint = np.array(boot_joint)

    results["bootstrap"] = {
        "var_sampling": boot_sampling.var(),
        "var_matching": boot_matching.var(),
        "var_joint": boot_joint.var(),
        "sd_sampling": boot_sampling.std(),
        "sd_matching": boot_matching.std(),
        "sd_joint": boot_joint.std(),
    }

    return results


# ============================================================
# TOY EXAMPLE (10 villages, 51 firms) — Section 3
# ============================================================

print("=" * 60)
print("TOY EXAMPLE")
print("=" * 60)

toy_villages = pd.DataFrame([
    {"vid": 0, "vname": "rampur", "literacy": 0.85},
    {"vid": 1, "vname": "rampura", "literacy": 0.70},
    {"vid": 2, "vname": "ramnagar", "literacy": 0.50},
    {"vid": 3, "vname": "sultanpur", "literacy": 0.65},
    {"vid": 4, "vname": "sultanganj", "literacy": 0.45},
    {"vid": 5, "vname": "dharamkot", "literacy": 0.40},
    {"vid": 6, "vname": "kankavli", "literacy": 0.35},
    {"vid": 7, "vname": "phulwari", "literacy": 0.50},
    {"vid": 8, "vname": "bagaha", "literacy": 0.30},
    {"vid": 9, "vname": "lakhisarai", "literacy": 0.65},
])
toy_villages["eta"] = np.random.normal(0, 0.8, len(toy_villages))

# Controlled firm assignments and name noise
toy_assign = (
    [0]*8 + [1]*7 + [2]*6 + [3]*5 + [4]*4 +
    [5]*5 + [6]*5 + [7]*5 + [8]*4 + [9]*2
)

noise_map = {
    "rampur": ["rampur", "rampr", "rampur", "rampure", "ramnpur",
               "rampur", "rampura", "rampr"],
    "ramnagar": ["ramnagar", "ramnagor", "ramnagr", "ramnagar", "ramnapur",
                 "ramnagar", "ramnagar"],
    "rampura": ["rampura", "rampura", "rampur", "rampura", "rampra", "rampura"],
    "sultanpur": ["sultanpur", "sultampur", "sultanpur", "sultanpor", "sultanpur"],
    "sultanganj": ["sultanganj", "sultangajn", "sultanganj", "sultangnj"],
    "dharamkot": ["dharamkot", "dharmkot", "dharamkot", "dharamkt", "dharamkot"],
    "kankavli": ["kankavli", "kankali", "kankavli", "kankavl", "kankavli"],
    "phulwari": ["phulwari", "phulwri", "phulwari", "fulwari", "phulwari"],
    "bagaha": ["bagaha", "baagaha", "bagha", "bagaha"],
    "lakhisarai": ["lakhisarai", "lakhisrai"],
}

toy_firms = []
for i, vid in enumerate(toy_assign):
    v = toy_villages.loc[vid]
    eps = np.random.normal(0, 1.5)
    y = 3.0 + 5.0 * v["literacy"] + v["eta"] + eps
    vname = v["vname"]
    opts = noise_map.get(vname, [vname])
    reported = opts[i % len(opts)]
    toy_firms.append({
        "fid": i, "true_vid": vid, "true_vname": vname,
        "reported_name": reported, "revenue": round(y, 3)
    })
toy_firms = pd.DataFrame(toy_firms)

TOY_TAU = 0.50
TOY_DELTA = 0.10

toy_pairs = build_candidates(toy_firms, toy_villages, TOY_TAU)
toy_csets = build_csets(toy_pairs, toy_firms, TOY_TAU, TOY_DELTA)

# --- TABLE: Candidate-match table for example firms ---
example_fids = [0, 6, 3, 11]
ex_pairs = toy_pairs[toy_pairs["fid"].isin(example_fids)].sort_values(["fid", "rank"])
tex = "\\begin{tabular}{rllllcr}\n\\toprule\n"
tex += "Firm & True village & Reported & Candidate & $s_{ij}$ & Rank & True? \\\\\n"
tex += "\\midrule\n"
for _, row in ex_pairs.iterrows():
    true_str = "\\checkmark" if row["is_true"] else ""
    tex += f"{row['fid']} & {row['target'] if row['is_true'] else '---'} & "
    tex += f"{row['reported']} & {row['target']} & {row['score']:.2f} & "
    tex += f"{int(row['rank'])} & {true_str} \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n"

# Rewrite with cleaner formatting: show true village only once per firm
tex = "\\begin{tabular}{rllccr}\n\\toprule\n"
tex += "Firm & True & Reported & Candidate & $s_{ij}$ & Rank \\\\\n"
tex += "\\midrule\n"
for fid in example_fids:
    fp = ex_pairs[ex_pairs["fid"] == fid].sort_values("rank")
    true_vname = toy_firms.loc[toy_firms["fid"] == fid, "true_vname"].values[0]
    reported = fp.iloc[0]["reported"]
    for idx_r, (_, row) in enumerate(fp.iterrows()):
        f_str = str(fid) if idx_r == 0 else ""
        tv_str = true_vname if idx_r == 0 else ""
        rp_str = reported if idx_r == 0 else ""
        marker = "$\\leftarrow$" if row["is_true"] else ""
        tex += f"{f_str} & {tv_str} & {rp_str} & {row['target']}{marker} & "
        tex += f"{row['score']:.2f} & {int(row['rank'])} \\\\\n"
    if fid != example_fids[-1]:
        tex += "\\midrule\n"
tex += "\\bottomrule\n\\end{tabular}\n"

with open("tables/example_candidates.tex", "w") as f:
    f.write(tex)
print("Wrote tables/example_candidates.tex")

# --- TABLE: Match outcome summary ---
status_ct = toy_csets["status"].value_counts()
outcome_ct = toy_csets["outcome"].value_counts()
n_total = len(toy_csets)

tex = "\\begin{tabular}{lrr}\n\\toprule\n"
tex += "Category & Count & Share \\\\\n\\midrule\n"
for s in ["unambiguous", "ambiguous", "no_match"]:
    c = status_ct.get(s, 0)
    tex += f"{s.replace('_', ' ').title()} & {c} & {100*c/n_total:.1f}\\% \\\\\n"
tex += "\\midrule\n"
for o in ["TP", "FP", "FN"]:
    c = outcome_ct.get(o, 0)
    tex += f"{o} & {c} & {100*c/n_total:.1f}\\% \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n"

with open("tables/example_summary.tex", "w") as f:
    f.write(tex)
print("Wrote tables/example_summary.tex")

# --- Run toy estimators ---
toy_results = run_estimators(
    toy_firms, toy_villages, toy_pairs, toy_csets, TOY_TAU, TOY_DELTA)

# --- TABLE: Toy estimator comparison ---
tex = "\\begin{tabular}{lrrrr}\n\\toprule\n"
tex += "Method & $\\hat\\beta_1$ & SE & $N$ (rows) & Clusters \\\\\n\\midrule\n"
tex += f"True DGP & 5.000 & --- & --- & --- \\\\\n"
for label, key in [("Oracle", "oracle"), ("Canonical", "canonical"),
                   ("Expanded", "expanded"), ("Collapsed", "collapsed"),
                   ("MI (Rubin)", "mi")]:
    r = toy_results[key]
    n_str = str(r["n_rows"])
    cl_str = str(r["n_cl"])
    tex += f"{label} & {r['beta']:.3f} & {r['se']:.3f} & {n_str} & {cl_str} \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n"

with open("tables/estimators_toy.tex", "w") as f:
    f.write(tex)
print("Wrote tables/estimators_toy.tex")


# ============================================================
# MEDIUM-SCALE SIMULATION (40 villages, 600 firms) — Section 7
# ============================================================

print("\n" + "=" * 60)
print("MEDIUM-SCALE SIMULATION")
print("=" * 60)

np.random.seed(73)

cluster_A = [
    {"vid": 0, "vname": "rampur", "literacy": 0.85},
    {"vid": 1, "vname": "rampura", "literacy": 0.70},
    {"vid": 2, "vname": "ramnagar", "literacy": 0.50},
    {"vid": 3, "vname": "ramgarh", "literacy": 0.35},
    {"vid": 4, "vname": "ramganj", "literacy": 0.60},
]
cluster_B = [
    {"vid": 5, "vname": "sultanpur", "literacy": 0.70},
    {"vid": 6, "vname": "sultanganj", "literacy": 0.50},
    {"vid": 7, "vname": "sultanabad", "literacy": 0.30},
    {"vid": 8, "vname": "sultangarh", "literacy": 0.55},
]
cluster_C = [
    {"vid": 9, "vname": "lakhisarai", "literacy": 0.65},
    {"vid": 10, "vname": "lakhimpur", "literacy": 0.35},
    {"vid": 11, "vname": "lakhanpur", "literacy": 0.50},
]
distinctive = [
    {"vid": 12+i, "vname": n, "literacy": l}
    for i, (n, l) in enumerate([
        ("dharamkot", 0.42), ("kankavli", 0.38), ("phulwari", 0.52),
        ("bagaha", 0.33), ("sasaram", 0.48), ("bithoor", 0.56),
        ("daltonganj", 0.44), ("munger", 0.62), ("bariarpur", 0.37),
        ("kahalgaon", 0.51), ("bhagalpur", 0.58), ("tamkuhi", 0.41),
        ("narsingh", 0.47), ("rajmahal", 0.53), ("chandauli", 0.46),
        ("ambika", 0.39), ("simri", 0.43), ("jaleshwar", 0.57),
        ("araria", 0.36), ("madhubani", 0.54), ("sitamarhi", 0.49),
        ("bettiah", 0.45), ("motihari", 0.50), ("gopalganj", 0.42),
        ("chhapra", 0.55), ("hajipur", 0.48), ("patna", 0.72),
        ("muzaffarpur", 0.61),
    ])
]

all_v = cluster_A + cluster_B + cluster_C + distinctive
main_villages = pd.DataFrame(all_v)
N_V = len(main_villages)
main_villages["eta"] = np.random.normal(0, 0.8, N_V)

# Cluster membership for diagnostics
main_villages["cluster"] = "distinctive"
for v in cluster_A:
    main_villages.loc[main_villages["vid"] == v["vid"], "cluster"] = "ram"
for v in cluster_B:
    main_villages.loc[main_villages["vid"] == v["vid"], "cluster"] = "sultan"
for v in cluster_C:
    main_villages.loc[main_villages["vid"] == v["vid"], "cluster"] = "lakhi"

# Firm assignments
village_weights = np.ones(N_V)
for v in cluster_A:
    village_weights[v["vid"]] = 6.0
for v in cluster_B:
    village_weights[v["vid"]] = 5.0
for v in cluster_C:
    village_weights[v["vid"]] = 4.0
village_weights /= village_weights.sum()

N_F = 600
firm_vids = np.random.choice(N_V, size=N_F, p=village_weights)

cluster_names = set(v["vname"] for v in cluster_A + cluster_B + cluster_C)


def add_noise(name, level=0.12):
    chars = list(name)
    out = []
    for c in chars:
        r = np.random.random()
        if r < level:
            op = np.random.choice(["swap", "drop", "insert", "none"], p=[0.3, 0.2, 0.2, 0.3])
            if op == "swap":
                out.append(np.random.choice(list("aeiou" if c in "aeiou" else "bcdfghjklmnpqrstvwxyz")))
            elif op == "drop":
                pass
            elif op == "insert":
                out.append(c)
                out.append(np.random.choice(list("aeiou")))
            else:
                out.append(c)
        else:
            out.append(c)
    return ''.join(out) if len(out) > 2 else name


main_firms = []
for i in range(N_F):
    vid = firm_vids[i]
    v = main_villages.loc[vid]
    eps = np.random.normal(0, 1.5)
    y = 3.0 + 5.0 * v["literacy"] + v["eta"] + eps
    vname = v["vname"]
    noise_level = 0.28 if vname in cluster_names else 0.12
    reported = add_noise(vname, level=noise_level)
    main_firms.append({
        "fid": i, "true_vid": vid, "true_vname": vname,
        "reported_name": reported, "revenue": round(y, 3)
    })
main_firms = pd.DataFrame(main_firms)

MAIN_TAU = 0.50
MAIN_DELTA = 0.06

main_pairs = build_candidates(main_firms, main_villages, MAIN_TAU)
main_csets = build_csets(main_pairs, main_firms, MAIN_TAU, MAIN_DELTA)

# Print diagnostics
status_ct = main_csets["status"].value_counts()
outcome_ct = main_csets["outcome"].value_counts()
print(f"\nMatch status: {dict(status_ct)}")
print(f"Match outcomes: {dict(outcome_ct)}")

# Run estimators
main_results = run_estimators(
    main_firms, main_villages, main_pairs, main_csets, MAIN_TAU, MAIN_DELTA)

# --- TABLE: Main estimator comparison ---
tex = "\\begin{tabular}{lrrrr}\n\\toprule\n"
tex += "Method & $\\hat\\beta_1$ & SE & $N$ (rows) & Clusters \\\\\n\\midrule\n"
tex += f"True DGP & 5.000 & --- & --- & --- \\\\\n"
for label, key in [("Oracle", "oracle"), ("Canonical", "canonical"),
                   ("Expanded", "expanded"), ("Collapsed", "collapsed"),
                   ("MI (Rubin)", "mi")]:
    r = main_results[key]
    n_str = str(r["n_rows"])
    cl_str = str(r["n_cl"])
    tex += f"{label} & {r['beta']:.3f} & {r['se']:.3f} & {n_str} & {cl_str} \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n"

with open("tables/estimators_main.tex", "w") as f:
    f.write(tex)
print("Wrote tables/estimators_main.tex")

# --- TABLE: MI diagnostics ---
mi = main_results["mi"]
tex = "\\begin{tabular}{lr}\n\\toprule\n"
tex += "Quantity & Value \\\\\n\\midrule\n"
tex += f"Imputations ($M$) & {mi['M']} \\\\\n"
tex += f"Pooled $\\hat\\beta_1$ & {mi['beta']:.4f} \\\\\n"
tex += f"Within-imputation var ($\\bar U$) & {mi['U_bar']:.4f} \\\\\n"
tex += f"Between-imputation var ($B$) & {mi['B_var']:.4f} \\\\\n"
tex += f"Total var ($T$) & {mi['T_var']:.4f} \\\\\n"
tex += f"Total SE & {mi['se']:.4f} \\\\\n"
tex += f"$\\lambda$ & {mi['lambda']:.4f} \\\\\n"
tex += f"95\\% CI & [{mi['beta']-1.96*mi['se']:.3f}, {mi['beta']+1.96*mi['se']:.3f}] \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n"

with open("tables/mi_diagnostics.tex", "w") as f:
    f.write(tex)
print("Wrote tables/mi_diagnostics.tex")

# --- TABLE: Bootstrap variance decomposition ---
bt = main_results["bootstrap"]
tex = "\\begin{tabular}{lrr}\n\\toprule\n"
tex += "Component & SD$(\\hat\\beta_1)$ & Var$(\\hat\\beta_1)$ \\\\\n\\midrule\n"
tex += f"Sampling only & {bt['sd_sampling']:.4f} & {bt['var_sampling']:.4f} \\\\\n"
tex += f"Matching only & {bt['sd_matching']:.4f} & {bt['var_matching']:.4f} \\\\\n"
tex += f"Joint & {bt['sd_joint']:.4f} & {bt['var_joint']:.4f} \\\\\n"
tex += f"Sum of components & --- & {bt['var_sampling']+bt['var_matching']:.4f} \\\\\n"
tex += "\\bottomrule\n\\end{tabular}\n"

with open("tables/variance_decomp.tex", "w") as f:
    f.write(tex)
print("Wrote tables/variance_decomp.tex")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nOracle:    beta={main_results['oracle']['beta']:.4f}")
print(f"Canonical: beta={main_results['canonical']['beta']:.4f}")
print(f"Expanded:  beta={main_results['expanded']['beta']:.4f}")
print(f"Collapsed: beta={main_results['collapsed']['beta']:.4f}")
print(f"MI:        beta={main_results['mi']['beta']:.4f}, lambda={main_results['mi']['lambda']:.4f}")
print(f"\nAll tables written to tables/")
