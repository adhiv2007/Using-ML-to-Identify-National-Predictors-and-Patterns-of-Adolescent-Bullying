from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations

RESULT_DIR = Path("../results")
OUT_DIR = RESULT_DIR / "pattern_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEING10_CSV = RESULT_DIR / "beingbullied_top10.csv"       # Country,Type,Rank,Feature,Importance
TYPES10_CSV = RESULT_DIR / "bullying_types_top10.csv"     # Country,Type,AUC,Rank,Feature,Importance

OUTCOMES = ["BEINGBULLIED", "BULLY_VERBAL", "BULLY_THREAT", "BULLY_PHYSICAL", "BULLY_RELATIONAL"]

POINTS_MAP = {1:10,2:9,3:8,4:7,5:6,6:5,7:4,8:3,9:2,10:1}

PRED_TO_CATEGORY = {
    # Individual
    "ST004D01T":"Individual","AGE":"Individual","GRADE":"Individual","BSMJ":"Individual","JOYREAD":"Individual",
    "SCREADCOMP":"Individual","SCREADDIFF":"Individual","COMPETE":"Individual","WORKMAST":"Individual",
    "GFOFAIL":"Individual","EUDMO":"Individual","RESILIENCE":"Individual","MASTGOAL":"Individual",
    "PV1READ":"Individual","PV1MATH":"Individual","PV1SCIE":"Individual",

    # Proximal
    "REPEAT":"Proximal","UNDREM":"Proximal","METASUM":"Proximal","METASPAM":"Proximal",

    # Microsystem
    "EMOSUPS":"Microsystem","DURECEC":"Microsystem","BELONG":"Microsystem","PERCOMP":"Microsystem",
    "PERCOOP":"Microsystem","ATTLNACT":"Microsystem","DISCLIMA":"Microsystem","TEACHSUP":"Microsystem",
    "DIRINS":"Microsystem","PERFEED":"Microsystem","STIMREAD":"Microsystem","ADAPTIVITY":"Microsystem",
    "TEACHINT":"Microsystem","SWBP":"Microsystem",
    "ST206Q01HA":"Microsystem","ST207Q01HA":"Microsystem","ST207Q02HA":"Microsystem","ST207Q03HA":"Microsystem",
    "ST207Q04HA":"Microsystem","ST207Q05HA":"Microsystem",

    # Macro
    "ESCS":"Macro","EDUSHORT":"Macro","RATCMP1":"Macro","RATCMP2":"Macro",
}

def load_top10():
    df_being = pd.read_csv(BEING10_CSV)
    df_types = pd.read_csv(TYPES10_CSV)

    # Normalize columns
    for df in (df_being, df_types):
        df.columns = [c.strip().title() for c in df.columns]
        if "Feature" in df.columns:
            df.rename(columns={"Feature": "Predictor"}, inplace=True)

    # Defensive filter to rank<=10
    df_being = df_being[df_being["Rank"].between(1, 10)]
    df_types = df_types[df_types["Rank"].between(1, 10)]
    return pd.concat([df_being, df_types], ignore_index=True)

def jaccard_similarity(sets_by_country):
    countries = sorted(sets_by_country.index)
    jacc = pd.DataFrame(0.0, index=countries, columns=countries)
    for i, ci in enumerate(countries):
        si = sets_by_country.loc[ci]
        for cj in countries[i:]:
            sj = sets_by_country.loc[cj]
            inter = len(si & sj)
            union = len(si | sj)
            sim = inter / union if union else 0.0
            jacc.loc[ci, cj] = jacc.loc[cj, ci] = sim
    return jacc

def cosine_similarity_rankpoints(pivot_points):
    A = pivot_points.values.astype(float)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    cos = (A @ A.T) / (norms @ norms.T)
    return pd.DataFrame(cos, index=pivot_points.index, columns=pivot_points.index)

def threshold_communities(sim_df, thresh):
    unvisited = set(sim_df.index)
    communities = []
    while unvisited:
        c = unvisited.pop()
        group = {c}
        added = True
        while added:
            added = False
            for x in list(unvisited):
                if any(sim_df.loc[x, g] >= thresh for g in group):
                    group.add(x); unvisited.remove(x); added = True
        communities.append(sorted(group))
    return communities

def lift(a_support, b_support, ab_support, universe):
    # support = count / universe; lift = P(A∩B) / (P(A) P(B))
    if a_support == 0 or b_support == 0:
        return np.nan
    pa = a_support / universe
    pb = b_support / universe
    pab = ab_support / universe
    return pab / (pa * pb) if pa*pb else np.nan

def itemset_mining(country_sets, max_k=3, top_n=30, min_support=2):
    """
    Simple frequent itemset mining up to size k (2 and 3 by default).
    Returns DataFrames for pairs and triples with support and lift.
    """
    countries = list(country_sets.index)
    N = len(countries)
    # 1-item supports
    item_support = Counter()
    for s in country_sets:
        for x in s:
            item_support[x] += 1

    def pairs_stats():
        rows = []
        all_items = sorted(item_support.keys())
        for a, b in combinations(all_items, 2):
            ab = sum(1 for s in country_sets if a in s and b in s)
            if ab >= min_support:
                L = lift(item_support[a], item_support[b], ab, N)
                rows.append((a, b, ab, item_support[a], item_support[b], L))
        df = pd.DataFrame(rows, columns=["A","B","Support_AB","Support_A","Support_B","Lift"])
        df.sort_values(["Support_AB","Lift"], ascending=[False, False], inplace=True)
        return df.head(top_n)

    def triples_stats():
        rows = []
        all_items = sorted(item_support.keys())
        for a, b, c in combinations(all_items, 3):
            abc = sum(1 for s in country_sets if (a in s and b in s and c in s))
            if abc >= min_support:
                # Approx lift 3-way vs independence
                pa, pb, pc = item_support[a]/N, item_support[b]/N, item_support[c]/N
                pabc = abc/N
                L = pabc / (pa*pb*pc) if pa*pb*pc else np.nan
                rows.append((a,b,c,abc,item_support[a],item_support[b],item_support[c],L))
        df = pd.DataFrame(rows, columns=["A","B","C","Support_ABC","Support_A","Support_B","Support_C","Lift3"])
        df.sort_values(["Support_ABC","Lift3"], ascending=[False, False], inplace=True)
        return df.head(top_n)

    pairs = pairs_stats()
    triples = triples_stats() if max_k >= 3 else pd.DataFrame()
    return (pd.DataFrame(item_support.items(), columns=["Predictor","Support_1"]).sort_values("Support_1", ascending=False),
            pairs, triples, N)

def community_signatures(sub, communities, label, out_path):
    """
    For each community (list of countries), compute:
      - top predictors by frequency and by summed Points
      - category mix
      - predictors over-represented vs global (simple risk ratio)
    """
    lines = []
    # global baselines
    global_sets = sub.groupby("Country")["Predictor"].apply(lambda s: set(s.tolist()))
    global_support = Counter()
    for s in global_sets:
        for x in s:
            global_support[x] += 1
    G = len(global_sets)

    for i, group in enumerate(communities, start=1):
        gname = f"{label} Community {i}"
        subg = sub[sub["Country"].isin(group)].copy()

        # Frequency and Points
        freq = subg.groupby("Predictor")["Country"].nunique().sort_values(ascending=False)
        pts  = subg.groupby("Predictor")["Points"].sum().sort_values(ascending=False)

        # Category mix
        subg["Category"] = subg["Predictor"].map(PRED_TO_CATEGORY).fillna("Uncategorized")
        cat_mix = subg.groupby("Category")["Country"].nunique().sort_values(ascending=False)

        # Over-representation vs global: RR = P(x|group)/P(x|global)
        group_sets = subg.groupby("Country")["Predictor"].apply(lambda s: set(s.tolist()))
        H = len(group_sets)
        group_support = Counter()
        for s in group_sets:
            for x in s:
                group_support[x] += 1

        rows = []
        for pred, gs in group_support.items():
            ps_group = gs / H if H else 0.0
            ps_global = (global_support[pred] / G) if G else 0.0
            rr = (ps_group / ps_global) if ps_global else np.nan
            rows.append((pred, gs, global_support[pred], ps_group, ps_global, rr))
        rr_df = (pd.DataFrame(rows, columns=["Predictor","GroupSupport","GlobalSupport","P_inGroup","P_Global","RiskRatio"])
                 .sort_values(["RiskRatio","GroupSupport"], ascending=[False, False])
                 .head(15))

        # Write section
        lines.append(f"### {gname}  (n={H} countries)")
        lines.append(f"Countries: {', '.join(group)}\n")
        lines.append("Top predictors by **frequency** (countries containing in top-10):")
        lines.append(rr_df[["Predictor","GroupSupport","GlobalSupport","RiskRatio"]].to_string(index=False))
        lines.append("\nTop predictors by **points** (rank-weighted):")
        pts_df = pts.reset_index().rename(columns={"index":"Predictor","Points":"TotalPoints"}).head(15)
        lines.append(pts_df.to_string(index=False))
        lines.append("\nCategory mix (countries with ≥1 predictor from category in their top-10):")
        lines.append(cat_mix.to_string())
        lines.append("\n")

    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def write_markdown_report(dv, sub, out_path, top1_counts, rank1_lists, in_top10, top10_lists,
                          pairs, triples, jacc_comms, cos_comms):
    lines = []
    lines.append(f"# {dv} — Top-10 Predictor Patterns\n")

    # 1) “Who had what as #1?”
    lines.append("## Predictors that were **#1** and where")
    lines.append(top1_counts.to_string(index=False))
    lines.append("\n### Countries by #1 predictor")
    # Make a compact bullet list
    for _, row in rank1_lists.iterrows():
        pred = row["Predictor"]
        countries = row["Countries_list"]
        lines.append(f"- **{pred}** — {len(countries)} countries: {', '.join(countries)}")
    lines.append("")

    # 2) “Who had what in top-10?”
    lines.append("## Predictors appearing in **top-10** (support across countries)")
    lines.append(in_top10.to_string(index=False))
    lines.append("\n### Countries by top-10 presence")
    for _, row in top10_lists.iterrows():
        pred = row["Predictor"]
        countries = row["Countries_list"]
        lines.append(f"- **{pred}** — {len(countries)} countries: {', '.join(countries)}")
    lines.append("")

    # 3) Pair & trio patterns
    lines.append("## Frequent **pairs** of predictors (support & lift)")
    if not pairs.empty:
        lines.append(pairs.to_string(index=False))
    else:
        lines.append("_No frequent pairs passing the support threshold._")
    lines.append("")

    lines.append("## Frequent **triples** of predictors (support & lift)")
    if not triples.empty:
        lines.append(triples.to_string(index=False))
    else:
        lines.append("_No frequent triples passing the support threshold._")
    lines.append("")

    # 4) Communities
    lines.append("## Country communities (similar top-10 profiles)")
    lines.append("### Jaccard-based (shared set overlap)")
    for i, comm in enumerate(jacc_comms, start=1):
        lines.append(f"- **Community {i}**: {', '.join(comm)}")
    lines.append("")
    lines.append("### Cosine-based (rank-weighted signature)")
    for i, comm in enumerate(cos_comms, start=1):
        lines.append(f"- **Community {i}**: {', '.join(comm)}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def analyze_outcome(df_all, dv,
                    jacc_thresh=0.5,
                    cos_thresh=0.85,
                    top_pairs=30,
                    top_triples=30,
                    min_support=2):
    sub = df_all[df_all["Type"] == dv].copy()
    if sub.empty:
        print(f"[skip] No rows for {dv}")
        return

    out_path = OUT_DIR / dv
    out_path.mkdir(exist_ok=True)

    # Prep
    sub["Points"] = sub["Rank"].map(POINTS_MAP).astype(int)

    # 1) Rank #1 commonalities 
    top1 = sub[sub["Rank"] == 1][["Country","Predictor"]]
    top1_counts = (top1.groupby("Predictor")["Country"]
                        .nunique()
                        .sort_values(ascending=False)
                        .rename("Countries_with_Rank1")
                        .reset_index())
    top1_counts.to_csv(out_path / "01_rank1_counts.csv", index=False)
    rank1_lists = (top1.groupby("Predictor")["Country"]
                        .apply(lambda s: sorted(s.unique()))
                        .reset_index()
                        .rename(columns={"Country":"Countries_list"}))
    rank1_lists.to_csv(out_path / "02_rank1_country_lists.csv", index=False)

    # 2) Top-10 presence 
    in_top10 = (sub.groupby("Predictor")["Country"]
                    .nunique()
                    .sort_values(ascending=False)
                    .rename("Countries_in_Top10")
                    .reset_index())
    in_top10.to_csv(out_path / "03_in_top10_counts.csv", index=False)
    top10_lists = (sub.groupby("Predictor")["Country"]
                      .apply(lambda s: sorted(s.unique()))
                      .reset_index()
                      .rename(columns={"Country":"Countries_list"}))
    top10_lists.to_csv(out_path / "04_in_top10_country_lists.csv", index=False)

    # 3) Summary w/ ranks, importance, category 
    rank_stats = sub.groupby("Predictor")["Rank"].agg(Avg_Rank="mean", Median_Rank="median").reset_index()
    imp_stats = sub.groupby("Predictor")["Importance"].mean().rename("Mean_Importance").reset_index()
    summary = (top1_counts.merge(in_top10, on="Predictor", how="outer")
                         .merge(rank_stats, on="Predictor", how="outer")
                         .merge(imp_stats, on="Predictor", how="outer"))
    summary["Category"] = summary["Predictor"].map(PRED_TO_CATEGORY).fillna("Uncategorized")
    summary.sort_values(["Countries_with_Rank1","Countries_in_Top10","Avg_Rank"],
                        ascending=[False,False,True], inplace=True)
    summary.to_csv(out_path / "05_predictor_summary.csv", index=False)

    # 4) Similar-country groupings 
    sets = sub.groupby("Country")["Predictor"].apply(lambda s: set(s.tolist()))
    jacc = jaccard_similarity(sets)
    jacc.to_csv(out_path / "06_country_jaccard_top10.csv")

    jacc_comms = threshold_communities(jacc, jacc_thresh)
    pd.DataFrame({"Community_ID": range(1, len(jacc_comms)+1),
                  "Countries": jacc_comms}).to_csv(out_path / "07_country_communities_by_jaccard.csv", index=False)

    pivot_pts = sub.pivot_table(index="Country", columns="Predictor", values="Points", aggfunc="sum", fill_value=0)
    cos = cosine_similarity_rankpoints(pivot_pts)
    cos.to_csv(out_path / "08_country_cosine_rankpoints.csv")

    cos_comms = threshold_communities(cos, cos_thresh)
    pd.DataFrame({"Community_ID": range(1, len(cos_comms)+1),
                  "Countries": cos_comms}).to_csv(out_path / "09_country_communities_by_cosine.csv", index=False)

    # 5) Category-level patterns per country 
    sub["Category"] = sub["Predictor"].map(PRED_TO_CATEGORY).fillna("Uncategorized")
    cat_counts = (sub.groupby(["Country","Category"])["Predictor"].count()
                    .rename("Top10_Count").reset_index())
    cat_wide = cat_counts.pivot(index="Country", columns="Category", values="Top10_Count").fillna(0).astype(int)
    cat_wide.to_csv(out_path / "10_country_category_counts_in_top10.csv")

    # #1 predictor’s category lists
    top1_cat = sub[sub["Rank"]==1].assign(Category=lambda x: x["Predictor"].map(PRED_TO_CATEGORY).fillna("Uncategorized"))
    top1_cat_lists = (top1_cat.groupby("Category")["Country"]
                           .apply(lambda s: sorted(s.unique()))
                           .reset_index()
                           .rename(columns={"Country":"Countries_list"}))
    top1_cat_lists.to_csv(out_path / "11_rank1_category_country_lists.csv", index=False)

    # 6) Frequent itemsets (pairs & triples) with lift 
    support1, pairs, triples, N = itemset_mining(sets, max_k=3, top_n=max(30, len(sets)//2), min_support=2)
    support1.to_csv(out_path / "12_item_support_1.csv", index=False)
    pairs.to_csv(out_path / "13_frequent_pairs.csv", index=False)
    triples.to_csv(out_path / "14_frequent_triples.csv", index=False)

    # 7) Markdown report (bullets) 
    report_md = out_path / "REPORT.md"
    write_markdown_report(dv, sub, report_md, top1_counts, rank1_lists, in_top10, top10_lists,
                          pairs, triples, jacc_comms, cos_comms)

    # 8) Community signatures (over-represented predictors/categories) 
    community_signatures(sub, jacc_comms, "Jaccard", out_path / "REPORT.md")
    community_signatures(sub, cos_comms, "Cosine",  out_path / "REPORT.md")

    print(f"[{dv}] wrote outputs → {out_path}")

if __name__ == "__main__":
    df_all = load_top10()
    # Quick sanity: ensure essential cols exist
    expected_cols = {"Country","Type","Rank","Predictor","Importance"}
    missing = expected_cols - set(df_all.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSVs: {missing}")

    for dv in OUTCOMES:
        analyze_outcome(df_all, dv,
                        jacc_thresh=0.5,   # overlap threshold for set-based communities
                        cos_thresh=0.85,   # similarity threshold for rank-weighted communities
                        top_pairs=30,
                        top_triples=30,
                        min_support=2)

    print(f"Done. See {OUT_DIR.resolve()}")
