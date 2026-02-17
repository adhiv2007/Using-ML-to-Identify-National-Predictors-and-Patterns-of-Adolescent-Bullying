from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat
from preprocessingFunctions import LFM_data_from_raw, process_data_beingbullied, process_data_bullying_types
from predictorAnalysisFunctions import top10_beingbullied, top10_bullying_types

STU_PATH = Path("../data/raw/Student Data.sav")
SCH_PATH = Path("../data/raw/School Data.sav")

PROC_DIR   = Path("../data/processed")
RESULT_DIR = Path("../results")
PROC_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTORS = [
    #Individual-level Predictors
    "ST004D01T", #Gender
    "AGE", #Age
    "GRADE", #Grade
    "BSMJ", #Expected Occupational Status
    "JOYREAD", #Joy of Reading
    "SCREADCOMP", #Reading Self-Concept: Competence
    "SCREADDIFF", #Reading Self-Concept: Difficulty
    "COMPETE", #Competitiveness
    "WORKMAST", #Work Mastery Orientation
    "GFOFAIL", # General Fear of Failure
    "EUDMO", #Sense of Meaning in Life (Eudaimonia)  
    "RESILIENCE", #Resilience
    "MASTGOAL", #Mastery Goal Orientation
    "ST185Q01HA", #Does life has meaning/purpose 
    "ST184Q01HA", #Growth Mindset
    "SWBP", #Well Being 
    "PV1MATH", #Math Performance
    "PV1READ", #Reading Performance
    "PV1SCIE", #Science Performance
    "ST207Q01HA", #Irritation feeling when other students are bullied
    "ST207Q02HA", #Feeling of whether it's good or not to help students who can't defend themselves
    "ST207Q03HA", #Is it wrong to join in bullying others
    "ST207Q04HA", #How you feel when you see other students being bullied
    "ST207Q05HA", #Liking when you see other students being bullied

    #Proximal-level Predictors
    "REPEAT", #Grade Repetition History
    "UNDREM", #Meta-cognition: Understanding & Remembering 
    "METASUM", #Meta-cognition: Summarizing
    "METASPAM", #Meta-cognition: Assessing Credibility

    #Microsystem-Level Factors (Family, Peers, & School CLimate)
    "EMOSUPS", #Parental Emotional Support
    "DURECEC", #Duration in Early Childhood Education and Care
    "BELONG", #School Belonging
    "PERCOMP", #Perceived School Competitiveness 
    "PERCOOP", #Perceived School Cooperation
    "ATTLNACT", #Attitudes Towards Learning Activities
    "DISCLIMA", #Disciplinary Climate (Language Lessons)
    "TEACHSUP", #Teacher Support (Language Lessons)
    "DIRINS", #Teacher-Directed Instruction 
    "PERFEED", #Perceived Feedback from Teachers
    "STIMREAD", #Teacher's Stimulation of Reading Engagement
    "ADAPTIVITY", #Adapation of Instruction
    "TEACHINT", #Perceived Teacher Interest 
    "ST206Q01HA", #How students value cooperation 
    "PA006Q09TA" #School Climate

    #Macrosystem/Exosystem-level Predictors
    "ESCS", #Family Socioeconomic Status(Index)
    "EDUSHORT", #Shortage of Educational Resources
    "RATCMP1", #Number of Computers per Student
    "RATCMP2", #Percentage of Computers Connected to the Internet                    
]

RAW_TYPE_ITEMS = ["ST038Q04NA","ST038Q05NA","ST038Q06NA","ST038Q07NA","ST038Q08NA"]
DV_TYPES = ["BULLY_VERBAL","BULLY_THREAT","BULLY_PHYSICAL","BULLY_RELATIONAL"]

error_log = []
def log_error(iso3, stage, err):
    msg = f"{iso3} | {stage} | {type(err).__name__}: {err}"
    print("!!", msg)
    error_log.append({"country": iso3, "stage": stage, "error": str(err)})


print("Preloading raw PISA files once...")
stu_raw, _ = pyreadstat.read_sav(STU_PATH, apply_value_formats=False)
sch_raw, _ = pyreadstat.read_sav(SCH_PATH, apply_value_formats=False)
print("Preload complete.")

# Figure out all 2018 ISO3 country codes directly from the data
ALL_ISO3 = sorted(stu_raw["CNT"].dropna().unique().tolist())

# Containers for outputs
rows_being = []
rows_types = []
prev_rows  = []

for iso3 in ALL_ISO3:
    try:
        print(f"\nPreprocessing PISA data for {iso3}...")
        df_merged = LFM_data_from_raw(stu_raw, sch_raw, iso3)

        try:
            df_being = process_data_beingbullied(df_merged, PREDICTORS, dv="BEINGBULLIED")
            # save processed per-country
            outp = PROC_DIR / f"{iso3}_DVBEING_data.pkl"
            df_being.to_pickle(outp)
            print(f"Saved processed BEINGBULLIED → {outp} (shape: {df_being.shape})")

            # prevalence proxy = mean of DV (works if 0/1 or standardized index)
            prev_rows.append({"Country": iso3, "Type": "BEINGBULLIED", "Prevalence": float(np.nanmean(df_being["BEINGBULLIED"]))})

            # top10
            try:
                top10 = top10_beingbullied(df_being, iso3)
                rows_being.append(top10)
                # pretty print
                print(f"Top 10 XGB predictors of BEINGBULLIED ( {iso3} ):")
                for _, r in top10.iterrows():
                    print(f" {int(r['Rank']):2d}. {r['Feature']:<12s}: {r['Importance']:.4f}")
            except Exception as e:
                log_error(iso3, "top10_beingbullied", e)

        except Exception as e:
            log_error(iso3, "process_data_beingbullied", e)

        try:
            df_types = process_data_bullying_types(df_merged, PREDICTORS, RAW_TYPE_ITEMS)
            outp2 = PROC_DIR / f"{iso3}_bully_types_data.pkl"
            df_types.to_pickle(outp2)
            print(f"Saved processed BullyTypes → {outp2} (shape: {df_types.shape})")

            # Prevalence for each binary type (mean of 0/1)
            for dv in [c for c in DV_TYPES if c in df_types.columns]:
                prev_rows.append({"Country": iso3, "Type": dv, "Prevalence": float(np.nanmean(df_types[dv]))})

            # top10 per type
            try:
                top_types = top10_bullying_types(df_types, iso3, DV_TYPES)
                rows_types.append(top_types)
                # optional pretty print (short)
                if not top_types.empty:
                    for dv in top_types["Type"].unique():
                        sub = top_types[top_types["Type"]==dv]
                        auc = sub["AUC"].iloc[0]
                        print(f"{iso3} — {dv.replace('BULLY_','').title()} (ROC-AUC {auc:.3f})")
                        for _, r in sub.iterrows():
                            print(f" {int(r['Rank']):2d}. {r['Feature']:<12s}: {r['Importance']:.4f}")
            except Exception as e:
                log_error(iso3, "top10_bullying_types", e)

        except Exception as e:
            log_error(iso3, "process_data_bullying_types", e)

    except Exception as e:
        log_error(iso3, "LFM_data_from_raw", e)

df_being_all = pd.concat(rows_being, ignore_index=True) if rows_being else pd.DataFrame(columns=["Country","Type","Rank","Feature","Importance"])
df_types_all = pd.concat(rows_types, ignore_index=True) if rows_types else pd.DataFrame(columns=["Country","Type","AUC","Rank","Feature","Importance"])
df_prev      = pd.DataFrame(prev_rows)

df_being_all.to_csv(RESULT_DIR / "beingbullied_top10.csv", index=False)
df_types_all.to_csv(RESULT_DIR / "bullying_types_top10.csv", index=False)
df_prev.to_csv(RESULT_DIR / "bullying_prevalence.csv", index=False)

pd.DataFrame(error_log).to_csv(RESULT_DIR / "errors_log.csv", index=False)
print("\nSaved:")
print(" - results/beingbullied_top10.csv")
print(" - results/bullying_types_top10.csv")
print(" - results/bullying_prevalence.csv")
print(" - results/errors_log.csv")

def add_points(df_long, has_auc=False):
    if df_long.empty: 
        return df_long
    pts_map = {1:10,2:9,3:8,4:7,5:6,6:5,7:4,8:3,9:2,10:1}
    out = df_long.copy()
    out["Points"] = out["Rank"].map(pts_map).fillna(0).astype(int)
    return out

being_pts = add_points(df_being_all, has_auc=False)
types_pts = add_points(df_types_all, has_auc=True)

# Global totals by feature (separately for BEINGBULLIED and for each Type)
agg_being = (being_pts.groupby("Feature", as_index=False)["Points"].sum()
                        .sort_values("Points", ascending=False))
agg_types = (types_pts.groupby(["Type","Feature"], as_index=False)["Points"].sum()
                        .sort_values(["Type","Points"], ascending=[True, False]))

agg_being.to_csv(RESULT_DIR / "aggregate_points_BEINGBULLIED.csv", index=False)
agg_types.to_csv(RESULT_DIR / "aggregate_points_byType.csv", index=False)

print("\nSaved:")
print(" - results/aggregate_points_BEINGBULLIED.csv")
print(" - results/aggregate_points_byType.csv")