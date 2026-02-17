import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def LFM_data_from_raw(stu_raw: pd.DataFrame, sch_raw: pd.DataFrame, country: str) -> pd.DataFrame:
    print("Loading student and school data...")
    print("Filtering student and school data for country...")
    stu = stu_raw[stu_raw["CNT"] == country].copy()
    sch = sch_raw[sch_raw["CNT"] == country].copy()

    keep_sch = [c for c in ["CNTSCHID","RATCMP1","RATCMP2","EDUSHORT"] if c in sch.columns]
    if keep_sch:
        sch = sch[["CNTSCHID"] + keep_sch[1:]] if "CNTSCHID" in sch.columns else sch

    print("Merging student + school data...")
    if "CNTSCHID" in stu.columns and "CNTSCHID" in sch.columns:
        df = stu.merge(sch, on="CNTSCHID", how="left")
    else:
        df = stu.copy()  # fall back if CNTSCHID missing
    return df

def process_data_beingbullied(df: pd.DataFrame, predictors: list, dv: str = "BEINGBULLIED") -> pd.DataFrame:
    print("Keeping predictors + DV, encoding gender...")
    cols = [c for c in predictors if c in df.columns]
    if dv not in df.columns:
        raise ValueError(f"DV '{dv}' not in dataframe.")
    keep = list(dict.fromkeys(cols + [dv] + (["CNTSCHID"] if "CNTSCHID" in df.columns else [])))
    df = df[keep].copy()

    # encode gender
    if "ST004D01T" in df.columns:
        df["ST004D01T"] = df["ST004D01T"].map({1:0, 2:1})

    # drop rows with missing DV
    df = df.dropna(subset=[dv])

    # X / y
    X = df.drop(columns=[dv, "CNTSCHID"], errors="ignore")
    print("Dropping columns with all missing values...")
    X = X.dropna(axis=1, how="all")

    print("Imputing missing values and scaling features...")
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    X_scl   = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns, index=X.index)

    X_scl[dv] = df[dv].values
    return X_scl

def process_data_bullying_types(df: pd.DataFrame, predictors: list, RAW_ITEMS) -> pd.DataFrame:
    print("Keeping predictors + raw bullying items, encoding gender...")
    keep_pred = [c for c in predictors if c in df.columns]
    keep_raw  = [c for c in RAW_ITEMS if c in df.columns]
    keep_cols = list(dict.fromkeys(keep_pred + keep_raw + (["CNTSCHID"] if "CNTSCHID" in df.columns else [])))
    df = df[keep_cols].copy()

    if "ST004D01T" in df.columns:
        df["ST004D01T"] = df["ST004D01T"].map({1:0, 2:1})

    print("Creating binary DVs for 4 bullying types...")
    def _bin4(s: pd.Series) -> pd.Series:
        if s.dropna().max() is not None and s.dropna().max() >= 4:
            return s.isin([3,4]).astype(float)
        return s.isin([2,3]).astype(float)

    dv_map = {}
    if "ST038Q04NA" in df.columns: dv_map["BULLY_VERBAL"]     = _bin4(df["ST038Q04NA"])
    if "ST038Q05NA" in df.columns: dv_map["BULLY_THREAT"]     = _bin4(df["ST038Q05NA"])
    if "ST038Q08NA" in df.columns: dv_map["BULLY_RELATIONAL"] = _bin4(df["ST038Q08NA"])
    if {"ST038Q06NA","ST038Q07NA"}.issubset(df.columns):
        dv_map["BULLY_PHYSICAL"] = (_bin4(df["ST038Q06NA"]).astype(bool) | _bin4(df["ST038Q07NA"]).astype(bool)).astype(float)

    if not dv_map:
        raise ValueError("No bullying type items found to create DVs.")

    dv_df = pd.DataFrame(dv_map, index=df.index)

    print("Dropping columns with all missing values...")
    X = df.drop(columns=RAW_ITEMS + ["CNTSCHID"], errors="ignore").dropna(axis=1, how="all")

    print("Imputing missing values and scaling features...")
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    X_scl   = pd.DataFrame(scaler.fit_transform(X_imp), columns=X.columns, index=X.index)

    final_df = pd.concat([X_scl, dv_df], axis=1)
    return final_df
