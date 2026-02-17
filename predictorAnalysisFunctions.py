import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import roc_auc_score

def best_xgb_reg():
    return XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=4, reg_alpha=0.6, min_child_weight=5,
        objective="reg:squarederror", gamma=0,
        random_state=42, verbosity=0, n_jobs=-1
    )

def best_xgb_clf():
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=4, reg_alpha=0.6, min_child_weight=5,
        objective="binary:logistic", gamma=0, eval_metric="auc",
        random_state=42, verbosity=0, n_jobs=-1
    )

def top10_beingbullied(df_proc: pd.DataFrame, country_label: str) -> pd.DataFrame:
    dv = "BEINGBULLIED"
    if dv not in df_proc.columns:
        raise ValueError(f"{country_label}: '{dv}' not found after processing.")

    X = df_proc.drop(columns=[dv], errors="ignore")
    y = df_proc[dv].values

    # Need at least some non-NaN variance to train
    if len(X.columns) == 0 or np.isclose(X.var(numeric_only=True), 0).all():
        raise ValueError(f"{country_label}: no usable predictors after cleaning.")

    model = best_xgb_reg().fit(X, y)
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

    out = (imp.reset_index()
              .rename(columns={"index":"Feature", 0:"Importance"})
              .assign(Country=country_label, Type="BEINGBULLIED"))
    out["Rank"] = np.arange(1, len(out)+1)
    return out[["Country","Type","Rank","Feature","Importance"]]


def top10_bullying_types(df_proc: pd.DataFrame, country_label: str, DVLIST) -> pd.DataFrame:
    X = df_proc.drop(columns=DVLIST, errors="ignore")
    if X.shape[1] == 0:
        raise ValueError(f"{country_label}: no predictors available.")

    rows = []
    for dv in [c for c in DVLIST if c in df_proc.columns]:
        y = df_proc[dv].dropna().astype(int)
        mask = df_proc[dv].notna()
        Xy = X.loc[mask]

        # skip if only one class
        if y.nunique() < 2:
            print(f"{country_label} — {dv.replace('BULLY_','').title()}: skipped (only one class).")
            continue

        model = best_xgb_clf().fit(Xy, y)
        auc = roc_auc_score(y, model.predict_proba(Xy)[:,1])

        imp = (pd.Series(model.feature_importances_, index=Xy.columns)
                 .sort_values(ascending=False)
                 .head(10))

        tmp = (imp.reset_index()
                 .rename(columns={"index":"Feature", 0:"Importance"})
                 .assign(Country=country_label, Type=dv, AUC=auc))
        tmp["Rank"] = np.arange(1, len(tmp)+1)
        rows.append(tmp[["Country","Type","AUC","Rank","Feature","Importance"]])

    if not rows:
        return pd.DataFrame(columns=["Country","Type","AUC","Rank","Feature","Importance"])
    return pd.concat(rows, ignore_index=True)
