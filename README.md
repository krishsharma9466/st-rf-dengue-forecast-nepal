"""
Dengue Spatio-Temporal Forecast Dashboard - Nepal
ST-RF + Holt-Winters Blend | 2016-2028
"""

import datetime
import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")


# File paths
BASE_PATH = r"D:\data"

FILES = {
    "dengue": os.path.join(BASE_PATH, "dengue.xlsx"),
    "ndvi":   os.path.join(BASE_PATH, "NDVI.xlsx"),
    "ndwi":   os.path.join(BASE_PATH, "NDWI.xlsx"),
    "lst":    os.path.join(BASE_PATH, "LST.xlsx"),
    "pop":    os.path.join(BASE_PATH, "pop.xlsx"),
}

YEARS  = list(range(2016, 2026))
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

MONTH_NUM_MAP  = {m: i + 1 for i, m in enumerate(MONTHS)}
MONTH_NAME_MAP = {i + 1: m for i, m in enumerate(MONTHS)}
MERGE_KEYS     = ["District", "Municipality", "Year", "Month_Num"]

CV_FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]

RISK_THRESHOLDS = [
    (5,  "Low to Moderate",  "#FFC107"),
    (20, "Moderate to High", "#FF5722"),
    (50, "High to Severe",   "#B71C1C"),
]


# Load one excel file in long format and return a clean dataframe
def load_long_format(filepath, value_name):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_excel(filepath, dtype=str)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").reset_index(drop=True)

    for col in df.columns:
        if col.lower().strip() in ("district", "municipality"):
            df[col] = df[col].ffill()

    known_lower = {"district", "municipality", "year", "month"}
    rename_map  = {}
    value_col   = None

    for col in df.columns:
        cl = col.lower().strip()
        if   cl == "district":     rename_map[col] = "District"
        elif cl == "municipality": rename_map[col] = "Municipality"
        elif cl == "year":         rename_map[col] = "Year"
        elif cl == "month":        rename_map[col] = "Month"
        elif cl not in known_lower:
            if value_col is None:
                value_col = col
                rename_map[col] = value_name

    if value_col is None:
        raise ValueError(f"{value_name}: no value column found. Columns: {df.columns.tolist()}")

    df = df.rename(columns=rename_map)

    required = {"District", "Municipality", "Year", "Month", value_name}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"{value_name}: missing columns {missing}")

    df["District"]     = df["District"].astype(str).str.strip()
    df["Municipality"] = df["Municipality"].astype(str).str.strip()
    df["Month"]        = df["Month"].astype(str).str.strip().str.capitalize()
    df["Month_Num"]    = df["Month"].map(MONTH_NUM_MAP)
    df["Year"]         = pd.to_numeric(df["Year"],     errors="coerce")
    df[value_name]     = pd.to_numeric(df[value_name], errors="coerce")

    bad = df["Month_Num"].isna() | df["Year"].isna()
    if bad.sum():
        print(f"  {value_name}: dropping {bad.sum():,} rows with unparseable year or month")

    df = df.dropna(subset=["Year", "Month_Num"]).reset_index(drop=True)
    df["Year"]      = df["Year"].astype(int)
    df["Month_Num"] = df["Month_Num"].astype(int)

    return df[["District", "Municipality", "Year", "Month_Num", value_name]]


print("Loading data files")

df_dengue = load_long_format(FILES["dengue"], "Dengue")
df_ndvi   = load_long_format(FILES["ndvi"],   "NDVI")
df_ndwi   = load_long_format(FILES["ndwi"],   "NDWI")
df_lst    = load_long_format(FILES["lst"],     "LST")

for label, d in [("Dengue", df_dengue), ("NDVI", df_ndvi), ("NDWI", df_ndwi), ("LST", df_lst)]:
    print(f"  {label}: {len(d):,} rows, {d['Municipality'].nunique()} municipalities")

HAS_POP = os.path.exists(FILES["pop"])
if HAS_POP:
    df_pop = pd.read_excel(FILES["pop"], dtype=str)
    df_pop.columns = df_pop.columns.str.strip()
    df_pop["Municipality"] = df_pop["Municipality"].astype(str).str.strip()
    df_pop["Population"]   = pd.to_numeric(df_pop["Population"], errors="coerce")
    df_pop = df_pop.dropna(subset=["Population"])
    print(f"  Pop: {len(df_pop):,} rows, {df_pop['Municipality'].nunique()} municipalities")
else:
    df_pop = None
    print("  pop.xlsx not found - incidence mode disabled")


# Add zero-case rows for municipalities that appear in env files but not in dengue file
print("\nAligning municipality coverage")

env_munis        = df_ndvi[["District", "Municipality"]].drop_duplicates().reset_index(drop=True)
dengue_munis_set = set(df_dengue["Municipality"])
missing_munis    = env_munis[~env_munis["Municipality"].isin(dengue_munis_set)]

if len(missing_munis) > 0:
    print(f"  Adding {len(missing_munis)} zero-case municipalities from env files")
    ym_grid = df_ndvi[["Year", "Month_Num"]].drop_duplicates().copy()
    ym_grid["Year"]      = ym_grid["Year"].astype(int)
    ym_grid["Month_Num"] = ym_grid["Month_Num"].astype(int)
    zero_rows            = missing_munis.merge(ym_grid, how="cross")
    zero_rows["Dengue"]  = 0
    df_dengue            = pd.concat([df_dengue, zero_rows], ignore_index=True)
else:
    print("  All env municipalities already present in dengue file")

common_munis = (
    set(df_dengue["Municipality"]) & set(df_ndvi["Municipality"])
    & set(df_ndwi["Municipality"]) & set(df_lst["Municipality"])
)
print(f"  Municipalities common to all four files: {len(common_munis)}")


# Merge dengue with environmental covariates
print("\nMerging datasets")

df = df_dengue.copy()
for other in [df_ndvi, df_ndwi, df_lst]:
    df = df.merge(other, on=MERGE_KEYS, how="left")

print(f"  Merged: {df.shape[0]:,} rows, {df['Municipality'].nunique()} municipalities, {df['District'].nunique()} districts")
print(f"  Missing env data: NDVI={df['NDVI'].isna().mean():.1%}  NDWI={df['NDWI'].isna().mean():.1%}  LST={df['LST'].isna().mean():.1%}")


# Build df_all (full coverage with zeros) and df_clean (complete cases only)
df_all           = df.copy()
df_all["Dengue"] = df_all["Dengue"].fillna(0)

if HAS_POP and df_pop is not None:
    df_all = df_all.merge(df_pop[["Municipality", "Population"]], on="Municipality", how="left")
    df_all["Population"] = df_all["Population"].fillna(df_all["Population"].median())
else:
    df_all["Population"] = np.nan

dist_to_munis_full = df_all.groupby("District")["Municipality"].unique().apply(sorted).to_dict()
all_munis_full     = sorted(df_all["Municipality"].unique().tolist())
all_districts_full = sorted(df_all["District"].unique().tolist())

df_clean = df.dropna(subset=["Dengue", "NDVI", "NDWI", "LST"]).copy()
df_clean["Dengue"] = df_clean["Dengue"].fillna(0).clip(lower=0)
df_clean = df_clean.sort_values(["Municipality", "Year", "Month_Num"]).reset_index(drop=True)

print(f"\n  df_all municipalities:   {df_all['Municipality'].nunique()}")
print(f"  df_clean municipalities: {df_clean['Municipality'].nunique()}")
print(f"  df_clean rows: {len(df_clean):,}  (zero cases retained: {(df_clean['Dengue'] == 0).sum():,})")

if df_clean.empty:
    raise RuntimeError("df_clean is empty - check that municipality names match across all files")


# Feature engineering: encode labels, lag features, seasonal signals, neighbour lag
print("\nBuilding features")

le_muni = LabelEncoder()
le_dist = LabelEncoder()
df_clean["Muni_ID"] = le_muni.fit_transform(df_clean["Municipality"])
df_clean["Dist_ID"] = le_dist.fit_transform(df_clean["District"])

muni_mean = df_clean.groupby("Municipality")["Dengue"].mean().rename("Muni_Mean_Dengue")
dist_mean = df_clean.groupby("District")["Dengue"].mean().rename("Dist_Mean_Dengue")
df_clean  = df_clean.merge(muni_mean, on="Municipality", how="left")
df_clean  = df_clean.merge(dist_mean, on="District",     how="left")

year_min = int(df_clean["Year"].min())
df_clean["Time_Index"] = (df_clean["Year"] - year_min) * 12 + df_clean["Month_Num"]
df_clean["Sin_Month"]  = np.sin(2 * np.pi * df_clean["Month_Num"] / 12)
df_clean["Cos_Month"]  = np.cos(2 * np.pi * df_clean["Month_Num"] / 12)

df_clean = df_clean.sort_values(["Municipality", "Year", "Month_Num"]).reset_index(drop=True)
grp = df_clean.groupby("Municipality")["Dengue"]
df_clean["Dengue_Lag1"] = grp.shift(1)
df_clean["Dengue_Lag2"] = grp.shift(2)
df_clean["Rolling3"]    = grp.shift(1).rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)

dist_month_lag1  = df_clean.groupby(["District", "Year", "Month_Num"])["Dengue_Lag1"].mean().rename("_dist_lag1_mean").reset_index()
dist_month_count = df_clean.groupby(["District", "Year", "Month_Num"])["Municipality"].count().rename("_dist_count").reset_index()
df_clean = df_clean.merge(dist_month_lag1,  on=["District", "Year", "Month_Num"], how="left")
df_clean = df_clean.merge(dist_month_count, on=["District", "Year", "Month_Num"], how="left")

df_clean["Neighbor_Lag1"] = (
    (df_clean["_dist_lag1_mean"] * df_clean["_dist_count"] - df_clean["Dengue_Lag1"].fillna(0))
    / (df_clean["_dist_count"] - 1).clip(lower=1)
).fillna(0)

df_clean = df_clean.drop(columns=["_dist_lag1_mean", "_dist_count"])
df_clean = df_clean.dropna(subset=["Dengue_Lag1", "Dengue_Lag2", "Rolling3"]).reset_index(drop=True)

FEATURES = [
    "Muni_ID", "Dist_ID", "Muni_Mean_Dengue", "Dist_Mean_Dengue",
    "Time_Index", "Sin_Month", "Cos_Month",
    "NDVI", "NDWI", "LST",
    "Dengue_Lag1", "Dengue_Lag2", "Rolling3",
    "Neighbor_Lag1",
]

X      = df_clean[FEATURES].values
y_log  = np.log1p(df_clean["Dengue"].values)
y_orig = df_clean["Dengue"].values
print(f"  {len(FEATURES)} features, {len(X):,} training samples")


# Walk-forward cross-validation: train on years up to cutoff, test on next year
print("\nWalk-forward cross-validation")

cv_records = []
for train_end, test_year in CV_FOLDS:
    train_mask = df_clean["Year"] <= train_end
    test_mask  = df_clean["Year"] == test_year

    X_tr, y_tr = X[train_mask], y_log[train_mask]
    X_te        = X[test_mask]
    y_te_orig   = y_orig[test_mask]

    if len(X_tr) == 0 or len(X_te) == 0:
        print(f"  Fold {train_end} to {test_year}: not enough data, skipped")
        continue

    rf_cv = RandomForestRegressor(
        n_estimators=200, max_depth=12,
        min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", bootstrap=True,
        random_state=42, n_jobs=-1,
    )
    rf_cv.fit(X_tr, y_tr)
    pred_log  = rf_cv.predict(X_te)
    pred_orig = np.expm1(pred_log)

    mae  = mean_absolute_error(y_te_orig, pred_orig)
    rmse = np.sqrt(mean_squared_error(y_te_orig, pred_orig))
    lmae = mean_absolute_error(np.log1p(y_te_orig), pred_log)

    cv_records.append({
        "Train up to": train_end,
        "Test year":   test_year,
        "MAE":         round(mae,  2),
        "RMSE":        round(rmse, 2),
        "Log-MAE":     round(lmae, 4),
        "N_test":      int(test_mask.sum()),
    })
    print(f"  Fold {train_end} to {test_year}: MAE={mae:.1f}  RMSE={rmse:.1f}  Log-MAE={lmae:.4f}  (n={test_mask.sum():,})")

df_cv = pd.DataFrame(cv_records)
if len(df_cv) > 0:
    print(f"\n  Mean Log-MAE: {df_cv['Log-MAE'].mean():.4f}")
    print(f"  Mean MAE:     {df_cv['MAE'].mean():.2f} cases")
    print(f"  Mean RMSE:    {df_cv['RMSE'].mean():.2f} cases")


# Train final random forest on all available data
print("\nTraining final random forest")

rf = RandomForestRegressor(
    n_estimators=500, max_depth=12,
    min_samples_split=4, min_samples_leaf=2,
    max_features="sqrt", bootstrap=True,
    random_state=42, n_jobs=-1,
)
rf.fit(X, y_log)

print("  Computing per-tree historical predictions for prediction intervals")
tree_preds_hist = np.expm1(
    np.clip(np.stack([tree.predict(X) for tree in rf.estimators_], axis=0), 0, None)
)

df_clean["Fitted"] = np.median(tree_preds_hist, axis=0)

r2_log = rf.score(X, y_log)
r2_raw = np.corrcoef(df_clean["Fitted"].values, y_orig)[0, 1] ** 2
print(f"  Done. R2(log)={r2_log:.4f}  R2(raw)={r2_raw:.4f}")

imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("  Top 5 features:")
for feat, val in imp.head(5).items():
    print(f"    {feat}: {val:.4f}")


# Fit Holt-Winters on the national monthly total to blend with RF forecasts
print("\nFitting Holt-Winters on national series")

monthly_national = (
    df.groupby(["Year", "Month_Num"])
    .agg(Dengue=("Dengue", "sum"))
    .reset_index()
    .sort_values(["Year", "Month_Num"])
    .reset_index(drop=True)
)
monthly_national["Dengue"] = monthly_national["Dengue"].fillna(0).clip(lower=1)

hw_model = ExponentialSmoothing(
    monthly_national["Dengue"].values.astype(float),
    trend="add", seasonal="add", seasonal_periods=12,
    initialization_method="estimated",
)
hw_fit          = hw_model.fit(optimized=True)
hw_forecast_all = np.clip(hw_fit.forecast(36), 1, None)
print(f"  HW AIC: {hw_fit.aic:.2f}")


# Recursively forecast each municipality for 2026-2028 using lagged predictions as inputs
print("\nRunning recursive municipality forecast 2026-2028")

muni_profiles = (
    df_clean.groupby("Municipality")
    .agg(
        District         =("District",         "first"),
        Dist_ID          =("Dist_ID",          "first"),
        Muni_ID          =("Muni_ID",          "first"),
        Muni_Mean_Dengue =("Muni_Mean_Dengue", "first"),
        Dist_Mean_Dengue =("Dist_Mean_Dengue", "first"),
    )
    .reset_index()
)
N_TREES = len(rf.estimators_)

df_recent     = df_clean[df_clean["Year"] >= 2022].copy()
env_clim      = df_recent.groupby(["Municipality", "Month_Num"])[["NDVI", "NDWI", "LST"]].mean().reset_index()
env_muni_mean = (
    df_clean.groupby("Municipality")[["NDVI", "NDWI", "LST"]]
    .mean().reset_index()
    .rename(columns={"NDVI": "NDVI_m", "NDWI": "NDWI_m", "LST": "LST_m"})
)
global_env = {
    "NDVI": float(df_clean["NDVI"].mean()),
    "NDWI": float(df_clean["NDWI"].mean()),
    "LST":  float(df_clean["LST"].mean()),
}

seed_lags = (
    df_clean.sort_values(["Municipality", "Year", "Month_Num"])
    .groupby("Municipality").tail(3)
    .groupby("Municipality")
    .agg(
        lag1  =("Dengue", lambda x: float(x.iloc[-1])),
        lag2  =("Dengue", lambda x: float(x.iloc[-2]) if len(x) >= 2 else float(x.iloc[-1])),
        roll3 =("Dengue", "mean"),
    )
    .reset_index()
)
muni_lags   = seed_lags.set_index("Municipality")[["lag1", "lag2", "roll3"]].to_dict("index")
default_lag = {"lag1": 0.0, "lag2": 0.0, "roll3": 0.0}

muni_to_district = {row["Municipality"]: row["District"] for _, row in muni_profiles.iterrows()}

last_time_idx         = int(df_clean["Time_Index"].max())
muni_forecast_records = []

for year in [2026, 2027, 2028]:
    t0 = datetime.datetime.now()
    for month in range(1, 13):
        step  = (year - 2026) * 12 + month
        t_idx = last_time_idx + step
        sin_m = float(np.sin(2 * np.pi * month / 12))
        cos_m = float(np.cos(2 * np.pi * month / 12))

        env_month = env_clim[env_clim["Month_Num"] == month].set_index("Municipality")
        env_muni  = env_muni_mean.set_index("Municipality")

        dist_lag1_sum   = {}
        dist_lag1_count = {}
        for muni, lags in muni_lags.items():
            d = muni_to_district.get(muni, "")
            dist_lag1_sum[d]   = dist_lag1_sum.get(d, 0.0) + lags["lag1"]
            dist_lag1_count[d] = dist_lag1_count.get(d, 0)  + 1

        rows, munis_ordered = [], []
        for _, mr in muni_profiles.iterrows():
            muni = mr["Municipality"]
            dist = mr["District"]
            lags = muni_lags.get(muni, default_lag)

            d_sum         = dist_lag1_sum.get(dist, 0.0)
            d_count       = dist_lag1_count.get(dist, 1)
            neighbor_lag1 = (d_sum - lags["lag1"]) / max(d_count - 1, 1)

            if muni in env_month.index:
                ndvi = float(env_month.loc[muni, "NDVI"])
                ndwi = float(env_month.loc[muni, "NDWI"])
                lst  = float(env_month.loc[muni, "LST"])
            elif muni in env_muni.index:
                ndvi = float(env_muni.loc[muni, "NDVI_m"])
                ndwi = float(env_muni.loc[muni, "NDWI_m"])
                lst  = float(env_muni.loc[muni, "LST_m"])
            else:
                ndvi, ndwi, lst = global_env["NDVI"], global_env["NDWI"], global_env["LST"]

            rows.append([
                mr["Muni_ID"], mr["Dist_ID"],
                mr["Muni_Mean_Dengue"], mr["Dist_Mean_Dengue"],
                t_idx, sin_m, cos_m,
                ndvi, ndwi, lst,
                lags["lag1"], lags["lag2"], lags["roll3"],
                neighbor_lag1,
            ])
            munis_ordered.append(muni)

        X_step = np.array(rows, dtype=np.float64)
        all_tree_preds = np.expm1(
            np.clip(np.stack([tree.predict(X_step) for tree in rf.estimators_], axis=0), 0, None)
        )

        step_preds = {}
        for i, muni in enumerate(munis_ordered):
            tree_col  = all_tree_preds[:, i]
            muni_pred = float(np.median(tree_col))
            step_preds[muni] = {"median": max(muni_pred, 0.0), "trees": tree_col}
            lags = muni_lags.get(muni, default_lag)
            muni_lags[muni] = {
                "lag1":  muni_pred,
                "lag2":  lags["lag1"],
                "roll3": (lags["lag1"] + lags["lag2"] + muni_pred) / 3,
            }

        muni_forecast_records.append({"Year": year, "Month_Num": month, "muni_preds": step_preds})

    elapsed = (datetime.datetime.now() - t0).seconds
    print(f"  {year} done ({elapsed}s)")

modelled_munis = set(muni_profiles["Municipality"].tolist())
print(f"\n  Forecast complete: {len(modelled_munis)} municipalities, 36 months")


# Convert year/month columns to a proper datetime column
def _make_date_col(df):
    return pd.to_datetime(dict(year=df["Year"].astype(int), month=df["Month_Num"].astype(int), day=1))


# Aggregate observed and fitted values across a list of municipalities
def build_historical_df(muni_list):
    obs = (
        df_all[df_all["Municipality"].isin(muni_list)]
        .groupby(["Year", "Month_Num"])
        .agg(Dengue=("Dengue", "sum"))
        .reset_index()
        .sort_values(["Year", "Month_Num"])
        .reset_index(drop=True)
    )

    active = [m for m in muni_list if m in modelled_munis]
    if not active:
        for col in ["Fitted", "Upper80", "Lower80", "Upper50", "Lower50"]:
            obs[col] = np.nan
        obs["Date"] = _make_date_col(obs)
        return obs

    mask      = df_clean["Municipality"].isin(active)
    sub_pos   = np.where(mask)[0]
    sub_clean = df_clean.loc[mask, ["Year", "Month_Num"]].copy()
    sub_clean["_pos"] = sub_pos
    pos_lookup = {int(gi): li for li, gi in enumerate(sub_pos)}
    sub_trees  = tree_preds_hist[:, sub_pos]

    fitted_rows = []
    for (yr, mn), grp in sub_clean.groupby(["Year", "Month_Num"]):
        local_positions = [pos_lookup[int(p)] for p in grp["_pos"]]
        tree_sum = np.clip(sub_trees[:, local_positions].sum(axis=1), 0, None)
        fitted_rows.append({
            "Year":      int(yr),
            "Month_Num": int(mn),
            "Fitted":    float(np.median(tree_sum)),
            "Upper80":   float(np.percentile(tree_sum, 90)),
            "Lower80":   float(np.percentile(tree_sum, 10)),
            "Upper50":   float(np.percentile(tree_sum, 75)),
            "Lower50":   float(np.percentile(tree_sum, 25)),
        })

    obs = obs.merge(pd.DataFrame(fitted_rows), on=["Year", "Month_Num"], how="left")
    obs["Date"] = _make_date_col(obs)
    return obs


# Aggregate tree-level forecasts and blend with Holt-Winters
def build_forecast_df(muni_list, hw_weight=0.5):
    active = [m for m in muni_list if m in modelled_munis]
    if not active:
        return None

    rows = []
    for i, rec in enumerate(muni_forecast_records):
        tree_agg = np.zeros(N_TREES)
        for muni in active:
            if muni in rec["muni_preds"]:
                tree_agg += rec["muni_preds"][muni]["trees"]
        tree_agg = np.clip(tree_agg, 0, None)

        if hw_weight > 0:
            tree_agg = np.clip((1 - hw_weight) * tree_agg + hw_weight * hw_forecast_all[i], 0, None)

        rows.append({
            "Year":      rec["Year"],
            "Month_Num": rec["Month_Num"],
            "Median":    float(np.median(tree_agg)),
            "Upper80":   float(np.percentile(tree_agg, 90)),
            "Lower80":   float(np.percentile(tree_agg, 10)),
            "Upper50":   float(np.percentile(tree_agg, 75)),
            "Lower50":   float(np.percentile(tree_agg, 25)),
        })

    df_fc = pd.DataFrame(rows)
    df_fc["Date"] = _make_date_col(df_fc)
    return df_fc


# Return total population for a list of municipalities
def get_population_total(muni_list):
    if not HAS_POP or df_pop is None:
        return np.nan
    sub = df_pop[df_pop["Municipality"].isin(muni_list)]
    return float(sub["Population"].sum()) if len(sub) > 0 else np.nan


# Build the main plotly figure for a given set of municipalities
def make_figure(muni_list, chart_title, hw_weight=0.5, show_incidence=False, show_risk_lines=True):

    hist   = build_historical_df(muni_list)
    df_fc  = build_forecast_df(muni_list, hw_weight=hw_weight)
    has_fc = df_fc is not None

    pop_total     = get_population_total(muni_list)
    use_incidence = show_incidence and not np.isnan(pop_total) and pop_total > 0

    def to_inc(val):
        return val / pop_total * 100_000 if use_incidence else val

    y_label = "Incidence per 100 000" if use_incidence else "Total Dengue Cases"

    # Colour constants for observed, fitted, and forecast lines
    C_OBS      = "#1a252f"
    C_FITTED   = "#2471a3"
    C_FORECAST = "#c0392b"

    C_HIST80_FILL = "rgba(36,113,163,0.12)"
    C_HIST80_LINE = "rgba(36,113,163,0.40)"
    C_HIST50_FILL = "rgba(20,80,140,0.70)"
    C_HIST50_LINE = "rgba(20,80,140,1.00)"

    C_FC80_FILL = "rgba(192,57,43,0.15)"
    C_FC80_LINE = "rgba(192,57,43,0.45)"
    C_FC50_FILL = "rgba(150,20,20,0.70)"
    C_FC50_LINE = "rgba(150,20,20,1.00)"

    fig = go.Figure()

    # Shaded background for the forecast zone
    if has_fc:
        fig.add_vrect(
            x0="2026-01-01",
            x1=df_fc["Date"].max().strftime("%Y-%m-%d"),
            fillcolor="rgba(255,235,235,0.50)",
            layer="below", line_width=0,
            annotation_text="Forecast zone",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#c0392b"),
        )

    # Dotted risk threshold lines shown only when population data is available
    if show_risk_lines and has_fc and not np.isnan(pop_total) and pop_total > 0:
        for inc_thresh, label, color in RISK_THRESHOLDS:
            y_thresh = inc_thresh if use_incidence else inc_thresh * pop_total / 100_000
            fig.add_hline(
                y=y_thresh,
                line_dash="dot", line_color=color, line_width=1.5,
                annotation_text=f"  {label}",
                annotation_position="right",
                annotation_font=dict(size=9, color=color),
            )

    hist_has_pi = (
        "Upper80" in hist.columns and hist["Upper80"].notna().any() and
        "Upper50" in hist.columns and hist["Upper50"].notna().any()
    )

    # Historical prediction interval bands: 80% outer then 50% inner on top
    if hist_has_pi:
        for up, lo, fill, bord, name, rank in [
            ("Upper80", "Lower80", C_HIST80_FILL, C_HIST80_LINE, "80% PI - Historical", 8),
            ("Upper50", "Lower50", C_HIST50_FILL, C_HIST50_LINE, "50% PI - Historical", 7),
        ]:
            hh = hist.dropna(subset=[up, lo]).reset_index(drop=True)
            if hh.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pd.concat([hh["Date"], hh["Date"][::-1]], ignore_index=True),
                y=pd.concat([hh[up].apply(to_inc), hh[lo][::-1].apply(to_inc)], ignore_index=True),
                fill="toself", fillcolor=fill,
                line=dict(color=bord, width=0.8),
                hoverinfo="skip", showlegend=True,
                name=name, legendrank=rank,
            ))

    # Forecast prediction interval bands: 80% outer then 50% inner on top
    if has_fc:
        for up, lo, fill, bord, name, rank in [
            ("Upper80", "Lower80", C_FC80_FILL, C_FC80_LINE, "80% PI - Forecast", 4),
            ("Upper50", "Lower50", C_FC50_FILL, C_FC50_LINE, "50% PI - Forecast", 3),
        ]:
            ff = df_fc.dropna(subset=[up, lo]).reset_index(drop=True)
            if ff.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pd.concat([ff["Date"], ff["Date"][::-1]], ignore_index=True),
                y=pd.concat([ff[up].apply(to_inc), ff[lo][::-1].apply(to_inc)], ignore_index=True),
                fill="toself", fillcolor=fill,
                line=dict(color=bord, width=0.8),
                hoverinfo="skip", showlegend=True,
                name=name, legendrank=rank,
            ))

    # Fitted line with clean hover: single date header, observed once, fitted and PI below
    if "Fitted" in hist.columns and hist["Fitted"].notna().any():
        hover_list = []
        for _, r in hist.iterrows():
            month_year = r["Date"].strftime("%B %Y")
            if pd.notna(r.get("Fitted")) and pd.notna(r.get("Upper80")) and pd.notna(r.get("Upper50")):
                hover_list.append(
                    f"<b>{month_year} - {chart_title}</b><br>"
                    f"Observed: {to_inc(r['Dengue']):.0f} {y_label}<br>"
                    f"Fitted: {to_inc(r['Fitted']):.1f}<br>"
                    f"80% PI: {to_inc(r['Lower80']):.1f} to {to_inc(r['Upper80']):.1f}<br>"
                    f"50% PI: {to_inc(r['Lower50']):.1f} to {to_inc(r['Upper50']):.1f}"
                )
            elif pd.notna(r.get("Fitted")):
                hover_list.append(
                    f"<b>{month_year} - {chart_title}</b><br>"
                    f"Fitted: {to_inc(r['Fitted']):.1f}"
                )
            else:
                hover_list.append("")

        fig.add_trace(go.Scatter(
            x=hist["Date"],
            y=hist["Fitted"].apply(to_inc),
            mode="lines",
            line=dict(color=C_FITTED, width=1.8, dash="dash"),
            name="ST-RF Fitted (median)",
            legendrank=2,
            text=hover_list,
            hovertemplate="%{text}<extra></extra>",
        ))

    # Observed line with hover suppressed since the fitted card already shows observed value
    fig.add_trace(go.Scatter(
        x=hist["Date"],
        y=hist["Dengue"].apply(to_inc),
        mode="lines",
        line=dict(color=C_OBS, width=2.2),
        name="Observed Cases",
        legendrank=1,
        hovertemplate="<extra></extra>",
    ))

    # Forecast median line with clean hover card matching the historical style
    if has_fc:
        hover_texts = []
        for _, r in df_fc.iterrows():
            month_year = r["Date"].strftime("%B %Y")
            hover_texts.append(
                f"<b>{month_year} - {chart_title}</b><br>"
                f"Median: {to_inc(r['Median']):.1f}<br>"
                f"80% PI: {to_inc(r['Lower80']):.1f} to {to_inc(r['Upper80']):.1f}<br>"
                f"50% PI: {to_inc(r['Lower50']):.1f} to {to_inc(r['Upper50']):.1f}"
            )

        fig.add_trace(go.Scatter(
            x=df_fc["Date"],
            y=df_fc["Median"].apply(to_inc),
            mode="lines+markers",
            line=dict(color=C_FORECAST, width=2.8),
            marker=dict(size=4, color=C_FORECAST),
            name="Median Forecast 2026-2028",
            legendrank=0,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))

        # Vertical dotted line marking start of forecast period
        fig.add_shape(
            type="line", xref="x", yref="paper",
            x0="2026-01-01", x1="2026-01-01", y0=0, y1=1,
            line=dict(color="#777", width=2, dash="dot"),
        )

    # Annotation for selections that have no recorded cases
    if hist["Dengue"].sum() == 0:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.5, y=0.5,
            text="No dengue cases recorded for this selection",
            showarrow=False, font=dict(size=13, color="#777"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc", borderwidth=1,
        )

    blend_pct = int(round(hw_weight * 100))
    subtitle  = (
        f"ST-RF ({100 - blend_pct}%) + Holt-Winters ({blend_pct}%) blend  -  "
        "Blue = Historical PI  Red = Forecast PI  50% dark band, 80% light band"
        if has_fc else
        "No model forecast - municipality had no environmental covariates"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Dengue Forecast - {chart_title}</b><br>"
                f'<span style="font-size:11px;color:#555">{subtitle}</span>'
            ),
            x=0.5, xanchor="center",
            font=dict(size=15, color="#1a252f"),
            pad=dict(b=10),
        ),
        xaxis=dict(
            title=dict(text="Month / Year", font=dict(size=12)),
            tickformat="%b\n%Y", dtick="M3",
            showgrid=True, gridcolor="rgba(180,180,180,0.25)",
            zeroline=False, showline=True, linecolor="#ccc", linewidth=1,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=12)),
            showgrid=True, gridcolor="rgba(180,180,180,0.25)",
            rangemode="tozero",
            zeroline=False, showline=True, linecolor="#ccc", linewidth=1,
        ),
        legend=dict(
            orientation="h",
            yanchor="top", y=-0.20,
            xanchor="center", x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ddd", borderwidth=1,
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white", bordercolor="#ccc",
            font=dict(size=13, color="#1a252f"),
            align="left", namelength=-1,
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        width=1200, height=580,
        margin=dict(l=70, r=50, t=130, b=145),
    )
    return fig


# Build HTML cross-validation results table
def make_cv_table():
    if df_cv.empty:
        return "<p style='color:#777'>No CV results available.</p>"

    rows_html = ""
    for _, r in df_cv.iterrows():
        rows_html += (
            f"<tr>"
            f"<td>up to {int(r['Train up to'])}</td>"
            f"<td>{int(r['Test year'])}</td>"
            f"<td>{r['N_test']:,}</td>"
            f"<td><b>{r['Log-MAE']:.4f}</b></td>"
            f"<td>{r['MAE']:.1f}</td>"
            f"<td>{r['RMSE']:.1f}</td>"
            f"</tr>"
        )
    mean_row = (
        f"<tr style='background:#eaf4fb;font-weight:bold'>"
        f"<td colspan='3'>Mean</td>"
        f"<td>{df_cv['Log-MAE'].mean():.4f}</td>"
        f"<td>{df_cv['MAE'].mean():.1f}</td>"
        f"<td>{df_cv['RMSE'].mean():.1f}</td>"
        f"</tr>"
    )
    return f"""
    <table style="border-collapse:collapse;font-size:12px;font-family:monospace;width:100%">
      <thead>
        <tr style="background:#1a252f;color:white">
          <th style="padding:6px 10px">Train up to</th>
          <th style="padding:6px 10px">Test year</th>
          <th style="padding:6px 10px">N rows</th>
          <th style="padding:6px 10px">Log-MAE</th>
          <th style="padding:6px 10px">MAE (cases)</th>
          <th style="padding:6px 10px">RMSE (cases)</th>
        </tr>
      </thead>
      <tbody>{rows_html}{mean_row}</tbody>
    </table>
    <p style="font-size:10px;color:#555;margin-top:4px">
      Log-MAE is the primary metric (robust to outbreak-spike skew).
    </p>
    """


# Widget setup
dist_to_munis = dist_to_munis_full
all_munis     = all_munis_full
all_districts = all_districts_full

level_toggle = widgets.ToggleButtons(
    options=["National", "District", "Municipality"],
    description="View Level:",
    value="National",
    button_style="info",
    style={"description_width": "initial"},
    layout=widgets.Layout(margin="0 0 10px 0"),
)

district_dd = widgets.Dropdown(
    options=all_districts,
    value=all_districts[0] if all_districts else None,
    description="District:",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="380px", display="none"),
)

muni_filter = widgets.Text(
    value="",
    placeholder="Type to search municipalities",
    description="Search:",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="420px", display="none"),
)

_init_munis = sorted(dist_to_munis.get(all_districts[0], [])) if all_districts else []

muni_select = widgets.Select(
    options=_init_munis,
    value=_init_munis[0] if _init_munis else None,
    description="Municipality:",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="420px", height="170px", display="none"),
)

hw_slider = widgets.FloatSlider(
    value=0.5, min=0.0, max=1.0, step=0.05,
    description="HW weight:",
    readout_format=".2f",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="420px"),
    tooltip="0 = pure RF, 1 = pure Holt-Winters, 0.5 = equal blend",
)

incidence_toggle = widgets.ToggleButton(
    value=False,
    description="Incidence / 100k",
    button_style="warning",
    icon="percent",
    layout=widgets.Layout(width="180px", display=("" if HAS_POP else "none")),
)

risk_toggle = widgets.ToggleButton(
    value=True,
    description="Risk Lines",
    button_style="",
    icon="minus",
    layout=widgets.Layout(width="130px", display=("" if HAS_POP else "none")),
)

cv_toggle = widgets.ToggleButton(
    value=False,
    description="Show CV Results",
    button_style="info",
    icon="table",
    layout=widgets.Layout(width="170px"),
)

plot_btn = widgets.Button(
    description="Plot",
    button_style="success",
    layout=widgets.Layout(width="120px", height="36px"),
)

out    = widgets.Output()
out_cv = widgets.Output()
_current_dist_munis = _init_munis[:]


def _update_visibility():
    level = level_toggle.value
    district_dd.layout.display = "" if level in ("District", "Municipality") else "none"
    muni_filter.layout.display = "" if level == "Municipality" else "none"
    muni_select.layout.display = "" if level == "Municipality" else "none"


def _do_plot(_btn=None):
    level = level_toggle.value
    with out:
        clear_output(wait=True)

        if level == "National":
            muni_list   = all_munis
            chart_title = "Nepal - All Districts"
        elif level == "District":
            sel_dist    = district_dd.value
            muni_list   = list(dist_to_munis.get(sel_dist, all_munis))
            chart_title = f"District: {sel_dist}"
        else:
            sel_muni = muni_select.value
            if not sel_muni:
                print("  No municipality selected.")
                return
            muni_list   = [sel_muni]
            chart_title = f"Municipality: {sel_muni}"

        hist      = build_historical_df(muni_list)
        active_fc = [m for m in muni_list if m in modelled_munis]
        pop_total = get_population_total(muni_list)

        print(f"  Scope:               {chart_title}")
        print(f"  Municipalities:      {len(muni_list)}")
        print(f"  With forecast:       {len(active_fc)}")
        if not np.isnan(pop_total):
            print(f"  Total population:    {pop_total:,.0f}")

        if hist["Dengue"].sum() > 0:
            pk = hist.loc[hist["Dengue"].idxmax()]
            print(f"  Historical peak:     {pk['Dengue']:.0f} cases ({MONTH_NAME_MAP[int(pk['Month_Num'])]} {int(pk['Year'])})")
        else:
            print("  Historical peak:     0 cases")

        if active_fc:
            fc_df = build_forecast_df(muni_list, hw_weight=hw_slider.value)
            pk_fc = fc_df.loc[fc_df["Median"].idxmax()]
            print(f"  Forecast peak:       {pk_fc['Median']:.0f} cases ({MONTH_NAME_MAP[int(pk_fc['Month_Num'])]} {int(pk_fc['Year'])})")
        else:
            print("  Forecast peak:       N/A")

        print()
        fig = make_figure(
            muni_list, chart_title,
            hw_weight=hw_slider.value,
            show_incidence=incidence_toggle.value,
            show_risk_lines=risk_toggle.value,
        )
        display(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True, "scrollZoom": True})))

    with out_cv:
        clear_output(wait=True)
        if cv_toggle.value:
            display(HTML(make_cv_table()))


# Widget event handlers
def on_level_change(change):
    _update_visibility()
    _do_plot()

def on_district_change(change):
    global _current_dist_munis
    _current_dist_munis = sorted(dist_to_munis.get(change["new"], all_munis))
    muni_filter.value   = ""
    muni_select.options = _current_dist_munis
    muni_select.value   = _current_dist_munis[0] if _current_dist_munis else None
    _do_plot()

def on_filter_change(change):
    global _current_dist_munis
    q        = change["new"].strip().lower()
    filtered = [m for m in _current_dist_munis if q in m.lower()] if q else _current_dist_munis
    prev     = muni_select.value
    muni_select.options = filtered if filtered else _current_dist_munis
    muni_select.value   = prev if prev in muni_select.options else (muni_select.options[0] if muni_select.options else None)

def on_muni_change(change):
    if level_toggle.value == "Municipality" and change["new"]:
        _do_plot()

def on_hw_change(change):
    _do_plot()

def on_incidence_change(change):
    _do_plot()

def on_risk_change(change):
    _do_plot()

def on_cv_toggle(change):
    with out_cv:
        clear_output(wait=True)
        if change["new"]:
            display(HTML(make_cv_table()))

level_toggle.observe(on_level_change,         names="value")
district_dd.observe(on_district_change,       names="value")
muni_filter.observe(on_filter_change,         names="value")
muni_select.observe(on_muni_change,           names="value")
hw_slider.observe(on_hw_change,               names="value")
incidence_toggle.observe(on_incidence_change, names="value")
risk_toggle.observe(on_risk_change,           names="value")
cv_toggle.observe(on_cv_toggle,               names="value")
plot_btn.on_click(_do_plot)

header = widgets.HTML("""
<div style="background:#1a1a2e;color:white;padding:14px 22px;border-radius:10px;
            font-family:'Courier New',monospace;font-size:15px;margin-bottom:12px;
            border-left:4px solid #e74c3c;">
  Dengue ST-RF Forecast Dashboard v5
  <span style="font-size:11px;color:#aaa;margin-left:14px;">
    Nepal | 2016-2028 | ST-RF + Holt-Winters | Walk-forward CV | log1p transform | Neighbour-lag
  </span>
</div>
""")

hint = widgets.HTML("""
<div style="margin:4px 0 10px 0;font-size:11px;color:#555;line-height:1.8;">
  Select a view level, then choose a district or municipality (use Search to filter).<br>
  Blue bands = Historical PI: dark inner = 50%, light outer = 80%.<br>
  Red bands = Forecast PI: dark inner = 50%, light outer = 80%.<br>
  HW weight 0 = pure Random Forest, 1 = pure Holt-Winters.<br>
  Toggle Show CV Results to see walk-forward validation metrics.
</div>
""")

controls = widgets.VBox(
    [
        header,
        level_toggle,
        widgets.HBox([district_dd]),
        widgets.HBox([muni_filter, muni_select]),
        widgets.HBox([hw_slider]),
        widgets.HBox([incidence_toggle, risk_toggle, cv_toggle], layout=widgets.Layout(gap="10px", margin="6px 0")),
        hint,
        plot_btn,
    ],
    layout=widgets.Layout(padding="8px"),
)

display(widgets.VBox([controls, out_cv, out]))
_update_visibility()
_do_plot()
