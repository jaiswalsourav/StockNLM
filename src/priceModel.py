"""Train a cross-sectional model on data/features.csv: which stocks beat the median stock tomorrow?

Why not "will it go up?": market direction dominates the noise and the features can't predict it,
and the up/down base rate drifts between periods (51.6% in train vs 46.7% in test). The target here,
target_rel_1d, is 1 when a stock beats the median stock THAT DAY, so the base rate is 50% every day.

Chronological split by date (the model never sees the future), with a gap so nothing straddles a boundary:
    train       oldest ~70% of days   fit the model
    validation  next   ~15%           choose feature set, depth and number of boosting rounds (by mean daily IC)
    test        newest ~15%           one final score for the chosen model, nothing else

Scoring is per day, not pooled:
    IC      Spearman correlation between predicted score and next-day excess return, one value per day.
            A real edge is small (mean IC 0.01-0.03); the t-stat says whether it is distinguishable from 0.
    spread  mean excess return of the top 20% of stocks minus the bottom 20%, per day.
Set HORIZON = 5 for the 5-day target (rows overlap, so the t-stat is corrected for it).
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from priceFeature import BASE_COLS, CORE_COLS, FEATURE_COLS

FEATURES_FILE = Path(__file__).resolve().parent.parent / "data" / "features.csv"
MODEL_FILE = FEATURES_FILE.parent / "priceModel.joblib"  # read by pricePredict.py
HORIZON = 1  # trading days: 1 or 5
TARGET = f"target_rel_{HORIZON}d"
RETURN = f"excess_return_{HORIZON}d"
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15
GAP_DAYS = max(5, HORIZON)  # trading days dropped between blocks
MAX_ITER = 400
CHECKPOINTS = (25, 50, 100, 200, 400)  # boosting rounds tried on validation
MIN_COVERAGE = 0.10  # a feature set needs every column present in at least 10% of train rows
MIN_STOCKS_FOR_IC = 20  # days with fewer scored stocks are skipped
FEATURE_SETS = {"core": CORE_COLS, "core+dii+context": BASE_COLS, "with delivery": FEATURE_COLS}


def split(df: pd.DataFrame):
    """Split by whole trading days, leaving GAP_DAYS days out before validation and before test."""
    days = df.index.unique().sort_values()
    i_val = int(len(days) * TRAIN_FRACTION)
    i_test = int(len(days) * (TRAIN_FRACTION + VAL_FRACTION))
    blocks = (days[:i_val], days[i_val + GAP_DAYS:i_test], days[i_test + GAP_DAYS:])
    return tuple(df[df.index.isin(b)] for b in blocks)


def describe(name: str, block: pd.DataFrame) -> None:
    print(f"{name:11s} {len(block):7d} rows  {block.index.nunique():5d} days  "
          f"{block.index.min().date()} to {block.index.max().date()}  beat-median {block[TARGET].mean():.1%}")


def _scored(pred, ret: pd.Series) -> pd.DataFrame:
    d = pd.DataFrame({"p": np.asarray(pred, dtype=float), "r": ret.to_numpy()}, index=ret.index)
    return d[d.groupby(level=0)["p"].transform("count") >= MIN_STOCKS_FOR_IC]


def daily_ic(pred, ret: pd.Series) -> pd.Series:
    """Spearman correlation of prediction vs forward return, computed separately for each day."""
    d = _scored(pred, ret)
    by_day = d.groupby(level=0)
    rp, rr = by_day["p"].rank(), by_day["r"].rank()
    rp = rp - rp.groupby(level=0).transform("mean")
    rr = rr - rr.groupby(level=0).transform("mean")
    num = (rp * rr).groupby(level=0).sum()
    den = np.sqrt((rp ** 2).groupby(level=0).sum() * (rr ** 2).groupby(level=0).sum())
    return (num / den).dropna()


def quintile_spread(pred, ret: pd.Series) -> pd.Series:
    """Per day: mean return of the top 20% predicted stocks minus the bottom 20%."""
    d = _scored(pred, ret)
    pct = d.groupby(level=0)["p"].rank(pct=True)
    top = d["r"].where(pct > 0.8).groupby(level=0).mean()
    bottom = d["r"].where(pct <= 0.2).groupby(level=0).mean()
    return (top - bottom).dropna()


def t_stat(daily: pd.Series) -> float:
    """Mean / standard error. With a multi-day horizon neighbouring days overlap, so n is divided by HORIZON."""
    n_eff = len(daily) / HORIZON
    return float(daily.mean() / (daily.std() / np.sqrt(n_eff))) if len(daily) > 2 else float("nan")


def score(block: pd.DataFrame, p: np.ndarray) -> dict:
    ic = daily_ic(p, block[RETURN])
    return {
        "ic": ic, "ic_mean": ic.mean(), "ic_t": t_stat(ic), "spread": quintile_spread(p, block[RETURN]),
        "auc": roc_auc_score(block[TARGET], p), "loss": log_loss(block[TARGET], p),
        "acc": accuracy_score(block[TARGET], p >= 0.5),
    }


def make_model(depth: int, rounds: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=depth, learning_rate=0.05, max_iter=rounds,
        min_samples_leaf=200, early_stopping=False, random_state=42,
    )


def main() -> None:
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
    # Features may be NaN (gradient boosting handles that); only the labels must exist
    df = df.dropna(subset=[TARGET, RETURN]).sort_index()
    train, val, test = split(df)
    describe("train", train)
    describe("validation", val)
    describe("test", test)

    # --- validation: pick feature set, depth and number of boosting rounds by mean daily IC ----------
    print("\nValidation (higher IC is better; 0 = no skill. AUC 0.5 and log loss 0.693 = knows nothing)")
    print(f"{'features':17s} depth  rounds  mean_IC   IC_t    AUC  log_loss")
    best = None
    for set_name, cols in FEATURE_SETS.items():
        coverage = train[cols].notna().mean().min()
        if coverage < MIN_COVERAGE:
            print(f"{set_name:17s} skipped: a column is filled in only {coverage:.1%} of train rows")
            continue
        for depth in (2, 3, 4, 6):
            model = make_model(depth, MAX_ITER).fit(train[cols], train[TARGET])
            for rounds, proba in enumerate(model.staged_predict_proba(val[cols]), start=1):
                if rounds not in CHECKPOINTS:
                    continue
                s = score(val, proba[:, 1])
                print(f"{set_name:17s} {depth:5d}  {rounds:6d}  {s['ic_mean']:7.4f}  {s['ic_t']:5.2f}  "
                      f"{s['auc']:.3f}  {s['loss']:.4f}")
                if best is None or s["ic_mean"] > best["ic_mean"]:
                    best = {"ic_mean": s["ic_mean"], "cols": cols, "set": set_name, "depth": depth, "rounds": rounds}

    # --- test: score only the chosen model, once -------------------------------------------------
    print(f"\nChosen on validation: {best['set']}, depth {best['depth']}, {best['rounds']} rounds "
          f"(validation mean IC {best['ic_mean']:.4f})")
    final = make_model(best["depth"], best["rounds"]).fit(train[best["cols"]], train[TARGET])
    s = score(test, final.predict_proba(test[best["cols"]])[:, 1])

    ic = s["ic"]
    print("\nTEST (final, scored once)")
    print(f"Mean daily IC:   {s['ic_mean']:.4f}   (std {ic.std():.3f}, t-stat {s['ic_t']:.2f}, "
          f"IC>0 on {(ic > 0).mean():.1%} of {len(ic)} days)")
    print(f"Top-minus-bottom quintile: {s['spread'].mean():.3%} per day, t-stat {t_stat(s['spread']):.2f}")
    print(f"Classifier:      accuracy {s['acc']:.3f} (50% = coin flip)   AUC {s['auc']:.3f}   log loss {s['loss']:.4f}")
    print("A t-stat above ~2 is the minimum to take an edge seriously; below that it is noise.")

    # Reference points: simple single-feature rules scored the same way on the same test days
    print("\nSingle-feature references on the same test days (mean daily IC):")
    for label, series in (("-return_1d", -test["return_1d"]), ("-return_5d", -test["return_5d"]),
                          ("ma_50_ratio", test["ma_50_ratio"])):
        r = daily_ic(series, test[RETURN])
        print(f"  {label:12s} {r.mean():7.4f}   t {t_stat(r):5.2f}")

    # Permutation importance on VALIDATION (not test): how much AUC drops when one column is shuffled
    imp = permutation_importance(final, val[best["cols"]], val[TARGET], scoring="roc_auc",
                                 n_repeats=3, random_state=42, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=best["cols"])
    print("\nTop 10 features by permutation importance (validation AUC drop; ~0 = useless):")
    print(importance.sort_values(ascending=False).head(10).round(4).to_string())

    joblib.dump(
        {"model": final, "cols": best["cols"], "set": best["set"], "target": TARGET,
         "test_ic": float(s["ic_mean"]), "test_ic_t": float(s["ic_t"]),
         "trained_until": str(train.index.max().date())},
        MODEL_FILE,
    )
    print(f"\nSaved chosen model -> {MODEL_FILE.name} (used by pricePredict.py)")


if __name__ == "__main__":
    main()