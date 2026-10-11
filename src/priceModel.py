"""Train an up/down classifier on data/features.csv with train / validation / test blocks.

Chronological split (the model never sees the future):
    train       oldest ~70%   fit the model
    validation  next   ~15%   choose depth, number of trees and feature set
    test        newest ~15%   one final score for the chosen model, nothing else

Gradient boosting accepts missing values, so delivery data (which only exists for about
the last year) is optional and no rows are dropped.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from priceFeature import BASE_COLS, FEATURE_COLS

FEATURES_FILE = Path(__file__).resolve().parent.parent / "data" / "features.csv"
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15
MAX_ITER = 400
MIN_COVERAGE = 0.10  # a feature set needs every column present in at least 10% of train rows
FEATURE_SETS = {"no delivery": BASE_COLS, "with delivery": FEATURE_COLS}


def split(df: pd.DataFrame):
    """Split by date so a trading day never appears in two blocks."""
    dates = df.index
    val_start = dates[int(len(df) * TRAIN_FRACTION)]
    test_start = dates[int(len(df) * (TRAIN_FRACTION + VAL_FRACTION))]
    return df[dates < val_start], df[(dates >= val_start) & (dates < test_start)], df[dates >= test_start]


def describe(name: str, block: pd.DataFrame) -> None:
    print(f"{name:11s} {len(block):6d} rows  {block.index.min().date()} to {block.index.max().date()}  "
          f"up {(block['target'] == 1).mean():.1%}")


def main() -> None:
    df = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True)
    df = df.dropna(subset=["target"] + BASE_COLS).sort_index()
    train, val, test = split(df)
    describe("train", train)
    describe("validation", val)
    describe("test", test)

    # --- validation: pick feature set, depth and number of boosting rounds -------------
    print("\nValidation (lower log loss is better; 0.693 = knows nothing)")
    print(f"{'features':14s} depth  best_round  log_loss  accuracy    AUC")
    best = None
    for set_name, cols in FEATURE_SETS.items():
        coverage = train[cols].notna().mean().min()
        if coverage < MIN_COVERAGE:
            print(f"{set_name:14s} skipped: a column is filled in only {coverage:.1%} of train rows")
            continue
        for depth in (2, 3, 4, 6):
            model = HistGradientBoostingClassifier(
                max_depth=depth, learning_rate=0.05, max_iter=MAX_ITER,
                min_samples_leaf=100, early_stopping=False, random_state=42,
            ).fit(train[cols], train["target"])
            losses = [log_loss(val["target"], p) for p in model.staged_predict_proba(val[cols])]
            rounds = int(pd.Series(losses).idxmin()) + 1
            p_val = list(model.staged_predict_proba(val[cols]))[rounds - 1][:, 1]
            acc = accuracy_score(val["target"], p_val >= 0.5)
            auc = roc_auc_score(val["target"], p_val)
            print(f"{set_name:14s} {depth:5d}  {rounds:10d}  {min(losses):8.4f}  {acc:8.3f}  {auc:7.3f}")
            if best is None or min(losses) < best["loss"]:
                best = {"loss": min(losses), "cols": cols, "set": set_name, "depth": depth,
                        "rounds": rounds, "acc": acc, "auc": auc}

    # --- test: score only the chosen model, once ----------------------------------------
    print(f"\nChosen on validation: {best['set']}, depth {best['depth']}, {best['rounds']} rounds")
    final = HistGradientBoostingClassifier(
        max_depth=best["depth"], learning_rate=0.05, max_iter=best["rounds"],
        min_samples_leaf=100, early_stopping=False, random_state=42,
    ).fit(train[best["cols"]], train["target"])
    p_test = final.predict_proba(test[best["cols"]])[:, 1]

    up = (test["target"] == 1).mean()
    print("\nTEST (final, scored once)")
    print(f"Always up:   {up:.3f}")
    print(f"Always down: {1 - up:.3f}")
    print(f"Model:       {accuracy_score(test['target'], p_test >= 0.5):.3f}   "
          f"AUC {roc_auc_score(test['target'], p_test):.3f}   "
          f"log loss {log_loss(test['target'], p_test):.4f}")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=50, n_jobs=-1, random_state=42
    ).fit(train[BASE_COLS], train["target"])
    importance = pd.Series(rf.feature_importances_, index=BASE_COLS)
    print("\nTop 10 features (random forest, for reference):")
    print(importance.sort_values(ascending=False).head(10).round(3).to_string())


if __name__ == "__main__":
    main()
