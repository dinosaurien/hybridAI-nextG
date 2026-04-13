# Inherited from old src/demo/ain/pipeline
import joblib, numpy as np, pandas as pd
from sktime.transformations.panel.rocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from ain.pipeline.offline_demo import make_series
import os


def to_windows(x: pd.Series, y: pd.Series, win=128, step=32):
    X, Y = [], []
    for s in range(0, len(x) - win + 1, step):
        e = s + win
        X.append(pd.Series(x.values[s:e]))
        Y.append(int(y.values[s:e].max()))  # window label = any anomaly
    return pd.DataFrame({"signal": X}), np.array(Y)


def main():
    os.makedirs(
        "models", exist_ok=True
    )  # Currently adds in root, should be changed to add into src/ain/models
    x, y = make_series(n=8000, seed=1)
    X, Y = to_windows(x, y)
    mr = MiniRocket().fit(X)
    Xf = mr.transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xf, Y, test_size=0.3, stratify=Y, random_state=42
    )
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, ytr)
    acc = clf.score(Xte, yte)
    print(f"MiniRocket training acc: {acc:.3f} (windows={len(Y)}, positives={Y.sum()})")
    joblib.dump({"mr": mr, "clf": clf}, "models/minirocket.joblib")
    print("Saved model → models/minirocket.joblib")


if __name__ == "__main__":
    main()
