import joblib, numpy as np, pandas as pd
from collections import deque


class MiniRocketRT:
    """
    Streaming wrapper:
      - keep rolling window
      - transform via saved MiniRocket
      - classify (0=normal, 1=deviation)
    """

    def __init__(self, model_path="models/minirocket.joblib", win=128):
        bundle = joblib.load(model_path)
        self.mr = bundle["mr"]
        self.clf = bundle["clf"]
        self.win = win
        self.buf = deque(maxlen=win)

    def push(self, v: float):
        self.buf.append(float(v))
        if len(self.buf) < self.win:
            return None  # not ready
        # sktime nested format: one row, one Series cell
        X = pd.DataFrame({"signal": [pd.Series(np.array(self.buf))]})
        Xf = self.mr.transform(X)
        pred = int(self.clf.predict(Xf)[0])  # 1 = deviation
        return {"ready": True, "pred": pred, "window": list(self.buf)}
