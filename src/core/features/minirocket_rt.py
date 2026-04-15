import joblib, numpy as np, pandas as pd
from collections import deque
import logging

logger = logging.getLogger(__name__)

class MiniRocketRT:
    """
    Streaming wrapper:
      - keep rolling window
      - transform via saved MiniRocket
      - classify (0=normal, 1=deviation)
    """

    def __init__(self, model_path="models/minirocket.joblib", win=128, entity_id="unknown"):
        bundle = joblib.load(model_path)
        self.mr = bundle["mr"]
        self.clf = bundle["clf"]
        self.win = win
        self.buf = deque(maxlen=win)
        self.entity_id = entity_id
        self._is_active = False # Flag to ensure we only print the "Active" message once

    def push(self, v: float):
        self.buf.append(float(v))
        current_len = len(self.buf)
        
        if current_len < self.win:
            # Print every 10% or so to avoid spamming the logs too heavily
            if current_len % max(1, (self.win // 10)) == 0:
                pct = int((current_len / self.win) * 100)
                logger.info(f"[MINIROCKET] {self.entity_id} Buffer filling: {current_len}/{self.win} ({pct}%)")
            return None  # not ready
            
        if not self._is_active:
            # Print when the model becomes active
            logger.info(f"[MINIROCKET] {self.entity_id} Buffer FULL ({self.win}). Model is now ACTIVE and predicting.")
            self._is_active = True

        try:
            # MiniRocket wants 3D input: (n_instances, n_channels, length)
            X_np = np.array(self.buf).reshape(1, 1, -1)
            Xf = self.mr.transform(X_np)
            pred = int(self.clf.predict(Xf)[0])  # 1 = deviation
        except Exception as e:
            # Fallback to pandas if your specific sktime version mandates it
            X = pd.DataFrame({"signal": [pd.Series(np.array(self.buf))]})
            Xf = self.mr.transform(X)
            pred = int(self.clf.predict(Xf)[0])
            
        return {"ready": True, "pred": pred, "window": list(self.buf)}