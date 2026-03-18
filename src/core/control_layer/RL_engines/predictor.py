
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.log_config import should_log, LOG_LEARNING, LOG_SCORING

logger = logging.getLogger(__name__)

from core.control_layer.RL_engines.proposer import (
    Playbook, ControlAction, ActionSpace,
    PLAYBOOK_K
)

from core.control_layer.RL_engines.model_defs import SlateDQNetwork

# -----------------------------
# Config (can be tweaked)
# -----------------------------

GAMMA = 0.95 # Discount factor (closer to one means longterm learning, lower means short term learning)
LR = 7e-5
BATCH_SIZE = 128  # Increased from 64 to 128 for more stable gradients with Huber loss
REPLAY_CAP = 100000
TAU = 0.001 # Target network soft update rate
EPS_START = 0.5 # Exploration vs exploitation
EPS_END = 0.1  # Increased from 0.05 to maintain exploration in non-stationary environment
EPS_DECAY_STEPS = 500  # Slower decay to handle traffic spikes and random events

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
# -----------------------------
# Replay Buffer
# -----------------------------

class ReplayBuffer: # This is where we store our previous experiences
    def __init__(self, capacity=REPLAY_CAP):
        self.capacity = capacity
        self.buf = []
        self.pos = 0
    def push(self, s, p, r, s2, d):
        data = (s, p, float(r), s2, bool(d))
        if len(self.buf) < self.capacity:
            self.buf.append(data)
        else:
            self.buf[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size): # Makes sure we can sample a batch of experiences for learning
        idxs = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idxs]
        s, p, r, s2, d = zip(*batch)
        return (
            np.array(s), 
            np.array(p), 
            np.array(r, dtype=np.float32), 
            np.array(s2), 
            np.array(d, dtype=np.float32)
        )
    def __len__(self):
        return len(self.buf)

# -----------------------------
# Predictor / Learner
# -----------------------------

class SlateDQNPredictor:
    def __init__(self, action_space: ActionSpace, feat_dim: int, seed=0):
        np.random.seed(seed); torch.manual_seed(seed)
        self.action_space = action_space
        self.cell_index = {c:i for i,c in enumerate(action_space.cells)}
        self.slice_index = {s:i for i,s in enumerate(action_space.slices)}

        cell_cap = len(self.cell_index)
        slice_cap = len(self.slice_index)

        self.model = SlateDQNetwork(feat_dim=feat_dim,
                                    cell_cap=cell_cap,
                                    slice_cap=slice_cap).to(DEVICE)
        self.target = SlateDQNetwork(feat_dim=feat_dim,
                                     cell_cap=cell_cap,
                                     slice_cap=slice_cap).to(DEVICE)
        self.target.load_state_dict(self.model.state_dict())
        self.optim = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.replay = ReplayBuffer(capacity=REPLAY_CAP)
        self.steps = 0
        self.action_onehot_dim = self.model.input_dim_action_onehot
        self.model.train()  # Start in training mode
        self.loss_history = []  # Track loss for plotting


    # Encodes our actions intpo one-hot vectors and stacks them
    def _one_hot(self, idx: int, dim: int):
        v = np.zeros(dim, dtype=np.float32)
        if 0 <= idx < dim: v[idx] = 1.0
        return v

    def _encode_action_np(self, a: ControlAction) -> np.ndarray:
        type_map = {"SCHEDULER_POLICY":0, "MCS_CAP":1, "PRB_WEIGHT":2, "TX_POWER":3, "REPORTING":4}
        scope_map = {"CELL":0, "UE":1, "SLICE":2}
        
        vecs = []
        vecs.append(self._one_hot(type_map.get(a.type, 4), 5))
        vecs.append(self._one_hot(scope_map.get(a.scope, 0), 3))

        n_cells = len(self.cell_index) if len(self.cell_index) > 0 else 2
        n_slices = len(self.slice_index) if len(self.slice_index) > 0 else 2
        
        vecs.append(self._one_hot(self.cell_index.get(a.cell_id, -1), n_cells))
        vecs.append(self._one_hot(self.slice_index.get(a.slice_id, -1), n_slices))
        
        p = np.zeros(8, dtype=np.float32)
        
        if a.type == "SCHEDULER_POLICY":
            pol = a.params.get("policy", "PF")
            pol_map = {"PF":0, "RR":1, "MAX_THROUGHPUT":2, "WEIGHTED_FAIR":3, "QOS_AWARE":4}
            p[pol_map.get(pol, 0)] = 1.0
            
        elif a.type == "MCS_CAP":
            v = float(a.params.get("dl_mcs_max", 28))
            p[0] = min(max(v / 28.0, 0.0), 1.0)
            
        elif a.type == "PRB_WEIGHT":
            w = float(a.params.get("weight", 1.0))
            p[1] = min(max((w - 0.5) / 1.5, 0.0), 1.0)
            
        elif a.type in ["TX_POWER", "POWER_CONTROL"]:
            tx = float(a.params.get("txPowerDbm", a.params.get("tx_power_dbm", 40.0)))
            p[2] = min(max((tx - 30.0) / 30.0, 0.0), 1.0)
            
        else:
            p[-1] = 1.0 # NOOP indicator
            
        vecs.append(p)
        
        encoded = np.concatenate(vecs, axis=0)
        
        if len(encoded) < 20:
            encoded = np.pad(encoded, (0, 20 - len(encoded)), 'constant')
        elif len(encoded) > 20:
            encoded = encoded[:20]
            
        return encoded


    def encode_playbook_onehot(self, pb: Playbook) -> np.ndarray: # Encodes actions into a one-hot vector (not smart encoding)
        K = PLAYBOOK_K; D = self.action_onehot_dim
        mat = np.zeros((K, D), dtype=np.float32)
        for i, a in enumerate(pb.actions[:K]):
            mat[i] = self._encode_action_np(a)
        return mat

    # --- API ---
    def score_playbooks(self, state_window: np.ndarray, playbooks: List[Playbook]) -> List[Tuple[Playbook, float]]:
        self.model.eval()
        with torch.no_grad():
            B = len(playbooks)
            state_batch = np.repeat(state_window[np.newaxis, :, :], B, axis=0)
            play_batch = np.stack([self.encode_playbook_onehot(pb) for pb in playbooks], axis=0)
            
            # Check for NaN/inf in inputs
            if np.any(np.isnan(state_batch)) or np.any(np.isinf(state_batch)):
                if should_log(LOG_SCORING):
                    logger.warning("NaN/inf detected in state input, replacing with zeros")
                state_batch = np.nan_to_num(state_batch, nan=0.0, posinf=0.0, neginf=0.0)
            if np.any(np.isnan(play_batch)) or np.any(np.isinf(play_batch)):
                if should_log(LOG_SCORING):
                    logger.warning("NaN/inf detected in playbook input, replacing with zeros")
                play_batch = np.nan_to_num(play_batch, nan=0.0, posinf=0.0, neginf=0.0)
            
            s = torch.tensor(state_batch, dtype=torch.float32, device=DEVICE)
            p = torch.tensor(play_batch, dtype=torch.float32, device=DEVICE)
            q_raw = self.model(s, p)
            
            # Clamp Q-values to reasonable range to prevent extreme values
            q_raw = torch.clamp(q_raw, -10.0, 10.0)
            q = q_raw.cpu().numpy().tolist()
            
            # Check for NaN/inf in outputs (untrained model issue)
            q_clean = []
            for q_val in q:
                if np.isnan(q_val) or not np.isfinite(q_val):
                    q_clean.append(0.0)  # Return 0 for invalid values
                else:
                    q_clean.append(float(q_val))
            q = q_clean
            
            # Debug: log Q-values if they're all zero (untrained model)
            if should_log(LOG_SCORING):
                if all(abs(qv) < 0.001 for qv in q):
                    logger.debug(f"All Q-values are ~0 (model may be untrained, steps={self.steps})")
                
                # Log scoring details
                q_mean = np.mean(q) if q else 0.0
                q_std = np.std(q) if q else 0.0
                q_min = min(q) if q else 0.0
                q_max = max(q) if q else 0.0
                logger.debug(f"[SCORING] Scored {len(playbooks)} playbooks: Q_mean={q_mean:.4f}, Q_std={q_std:.4f}, Q_range=[{q_min:.4f}, {q_max:.4f}], steps={self.steps}")
        return list(zip(playbooks, q))

    def epsilon(self):
        t = min(self.steps, EPS_DECAY_STEPS)
        return EPS_END + (EPS_START - EPS_END) * math.exp(-5.0 * t / EPS_DECAY_STEPS)
    
    # --- training ---

    def learn_step(self):
        if len(self.replay) < BATCH_SIZE:
            if should_log(LOG_LEARNING):
                logger.debug(f"[LEARNING] Skipping learn_step: replay buffer size ({len(self.replay)}) < BATCH_SIZE ({BATCH_SIZE})")
            return None
        
        self.model.train()  # Ensure model is in training mode
        
        s, p, r, s2, d = self.replay.sample(BATCH_SIZE)
        if should_log(LOG_LEARNING):
            logger.debug(f"[LEARNING] Sampling batch: replay_size={len(self.replay)}, batch_size={BATCH_SIZE}, step={self.steps}")
        
        # Check for NaN/inf in inputs and clean them
        if np.any(np.isnan(s)) or np.any(np.isinf(s)):
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(s2)) or np.any(np.isinf(s2)):
            s2 = np.nan_to_num(s2, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(np.isnan(r)) or np.any(np.isinf(r)):
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize rewards to [-1, 1] range to stabilize gradients
        # Since original rewards are clipped to [-20, 20], we divide by 20.0
        r = r / 20.0
        
        s = torch.tensor(s, dtype=torch.float32, device=DEVICE)
        p = torch.tensor(p, dtype=torch.float32, device=DEVICE)
        r = torch.tensor(r, dtype=torch.float32, device=DEVICE)
        s2 = torch.tensor(s2, dtype=torch.float32, device=DEVICE)
        d = torch.tensor(d, dtype=torch.float32, device=DEVICE)

        q = self.model(s, p)

        with torch.no_grad():
            q2 = self.target(s2, p)
            # Clip Q values to prevent explosion (normalized range)
            q2 = torch.clamp(q2, min=-1.0, max=1.0)
            y = r + GAMMA * (1.0 - d) * q2
            # Clip targets as well (normalized range)
            y = torch.clamp(y, min=-1.0, max=1.0)

        # Use Huber loss (smooth_l1_loss) instead of MSE for robustness to outliers
        # This prevents gradient explosion from extreme reward values
        loss = F.smooth_l1_loss(q, y)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            if should_log(LOG_LEARNING):
                logger.warning(f"[LEARNING] NaN/inf loss detected, skipping update (step={self.steps})")
            return None
        
        # Log learning metrics before update
        loss_val = float(loss.item())
        q_mean = float(q.mean().item())
        y_mean = float(y.mean().item())
        reward_mean = float(r.mean().item()) # Normalized reward mean
        
        self.optim.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optim.step()
        
        # Increment step counter
        self.steps += 1

        # soft target update
        with torch.no_grad():
            for tp, p_ in zip(self.target.parameters(), self.model.parameters()):
                tp.data.mul_(1 - TAU).add_(p_.data * TAU)

        # Track loss history
        self.loss_history.append({
            'step': self.steps,
            'loss': loss_val,
            'q_mean': q_mean,
            'target_mean': y_mean,
            'reward_mean': reward_mean,
            'grad_norm': float(grad_norm)
        })
        
        # Log learning progress
        if should_log(LOG_LEARNING):
            if self.steps % 10 == 0 or loss_val > 1.0:  # Log every 10 steps or if loss is high
                logger.info(f"[LEARNING] Step={self.steps}, loss={loss_val:.6f}, Q_mean={q_mean:.4f}, target_mean={y_mean:.4f}, reward_mean={reward_mean:.4f}, grad_norm={grad_norm:.4f}, replay_size={len(self.replay)}")
            else:
                logger.debug(f"[LEARNING] Step={self.steps}, loss={loss_val:.6f}, Q_mean={q_mean:.4f}, target_mean={y_mean:.4f}, reward_mean={reward_mean:.4f}")

        return loss_val
    
    def observe(self, s: np.ndarray, playbook: Playbook, r: float, s2: np.ndarray, done: bool):
        p = self.encode_playbook_onehot(playbook)   # [K,D]
        self.replay.push(s.astype(np.float32), p.astype(np.float32),
                         float(r), s2.astype(np.float32), bool(done))
        
    def load_offline(self, path="models/qnet_offline.pt"):
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt.get("meta", {})
        cell_cap = meta.get("cell_cap", len(self.cell_index))
        slice_cap = meta.get("slice_cap", len(self.slice_index))
        new = SlateDQNetwork(feat_dim=self.model.state_enc.gru.input_size, cell_cap=cell_cap, slice_cap=slice_cap).to(DEVICE)
        new.load_state_dict(ckpt["state_dict"])
        self.model = new
        self.target = SlateDQNetwork(feat_dim=self.model.state_enc.gru.input_size, cell_cap=cell_cap, slice_cap=slice_cap).to(DEVICE)
        self.target.load_state_dict(self.model.state_dict())
        self.action_onehot_dim = self.model.input_dim_action_onehot
        self.model.train()  # Set to training mode for online learning
    
    def save_checkpoint(self, path="models/qnet_online.pt", save_replay_buffer=False):
        """Save current model state for resuming training later."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "steps": self.steps,
            "meta": {
                "feat_dim": self.model.state_enc.gru.input_size,
                "cell_cap": len(self.cell_index),
                "slice_cap": len(self.slice_index),
                "cell_index": self.cell_index,
                "slice_index": self.slice_index,
            }
        }
        
        # Optionally save replay buffer (can be large)
        if save_replay_buffer:
            checkpoint["replay_buffer"] = {
                "buf": self.replay.buf,
                "pos": self.replay.pos,
                "capacity": self.replay.capacity
            }
        
        torch.save(checkpoint, path)
        return path
    
    def load_checkpoint(self, path="models/qnet_online.pt", load_replay_buffer=False):
        """Load checkpoint to resume training from previous run."""
        import os
        if not os.path.exists(path):
            return False
        
        ckpt = torch.load(path, map_location="cpu")
        meta = ckpt.get("meta", {})
        
        # Verify dimensions match
        feat_dim = meta.get("feat_dim", self.model.state_enc.gru.input_size)
        cell_cap = meta.get("cell_cap", len(self.cell_index))
        slice_cap = meta.get("slice_cap", len(self.slice_index))
        
        if (feat_dim != self.model.state_enc.gru.input_size or
            cell_cap != len(self.cell_index) or
            slice_cap != len(self.slice_index)):
            if should_log(LOG_LEARNING):
                logger.warning("Checkpoint dimensions don't match. Skipping load.")
            return False
        
        # Load model weights
        self.model.load_state_dict(ckpt["state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        
        # Load optimizer state if available
        if "optimizer_state_dict" in ckpt:
            self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        
        # Load training step count
        if "steps" in ckpt:
            self.steps = ckpt["steps"]
        
        # Load replay buffer if requested and available
        if load_replay_buffer and "replay_buffer" in ckpt:
            rb_data = ckpt["replay_buffer"]
            self.replay.buf = rb_data.get("buf", [])
            self.replay.pos = rb_data.get("pos", 0)
            self.replay.capacity = rb_data.get("capacity", REPLAY_CAP)
        
        self.model.train()  # Set to training mode for online learning
        if should_log(LOG_LEARNING):
            logger.info(f"[LEARNING] Loaded checkpoint: step={self.steps}, replay_size={len(self.replay)}")
        return True
    
    def save_loss_history(self, path: str):
        """Save loss history to JSON file for plotting."""
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        if should_log(LOG_LEARNING):
            logger.info(f"[LEARNING] Saved loss history ({len(self.loss_history)} entries) to {path}")
    
    def load_loss_history(self, path: str):
        """Load loss history from JSON file."""
        import json
        import os
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.loss_history = json.load(f)
            if should_log(LOG_LEARNING):
                logger.info(f"[LEARNING] Loaded loss history ({len(self.loss_history)} entries) from {path}")
            return True
        return False






def main():    pass

if __name__ == "__main__":
    main()