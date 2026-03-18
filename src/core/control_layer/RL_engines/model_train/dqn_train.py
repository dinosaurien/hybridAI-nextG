import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import custom modules from your src directory
from core.control_layer.RL_engines.observer_rl import RLObserver, Intent
from core.control_layer.RL_engines.model_defs import SlateDQNetwork

# --- GPU Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Constants & Setup ---
ACTION_TYPES = ["MCS_CAP", "PRB_WEIGHT", "SLICE_QOS", "SCHEDULER_POLICY", "REPORTING"]
SCOPES = ["CELL", "SLICE"]
CELLS = ["CELL_001", "CELL_002"]
SLICES = ["SLICE_A", "SLICE_B"]

W = 12
CELL_CAP  = len(CELLS) 
SLICE_CAP = len(SLICES)

# Dynamically fetch F
F = len(RLObserver(None, Intent(type="REDUCE_LATENCY", metric="delay_p95_ms", target=40)).features)
print(f"Trainer F = {F}")

# --- 1. Generate Dummy Replay Data ---
K = 3
D = 5 + 3 + CELL_CAP + SLICE_CAP + 8  # = 20
N = 5000
data = []

for _ in range(N):
    sample = {
        "s":  np.random.randn(W, F).astype(np.float32),
        "a":  np.random.randn(K, D).astype(np.float32),
        "r":  float(np.random.randn()),
        "s2": np.random.randn(W, F).astype(np.float32),
        "done": bool(np.random.rand() < 0.1),
    }
    data.append(sample)

np.savez_compressed("replay_buffer.npz", data=data)
print(f"Dummy replay regenerated with D= {D}")

# --- 2. Define Dataset ---
class Replay(Dataset):
    def __init__(self, path):
        buf = np.load(path, allow_pickle=True)["data"]
        self.data = buf
        
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, i):
        t = self.data[i]
        # Cleanly convert to tensors here to avoid the warnings you saw in Jupyter
        return (
            torch.tensor(t["s"], dtype=torch.float32),
            torch.tensor(t["a"], dtype=torch.float32),
            torch.tensor(t["r"], dtype=torch.float32),
            torch.tensor(t["s2"], dtype=torch.float32),
            torch.tensor(t["done"], dtype=torch.float32)
        )

# --- 3. Training Loop ---
ds = Replay("replay_buffer.npz")
dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)

# Initialize models and move to ROCm GPU
net = SlateDQNetwork(feat_dim=F, cell_cap=CELL_CAP, slice_cap=SLICE_CAP).to(device)
tgt = SlateDQNetwork(feat_dim=F, cell_cap=CELL_CAP, slice_cap=SLICE_CAP).to(device)
tgt.load_state_dict(net.state_dict())

opt = optim.Adam(net.parameters(), lr=1e-3)
gamma = 0.95
tau = 0.005

print("Starting training...")
for epoch in range(10):
    epoch_loss = 0.0
    for s, p, r, s2, done in dl:
        # Move batch to GPU
        s, p, r, s2, done = s.to(device), p.to(device), r.to(device), s2.to(device), done.to(device)

        q = net(s, p)

        with torch.no_grad():
            q2 = tgt(s2, p)
            y  = r + (1.0 - done) * gamma * q2

        loss = (q - y).pow(2).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        epoch_loss += loss.item()

        # Soft update target network
        with torch.no_grad():
            for tp, p_ in zip(tgt.parameters(), net.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p_.data)
                
    print(f"Epoch {epoch+1}/10 | Loss: {epoch_loss/len(dl):.4f}")

# --- 4. Save Model ---
os.makedirs("models", exist_ok=True) # Ensure directory exists
torch.save({
    "state_dict": net.state_dict(),
    "meta": {"W": W, "F": F, "cell_cap": CELL_CAP, "slice_cap": SLICE_CAP}
}, "models/qnet_offline.pt")

print("Saved model to models/qnet_offline.pt")