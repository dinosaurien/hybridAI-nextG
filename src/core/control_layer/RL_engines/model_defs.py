import torch 
import torch.nn as nn

class StateEncoder(nn.Module): # Takes a tensor representing the state and turns it into a vector using nn.GRU(this tensor needs to be created in the obersver)
    def __init__(self, feat_dim: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feat_dim, 
            hidden_size=hidden, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, x):
        _, h = self.gru(x)
        return h.squeeze(0)
    
class ActionEncoder(nn.Module): # Takes our actions (JSON style) and turns them into dense vectors using nn.MLP
    def __init__(self, embed_dim: int = 64, cell_cap: int = 16, slice_cap: int = 16):
        super().__init__()
        self.type_dim = 3   # MCS_CAP, TX_POWER, REPORTING
        self.scope_dim = 1  # CELL only
        self.cell_cap = cell_cap
        self.slice_cap = slice_cap
        self.param_dim = 8
        self.input_dim = self.type_dim + self.scope_dim + self.cell_cap + self.slice_cap + self.param_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU(),
        )

    def forward(self, action_batch_tensor: torch.Tensor) -> torch.Tensor:
        B, K, D = action_batch_tensor.shape
        x = action_batch_tensor.view(B*K, D)
        z = self.mlp(x)
        return z.view(B, K, -1)
    
class PlaybookEncoder(nn.Module): # Takes encoded actions (multiple dense vectors) and turns them into a single dense vector using nn.GRU
    def __init__(self, action_embed_dim: int = 64, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=action_embed_dim, 
            hidden_size=hidden, 
            num_layers=1, 
            batch_first=True
        )

    def forward(self, action_embeds):
        _, h = self.gru(action_embeds)
        return h.squeeze(0)
    
class SlateDQNetwork(nn.Module):
    def __init__(self, feat_dim: int, state_hidden=64, action_embed=64, play_hidden=64, fusion_hidden=128, cell_cap: int = 16, slice_cap: int = 16):
        super().__init__()
        self.state_enc = StateEncoder(feat_dim, hidden=state_hidden)
        self.action_enc = ActionEncoder(embed_dim=action_embed, cell_cap=cell_cap, slice_cap=slice_cap)
        self.play_enc = PlaybookEncoder(action_embed_dim=action_embed, hidden=play_hidden)
        self.fusion = nn.Sequential(
            nn.Linear(state_hidden + play_hidden, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1),
        )
        self.input_dim_action_onehot = self.action_enc.input_dim
        
        # Initialize weights with smaller values to prevent NaN in untrained model
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with smaller values to prevent NaN."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # Smaller gain
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)


    def forward(self, state_seq: torch.Tensor, playbook_onehots: torch.Tensor) -> torch.Tensor:
        hs = self.state_enc(state_seq)                 # [B,Hs]
        act_embeds = self.action_enc(playbook_onehots) # [B,K,E]
        hp = self.play_enc(act_embeds)                 # [B,Hp]
        q = self.fusion(torch.cat([hs, hp], dim=-1)).squeeze(-1)
        return q