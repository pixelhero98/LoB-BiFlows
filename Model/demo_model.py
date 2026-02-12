import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

class BiFlowLOB(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Context Encoder (LSTM) - Processes History "Memory"
        self.context_encoder = nn.LSTM(config.input_dim, config.hidden_dim, batch_first=True)
        
        # 2. Condition Embedding (Project Imbalance to Hidden Dim) "Signal"
        # We project the 1D imbalance feature to the hidden dimension so we can mask it easily
        self.cond_projector = nn.Linear(1, config.hidden_dim) 
        
        # 3. Flow Network (Velocity Field)
        # Input: [Noisy_State(2) + Time(1) + History(64) + Condition(64)]
        net_input_dim = config.input_dim + 1 + config.hidden_dim + config.hidden_dim
        
        self.generator_flow = nn.Sequential(
            nn.Linear(net_input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, config.input_dim)
        )
        
        # Encoder Flow (Optional - for Cycle Consistency if needed later)
        self.encoder_flow = nn.Sequential(
            nn.Linear(config.input_dim + 1 + config.hidden_dim, 128),
            nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, config.input_dim)
        )

    def get_context_and_cond(self, history, dropout_prob=0.1):
        """
        Separates History (Unconditional context) and Imbalance (Conditional).
        Applies Classifier-Free Guidance Masking during training.
        """
        # history shape: [Batch, Seq, 2] -> (Returns, Imbalance)
        
        # A. Process History (The "Market Memory")
        # FIX: Explicitly unpack the tuple (output, (h_n, c_n)) from LSTM
        _, (h_n, _) = self.context_encoder(history)
        base_context = h_n.squeeze(0) # [Batch, Hidden]
        
        # B. Process Condition (The "Signal")
        # Extract the *current* imbalance (last step, feature index 1)
        imbalance = history[:, -1, 1].unsqueeze(1) # [Batch, 1]
        cond_emb = self.cond_projector(imbalance)  # [Batch, Hidden]
        
        # C. Random Masking (CFG Training)
        # If training, zero out the condition 10% of the time (dropout_prob)
        if self.training and dropout_prob > 0:
            # Create a mask: 1 = Keep, 0 = Drop
            mask = torch.bernoulli(torch.ones(history.shape[0], 1) * (1 - dropout_prob)).to(history.device)
            cond_emb = cond_emb * mask 
            
        return base_context, cond_emb

    def compute_loss(self, x_real, history):
        # x_real: [Batch, 2] (Target Returns, Target Imbalance)
        # history: [Batch, 50, 2]
        
        # 1. Get Contexts with Random Masking (p=0.1)
        # This teaches the model to work both with and without the signal
        base_ctx, cond_ctx = self.get_context_and_cond(history, dropout_prob=0.1)
        
        batch_size = x_real.shape[0]

        # --- Optimal Transport Pairing (Hungarian Algo) ---
        z_random = torch.randn_like(x_real)
        
        # Detach and move to CPU for scipy calculation
        x_flat = x_real.view(batch_size, -1).detach().cpu()
        z_flat = z_random.view(batch_size, -1).detach().cpu()
        
        dist_matrix = torch.cdist(x_flat, z_flat, p=2) ** 2
        _, col_ind = linear_sum_assignment(dist_matrix.numpy())
        z_aligned = z_random[col_ind].to(x_real.device)

        # --- Generator Loss ---
        t = torch.rand(batch_size, 1).to(x_real.device)
        x_t = (1 - t) * z_aligned + t * x_real
        
        # Input: [x_t, t, base_context, cond_context]
        net_input = torch.cat([x_t, t, base_ctx, cond_ctx], dim=1)
        v_pred = self.generator_flow(net_input)
        
        target_v = x_real - z_aligned

        # 75% L1 (Precision) + 25% MSE (Volatility)
        loss_l1 = nn.SmoothL1Loss()(v_pred, target_v)
        loss_l2 = nn.MSELoss()(v_pred, target_v)
        loss_gen = 0.75 * loss_l1 + 0.25 * loss_l2

        return loss_gen 

    @torch.no_grad()
    def generate_step(self, history, guidance_scale=1.0):
        """
        CFG Generation: v = v_uncond + scale * (v_cond - v_uncond)
        """
        # 1. Prepare Inputs
        # A. Conditional Context (Full Signal)
        base_ctx, cond_ctx = self.get_context_and_cond(history, dropout_prob=0.0)
        
        # B. Unconditional Context (Masked Signal)
        zero_cond = torch.zeros_like(cond_ctx)
        
        # 2. Sample Noise (z)
        z = torch.randn(history.shape[0], self.config.input_dim).to(history.device)
        
        # 3. Flow Integration (Euler Step dt=1.0)
        # Since Flow Matching (OT) creates straight paths, v is constant. 
        # We calculate v at t=0.
        t = torch.zeros(history.shape[0], 1).to(history.device)
        x_t = z # Start at noise
        
        # Calc v_cond (With signal)
        in_cond = torch.cat([x_t, t, base_ctx, cond_ctx], dim=1)
        v_cond = self.generator_flow(in_cond)
        
        # Calc v_uncond (Without signal)
        in_uncond = torch.cat([x_t, t, base_ctx, zero_cond], dim=1)
        v_uncond = self.generator_flow(in_uncond)
        
        # 4. Apply Guidance Formula
        # scale = 0.0 -> Unconditional
        # scale = 1.0 -> Standard Conditional
        # scale > 1.0 -> Amplify Signal
        v_final = v_uncond + guidance_scale * (v_cond - v_uncond)
        
        return z + v_final * 1.0 # x_1 = x_0 + v * dt
