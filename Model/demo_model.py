import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import linear_sum_assignment

class BiFlowLOB(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Components
        self.context_encoder = nn.LSTM(config.input_dim, config.hidden_dim, batch_first=True)
        self.generator_flow = self._build_net(config)
        self.encoder_flow = self._build_net(config)
        
    def _build_net(self, config):
        return nn.Sequential(
            nn.Linear(config.input_dim + 1 + config.hidden_dim, 128),
            nn.SiLU(), nn.Linear(128, 128),
            nn.SiLU(), nn.Linear(128, config.input_dim)
        )

    def get_context(self, history):
        _, (h_n, _) = self.context_encoder(history)
        return h_n.squeeze(0)

    def flow_step(self, net, x, context, dt=1.0):
        t = torch.zeros(x.shape[0], 1).to(x.device)
        v = net(torch.cat([x, t, context], dim=1))
        return x + v * dt

    def compute_loss(self, x_real, history):
        context = self.context_encoder(history)
        batch_size = x_real.shape[0]

        # --- 1. OT PAIRING (The "MeanFlow" style) ---
        # Sample random noise
        z_random = torch.randn_like(x_real)

        # Detach for OT calculation (we don't differentiate through the sorting)
        x_flat = x_real.view(batch_size, -1).detach().cpu()
        z_flat = z_random.view(batch_size, -1).detach().cpu()

        # Calculate Cost Matrix (Euclidean Distance squared)
        # We want to match x[i] with the z[j] that is closest to it
        dist_matrix = torch.cdist(x_flat, z_flat, p=2) ** 2
        
        # Hungarian Algorithm to find optimal pairs (Minimize total distance)
        row_ind, col_ind = linear_sum_assignment(dist_matrix.numpy())
        
        # Reorder Z to match X optimally
        z_aligned = z_random[col_ind].to(x_real.device)

        # --- 2. Generator Loss (Velocity Matching) ---
        # Now we regress the path from z_aligned -> x_real
        # This path is statistically much "straighter" than random pairing
        t = torch.rand(batch_size, 1).to(x_real.device)
        x_t = (1 - t) * z_aligned + t * x_real
        
        v_pred = self.generator_flow(torch.cat([x_t, t, context], dim=1))
        target_v = x_real - z_aligned # Vector points to target

        # Mixed Loss (75% L1, 25% MSE)
        loss_l1 = nn.SmoothL1Loss()
        loss_l2 = nn.MSELoss()
        loss_gen = 0.75 * loss_l1(v_pred, target_v) + 0.25 * loss_l2(v_pred, target_v)

        # --- 3. Cycle Loss (BiFlow Consistency) ---
        # We use the Generator (trained with OT) to check consistency
        z_pred = self.flow_step(self.encoder_flow, x_real, context)
        x_recon = self.flow_step(self.generator_flow, z_pred, context)
        
        loss_cycle = 0.75 * loss_l1(x_real, x_recon) + 0.25 * loss_l2(x_real, x_recon)

        return loss_gen + (LOBConfig.lambda_cycle * loss_cycle)

    @torch.no_grad()
    def generate_step(self, history):
        context = self.get_context(history)
        z = torch.randn(history.shape[0], LOBConfig.input_dim).to(history.device)
        return self.flow_step(self.generator_flow, z, context)
