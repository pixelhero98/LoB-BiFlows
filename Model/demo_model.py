import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 2. The Model (BiFlow)
# ==========================================
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
        context = self.get_context(history)
        
        # Define loss functions
        loss_l1 = nn.SmoothL1Loss() # Precision (Peak)
        loss_l2 = nn.MSELoss()      # Volatility (Tails)
        
        # --- 1. Generator Loss (Velocity Matching) ---
        z = torch.randn_like(x_real)
        t = torch.rand(x_real.shape[0], 1).to(x_real.device)
        x_t = (1 - t) * z + t * x_real
        
        v_pred = self.generator_flow(torch.cat([x_t, t, context], dim=1))
        target_v = x_real - z
        
        # Mix: L1 + L2 objective
        loss_gen_l1 = loss_l1(v_pred, target_v)
        loss_gen_l2 = loss_l2(v_pred, target_v)
        loss_gen = (0.82 * loss_gen_l1) + (0.18 * loss_gen_l2)
        
        # --- 2. Cycle Loss (Consistency) ---
        z_pred = self.flow_step(self.encoder_flow, x_real, context)
        x_recon = self.flow_step(self.generator_flow, z_pred, context)
        
        # Mix for Cycle as well
        loss_cycle_l1 = loss_l1(x_real, x_recon)
        loss_cycle_l2 = loss_l2(x_real, x_recon)
        loss_cycle = (0.82 * loss_cycle_l1) + (0.18 * loss_cycle_l2)
        
        return loss_gen + (LOBConfig.lambda_cycle * loss_cycle)

    @torch.no_grad()
    def generate_step(self, history):
        context = self.get_context(history)
        z = torch.randn(history.shape[0], LOBConfig.input_dim).to(history.device)
        return self.flow_step(self.generator_flow, z, context)
