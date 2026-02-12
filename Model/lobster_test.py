import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from demo_model import BiFlowLOB


class LobsterDataset(Dataset):
    def __init__(self, orderbook_file, history_len=50, prediction_horizon=1):
        
        # --- 1. Robust Loading Logic ---
        import os
        if not os.path.exists(orderbook_file):
            print(f"FILE NOT FOUND: {orderbook_file}")
            print("Generating REALISTIC Dummy Data (Random Walk)...")
            
            # FIX: Use Random Walk (cumsum), not random noise
            # Generate 10,000 steps of log-returns
            dummy_returns = np.random.randn(10000) * 0.0001 # Realistic volatility
            dummy_prices = 100.0 * np.exp(np.cumsum(dummy_returns))
            
            # Create DataFrame to mimic LOBSTER
            self.df = pd.DataFrame(index=range(10000), columns=range(4))
            self.df.iloc[:, 0] = dummy_prices + 0.01 # Ask
            self.df.iloc[:, 2] = dummy_prices - 0.01 # Bid
            self.df.iloc[:, 1] = 100 # Ask Vol
            self.df.iloc[:, 3] = 100 # Bid Vol
        else:
            print(f"Loading LOBSTER file: {orderbook_file}...")
            self.df = pd.read_csv(orderbook_file, header=None)

        # --- 2. Processing (Same as before) ---
        best_ask = self.df.iloc[:, 0]
        best_bid = self.df.iloc[:, 2]
        mid_prices = (best_ask + best_bid) / 2.0
        
        log_prices = np.log(mid_prices + 1e-8)
        returns = log_prices.diff().fillna(0.0).values
        
        # --- 3. DYNAMIC SCALING (The Fix) ---
        # Instead of hardcoding 1000, we normalize based on the data
        # We want the standard deviation to be ~1.0
        std_dev = np.std(returns) + 1e-8
        self.scale_factor = 1.0 / std_dev
        
        self.returns = returns * self.scale_factor
        
        print(f"   [Data Stats] Raw Std: {std_dev:.6f} | Applied Scale: {self.scale_factor:.2f}")
        
        # 3. Order Imbalance (Conditioning)
        # rho = (BidVol - AskVol) / (BidVol + AskVol)
        # Col 1 = Best Ask Vol, Col 3 = Best Bid Vol
        ask_vol = self.df.iloc[:, 1]
        bid_vol = self.df.iloc[:, 3]
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)
        self.imbalance = imbalance.values
        
        self.data = torch.tensor(
            np.stack([self.returns, self.imbalance], axis=1), 
            dtype=torch.float32
        )
        self.history_len = history_len
        self.horizon = prediction_horizon
        
        print(f"Loaded {len(self.data)} ticks. Returns Scaled by {self.scale_factor}x")

    def __len__(self):
        # We need enough data for history + target
        return len(self.data) - self.history_len - self.horizon

    def __getitem__(self, idx):
        # Input: History [t : t+50]
        history = self.data[idx : idx + self.history_len]
        
        # Target: Future [t+50+horizon]
        # We predict the return 'horizon' steps ahead
        target = self.data[idx + self.history_len + self.horizon - 1]
        
        return history, target

csv_path = "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv" 

# Check if file exists, else use dummy (for testing code without file)
import os
if os.path.exists(csv_path):
    dataset = LobsterDataset(csv_path, history_len=LOBConfig.history_len)
    dataloader = DataLoader(dataset, batch_size=LOBConfig.batch_size, shuffle=True, drop_last=True)
    print("Real Data Loaded Successfully.")
else:
    print("WARNING: CSV not found. Generating Dummy LOBSTER data for demo.")
    # Create a dummy CSV for testing logic
    dummy_df = pd.DataFrame(np.random.rand(10000, 4) * 100 + 100) # Prices
    dummy_df.iloc[:, 1] = np.random.randint(1, 100, 10000) # Vols
    dummy_df.iloc[:, 3] = np.random.randint(1, 100, 10000) # Vols
    dummy_df.to_csv("dummy_lobster.csv", header=False, index=False)
    dataset = LobsterDataset("dummy_lobster.csv", history_len=LOBConfig.history_len)
    dataloader = DataLoader(dataset, batch_size=LOBConfig.batch_size, shuffle=True)

# 2. Train Loop (Updated)
model = BiFlowLOB(LOBConfig).to(LOBConfig.device)
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower LR for real data!

print("Starting Training on LOBSTER...")

# We iterate through epochs now, not just random steps
num_epochs = 500
global_step = 0

for epoch in range(num_epochs):
    for batch_hist, batch_target in dataloader:
        
        # Move to GPU
        batch_hist = batch_hist.to(LOBConfig.device)
        batch_target = batch_target.to(LOBConfig.device)
        
        optimizer.zero_grad()
        
        # We only predict the Return (feature 0), not the Imbalance (feature 1)
        # But the model outputs both dimensions.
        loss = model.compute_loss(batch_target, batch_hist)
        
        loss.backward()
        optimizer.step()
        
        global_step += 1
        if global_step % 100 == 0:
            print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.6f}")

print("Training Complete.")

print("\n3. Visualizing Model Performance on Loader Data...")

# 1. Get a sample batch from the loader (Real/Dummy Data)
# We need an iterator to grab just one batch
data_iter = iter(dataloader)
real_hist, real_target = next(data_iter)

# Move to GPU
real_hist = real_hist.to(LOBConfig.device)

# 2. Generate Autoregressive Sequence
# We take the first sample in the batch [Index 0]
initial_history = real_hist[0].unsqueeze(0) # Shape: [1, 50, 2]

# Generate 100 future steps
model.eval()
gen_path = []
curr_hist = initial_history.clone()

for _ in range(100):
    with torch.no_grad():
        # Predict next return
        next_step = model.generate_step(curr_hist) # Output: [1, 2]
        gen_path.append(next_step)
        
        # Update history (Slide window)
        # Append new prediction to the end, drop the first history point
        curr_hist = torch.cat([curr_hist[:, 1:, :], next_step.unsqueeze(1)], dim=1)

# 3. Process Data for Plotting
# Stack the list into a tensor
gen_tensor = torch.stack(gen_path).squeeze(1).cpu() # Shape: [100, 2]
gen_returns = gen_tensor[:, 0].numpy()

# Get Real Future (just use the target batch for distribution comparison)
# Note: In a real loader, we can't easily get the "next 100 steps" for the same sample 
# without iterating, so we compare distribution against the *batch* targets.
real_batch_returns = real_target[:, 0].cpu().numpy()

# 4. Plot
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot A: Autoregressive Path (Cumulative Sum -> Price Shape)
ax = axes[0]
# We reconstruct a "Price Index" from returns
gen_price = np.cumsum(gen_returns)
ax.plot(gen_price, label='Generated Price Path', color='dodgerblue')
ax.set_title("Generated Scenario (100 Steps)")
ax.grid(alpha=0.3)
ax.legend()

# Plot B: Distribution Matching
ax = axes[1]
sns.kdeplot(real_batch_returns, ax=ax, fill=True, label='Real Data (Batch)', color='gray')
sns.kdeplot(gen_returns, ax=ax, fill=True, label='Generated Data', color='dodgerblue')
ax.set_title("Volatility Check (Real vs Generated)")
ax.legend()
ax.grid(alpha=0.3)

plt.show()
