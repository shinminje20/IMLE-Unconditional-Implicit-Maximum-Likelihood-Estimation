import os
files = os.listdir("wandb")
files = [f for f in files if f.startswith("offline-run")]
for f in files:
    os.system(f"wandb sync wandb/{f}")