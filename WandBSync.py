import os
from tqdm import tqdm

files = [f"wandb/{f}" for f in os.listdir("wandb") if f.startswith("offline")]
for f in tqdm(files):
    os.system(f"wandb sync {f}")