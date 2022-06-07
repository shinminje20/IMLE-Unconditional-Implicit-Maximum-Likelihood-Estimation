import os
from tqdm import tqdm

files = [f"wandb/{f}" for f in os.listdir("wandb") if f.startswith("offline")]
for f in tqdm(files):
    os.system(f"wandb sync {f} > wandb_sync_results.txt") # The documentation of subprocess is gross
    
    with open("wandb_sync_results.txt", "r") as f:
        result = f.read()

    if result.endswith("done."):
        continue
    else:
        shutil.rmtree(f)
        tqdm.write(f"{f} threw an error, and it was removed")

shutil.rmtree("wandb_sync_results.txt")