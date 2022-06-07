import os
from tqdm import tqdm
import shutil

files = [f"wandb/{f}" for f in os.listdir("wandb") if f.startswith("offline")]
for f in tqdm(files):
    os.system(f"wandb sync {f} > wandb_sync_results.txt 2>&1") # The documentation of subprocess is gross
    
    with open("wandb_sync_results.txt", "r") as result:
        result = result.read()

    if result.strip().endswith("done."):
        continue
    else:
        shutil.rmtree(f)
        tqdm.write(f"{f} threw an error, and it was removed")

os.remove("wandb_sync_results.txt")