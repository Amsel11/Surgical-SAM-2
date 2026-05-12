# Working on BigPurple — the SurgSAM-2 workflow

The setup below makes the cluster feel as close to a local box as it can. One
command from your laptop gets you a working Jupyter on an A100. Everything
else — clicking prompts, submitting batch jobs, viewing results — happens
inside that session.

## One-time setup (laptop)

### 1. SSH config

Add this block to `~/.ssh/config` on your laptop:

```ssh-config
Host bigpurple bp
    Hostname bigpurple.nyumc.org
    User schula12
    ServerAliveInterval 60
    ServerAliveCountMax 3
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

`ControlMaster auto` reuses one SSH connection across commands, so subsequent
SSHs to bigpurple are instant and don't re-auth. After this is in place,
`ssh bigpurple` and `ssh bp` both work.

### 2. Pull the connect script onto your laptop

```sh
mkdir -p ~/bin
scp bigpurple:/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2/tools/bp_connect.sh ~/bin/
chmod +x ~/bin/bp_connect.sh
# add ~/bin to PATH if you haven't (in ~/.zshrc or ~/.bashrc):
#   export PATH="$HOME/bin:$PATH"
```

Confirm it works:

```sh
which bp_connect.sh
```

## Daily flow

### Start a session

On your laptop:

```sh
bp_connect.sh
```

What happens, end-to-end:

1. SSHes to bigpurple.
2. Looks for a running `surgsam2_jupyter` job. If one exists, reuses it (so
   you can disconnect and reconnect without losing the kernel).
3. If none is running, submits `run_jupyter.sh`. Waits up to 10 min for
   SLURM to give us an A100 node, then up to 8 min for Jupyter to boot and
   bootstrap the venv (only the very first ever run takes that long;
   subsequent ones are seconds).
4. Reads the node name, port, and token from the job log.
5. Opens an SSH tunnel **laptop:9001 → bigpurple-login → compute-node:8889**.
6. Prints the URL (and tries to open your browser).

Leave the script running. Ctrl-C when you're done; the SLURM job keeps
running for its 4-hour walltime, so the next `bp_connect.sh` re-attaches.

### Click on prompt frames

Inside the Jupyter session, open `collect_prompts.ipynb`.

- Cell 1 lists every video on bigpurple with frame counts.
- Cell 2 defines the helpers (no need to read the code).
- Cell 3 is where you actually click. Set `VIDEO = 'XX_whip_NNNNNNNNN'` to a
  video name from cell 1's list, then run it.
- For each prompt frame (3 per video at 0, N/3, 2·N/3), a matplotlib window
  appears. **Left click** = positive, **Right click** = negative. Press
  `1`/`2`/`3` to switch which object you're clicking for. Close the window
  to advance to the next frame. After three, the JSON is saved.

When you've clicked a few videos, hop into a terminal cell or a separate
ssh tab and submit the batch run:

### Run inference on every clicked video

```sh
cd /gpfs/data/oermannlab/users/schula12/Surgical-SAM-2
N=$(ls prompts/*.json | wc -l)
sbatch --array=0-$((N-1))%8 run_inference_array.sh
```

`%8` caps concurrency at 8 A100s (polite to the cluster). Drop it if the
partition is empty.

Watch:

```sh
squeue -u $USER
tail -f logs/array_<jobid>_0.out
```

Each job is idempotent: it skips videos that already have a
`results/<video>/seed_1/log.json`. Re-running the array after adding more
prompts only processes the new ones.

### Aggregate results

```sh
python aggregate_results.py --results-dir results --out results/_summary.csv
```

Produces a CSV with per-video, per-seed: runtime, mean mask area, frames
where the mask collapsed.

### View overlay videos / drift

`view_results.ipynb` is the same notebook from the demo. It plays the
overlay videos and renders per-frame mask grids. To switch which video it
shows, edit the `GALLERY` list at the top.

## Alternative: VS Code Remote-SSH

If you'd rather not run a browser, VS Code can connect straight to the
notebook kernel running on bigpurple.

1. Install the **Remote - SSH** and **Jupyter** extensions in VS Code.
2. Run `bp_connect.sh` on your laptop as usual (this gives you a Jupyter
   server running at `127.0.0.1:9001` on your laptop).
3. In VS Code on your laptop, open the file `view_results.ipynb`. Either
   clone the repo to your laptop or — cleaner — use **Remote-SSH** to open
   the folder at `bigpurple:/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2/`.
4. Click the kernel picker (top right of the notebook) →
   **Select Another Kernel** → **Existing Jupyter Server** → paste
   `http://127.0.0.1:9001/?token=...` (the token from `bp_connect.sh`'s
   output).
5. Pick the Python 3 kernel that appears.

Cell execution now runs on the A100, output renders in VS Code. No browser
involved.

## Common issues

**Tunnel says `Address already in use`**
Another `bp_connect.sh` is still running on a different terminal. Either
Ctrl-C that one or override the port: `LOCAL_PORT=9002 bp_connect.sh`, then
browse to `http://127.0.0.1:9002/...`.

**Browser says "Connection refused" or "URL not found"**
Tunnel is up but Jupyter isn't responding yet. Wait 30 s, refresh. If it
persists, run `ssh bigpurple "squeue -u $USER"`: if the job died, run
`bp_connect.sh` again — it'll submit a new one.

**Matplotlib clicks aren't registering in `collect_prompts.ipynb`**
Click anywhere on the image once before the "real" clicks; `%matplotlib
widget` sometimes misses the very first event while it sets up. For the
number-key shortcuts to switch object id, make sure the canvas has focus
(click on it once first).

**Jupyter Lab UI is slow to load**
First load through an SSH tunnel pulls ~5 MB of JS. ~30 s is normal.
After it loads, it's snappy.

**SLURM job pending forever**
Check `squeue -p oermannlab`. If the partition is full, your job will wait.
Add `--time=01:00:00` (shorter walltime) to your sbatch to get prioritised
into a backfill slot — at the cost of being killed sooner. Default is 4 h.

**Need to reset everything**
```sh
ssh bigpurple "scancel -n surgsam2_jupyter && scancel -n surgsam2_array"
```
Then `bp_connect.sh` again.

## File map

| Path on bigpurple                                         | What                                                |
| --------------------------------------------------------- | --------------------------------------------------- |
| `/gpfs/data/oermannlab/users/schula12/Surgical-SAM-2/`    | The repo (branch `training`)                        |
| `…/.venv/`                                                | uv-managed Python 3.11 env (torch+SAM2+jupyter)     |
| `…/checkpoints/sam2.1_hiera_s_endo18.pth`                 | The finetuned SurgSAM-2 weights (184 MB)            |
| `…/prompts/<video>.json`                                  | Per-video click prompts                             |
| `…/results/<video>/seed_<k>/`                             | Inference output (mp4 + masks + log.json)           |
| `…/logs/jupyter_<jobid>.{out,err}`                        | Per-jupyter-job logs                                |
| `…/logs/array_<jobid>_<idx>.{out,err}`                    | Per-batch-task logs                                 |
| `/gpfs/data/oermannlab/private_data/whip/frames_attempt2/` | The whip frames (read-only; ~484 GB; not touched)  |
| `…/whip/raw/`                                              | Source MP4s (~3.7 TB; not used in current pipeline)|
