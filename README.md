# SIET talk demos: From Equation to World Models

| Notebook | Slide section | What it shows | Runtime (laptop) |
|---|---|---|---|
| `01_tensors_and_PINN.ipynb` | Tensors + PINNs | Tensors, einsum (stress, inertia, attention), autograd; a PINN predicts a damped oscillator from 10 points and discovers the friction coefficient | ~2 min |
| `02_fourier_neural_operator.ipynb` | Neural operators | Burgers equation: classical solver vs a Fourier Neural Operator built from scratch, speed-up, zero-shot super-resolution, "try your own" cell | ~2 min first run, instant after (cached) |
| `03_world_model_vjepa2.ipynb` | World models (Hugging Face) | Meta V-JEPA 2: PCA of what the vectors see, "surprise" on physically impossible videos, action recognition, your own phone video | ~3–5 min |

## Tonight (with internet)

```bash
cd ~/Documents/SIET
bash setup_mac.sh     # creates .venv, installs packages, downloads models (~2.5 GB)
bash run_all.sh       # rehearsal: runs all 3 notebooks and saves outputs inside them
```

`run_all.sh` leaves every plot saved in the notebooks. If something breaks live, just scroll through the pre-run outputs.

## During the talk

```bash
cd ~/Documents/SIET && source .venv/bin/activate && jupyter lab
```
Choose the kernel **Python 3 (SIET demo)**.

- **Demo 2:** `burgers_data.npz` and `fno_burgers.pt` are cached after the first run, so the notebook loads instantly. Delete `fno_burgers.pt` if you want to train live (~1 min).
- **Demo 3, live wow moment:** record a 3-second phone video (drop a ball, roll a bottle, pour water), AirDrop it into this folder as `my_video.mp4`, and run the last code cell.
- Everything works offline once the models are downloaded.

## Extra world-model tests
`04_more_physics_videos.ipynb` runs 5 more scenes (pendulum, collision, hidden ball, table edge, ramp), each with a physical and an impossible version, and prints a "surprise vs normal" table. It also runs every clip you drop into the `videos/` folder, and its last cell saves all synthetic scenes to `videos/generated/` as .mp4 files for full-screen playback.
# physics_neural_network
