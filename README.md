# From Equation to World Models: AI and the Simulation of Physics

Code and slides from the SIET talk by **Balaji Muthusubramanian**.

The talk followed one question, *"where will this ball land?"*, through three ways of answering it:

1. **Write the law:** Newton's equations, solved numerically.
2. **Learn from data, guided by the law:** physics-informed neural networks and neural operators.
3. **Learn by watching:** a world model that picks up intuitive physics from video.

Every demo from the talk is here as a Jupyter notebook. Run them, break them, change the numbers, and see what happens.

---

## What's inside

| File | What you'll do | Runs on a laptop? |
|---|---|---|
| `01_tensors_and_PINN.ipynb` | Vectors, matrices and tensors in PyTorch; `einsum` for stress, inertia and attention; autograd. Then a **physics-informed neural network** predicts a damped spring's motion from just 10 data points and **discovers the friction coefficient** from noisy data. | Yes, about 2 min |
| `02_fourier_neural_operator.ipynb` | Solve the **Burgers equation** with a classical spectral solver, then build a **Fourier Neural Operator** from scratch that learns the solver itself. Compare accuracy and speed, and run it on a finer grid than it was trained on. | Yes, about 2 min |
| `03_world_model_vjepa2.ipynb` | Load Meta's **V-JEPA 2** world model from Hugging Face. See what its internal vectors "see", measure its **surprise** at physically impossible videos, and let it name the action in **your own phone video**. | Yes; downloads about 2.5 GB of model weights the first time |
| `04_more_physics_videos.ipynb` | Five more physics tests for the world model (pendulum, collision, hidden ball, table edge, ramp), each with a physical and an impossible version. | Yes |
| `physics_videos.py` | The code that draws the synthetic test videos. Easy to extend with your own scenes. | |
| `From_Equation_to_World_Models.pptx` | The slides. | |
| `burgers_data.npz`, `fno_burgers.pt` | Cached training data and a trained FNO, so notebook 2 runs instantly. Delete them to regenerate and retrain from scratch. | |
| `videos/generated/` | The synthetic test videos as `.mp4` files. | |

---

## Getting started

You need **Python 3.10 or newer** and about **4 GB of free disk space** (mostly for the world model in notebooks 3 and 4). No GPU is needed.

### macOS / Linux

```bash
git clone https://github.com/astrobalaji/physics_neural_network.git
cd physics_neural_network
bash setup_mac.sh          # creates .venv, installs packages, downloads the models
source .venv/bin/activate
jupyter lab
```

### Windows (PowerShell)

```powershell
git clone https://github.com/astrobalaji/physics_neural_network.git
cd physics_neural_network
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prefetch_models.py   # optional: downloads the world model now instead of later
jupyter lab
```

When a notebook opens, pick the kernel **Python 3 (SIET demo)**, or any Python 3 kernel from your `.venv`, then run the cells top to bottom with **Shift + Enter**.

### Google Colab (no installation)

Notebooks 1 and 2 run as-is on Colab. Open them straight from GitHub:

- [Open notebook 1 in Colab](https://colab.research.google.com/github/astrobalaji/physics_neural_network/blob/main/01_tensors_and_PINN.ipynb)
- [Open notebook 2 in Colab](https://colab.research.google.com/github/astrobalaji/physics_neural_network/blob/main/02_fourier_neural_operator.ipynb)

Notebooks 3 and 4 need the helper file and cached data from the repository. Run this in the first cell of a new Colab notebook, then open them from the file browser on the left:

```python
!git clone https://github.com/astrobalaji/physics_neural_network.git
%cd physics_neural_network
!pip install -q "transformers>=4.53" opencv-python
```

---

## Suggested path

1. **Notebook 1, Part 1:** play with the grid-warp cell. Change the matrix `W` and watch how space bends. That picture *is* a neural network layer.
2. **Notebook 1, Part 2:** run the plain network and the PINN side by side. Why does the plain one fail after `t = 0.36`?
3. **Notebook 2:** use the *"Try your own initial condition"* cell. Find a starting shape where the FNO fails. Why does it fail there?
4. **Notebook 3:** record a 3-second video on your phone, save it next to the notebook as `my_video.mp4`, and run the last cell.
5. **Notebook 4:** which kinds of impossible events does the world model notice, and which does it miss?

---

## Things to try (mini-projects)

**Physics-informed networks (notebook 1)**
- Remove the physics loss (`lam = 0`) and confirm you get the plain network back. Then try `lam = 1e-2`. What goes wrong when it's too large?
- Give the PINN fewer data points: 5, 3, or just the initial condition. How far can you push it?
- Swap in a different equation: a **projectile with air drag** ($\ddot y = -g - c\,\dot y$) or a **pendulum** ($\ddot\theta = -\tfrac{g}{L}\sin\theta$).
- In the inverse problem, make `k` (the spring constant) learnable too. Can it find both constants at once?

**Neural operators (notebook 2)**
- Change `modes` from 16 to 4 or 32. What happens to accuracy and speed?
- Make the physics harder: lower the viscosity `NU` to `0.02` so sharper fronts form. Does the FNO keep up?
- Train with only 100 simulations instead of 1,000. How much data does it really need?
- Extend it to 2-D (the heat equation or 2-D Burgers). The [`neuraloperator`](https://neuraloperator.github.io) library has ready-made 2-D FNOs.

**World models (notebooks 3 and 4)**
- Record **pairs of real videos with the same start**. For example, roll a ball behind a book and let it come out; then do it again but secretly stop it. Which one surprises the model more?
- Add a new scene to `physics_videos.py`. A simple pattern: copy `table_edge`, change the physics, and add it to `SCENES`.
  - Ideas: a ball bouncing *higher* each time, an object changing shape behind a screen, two balls sticking together.
- Compare cartoon videos with real footage of the same event. Is the model better on real video? Why might that be?

---

## Key ideas from the talk

- **An AI model is a function** $\hat y = f_\theta(x)$ with adjustable parameters θ. Learning is statistics (minimise the error) plus calculus (gradient descent via the chain rule).
- **Everything is a tensor.** A particle's state, an image, a video, a neural-network layer. Physics and AI share the same algebra: $t_i = \sigma_{ij}n_j$ and $\mathrm{softmax}(QK^\top/\sqrt d)\,V$ are both tensor contractions.
- **Data alone doesn't know the laws of nature.** Putting physics into the loss (PINNs) lets a network generalise from very little data.
- **A neural operator learns the solver, not one solution.** It maps a whole input function to a whole output function, and the Fourier version works like a learned equaliser on frequencies.
- **A world model learns from observation.** V-JEPA 2 predicts the *meaning* of the future rather than its pixels, and its prediction error behaves like surprise.

---

## Further reading

- **PINNs:** Raissi, Perdikaris & Karniadakis, *Physics-informed neural networks* (J. Comp. Phys., 2019). Ben Moseley's blog post *"So, what is a physics-informed neural network?"* is a very good gentle introduction.
- **Fourier Neural Operator:** Li et al., *Fourier Neural Operator for Parametric Partial Differential Equations* (ICLR 2021) · [neuraloperator.github.io](https://neuraloperator.github.io)
- **V-JEPA 2:** Meta AI, *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* (2025) · model on Hugging Face: [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)
- **Intuitive physics in video models:** Garrido et al., *Intuitive physics understanding emerges from self-supervised pretraining on natural videos* (2025)
- **Scientific foundation models:** [Polymathic AI](https://polymathic-ai.org) and *The Well* dataset of physics simulations
- **Linear algebra refresher:** 3Blue1Brown, *Essence of Linear Algebra* (YouTube)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named torch` (or similar) | Activate the environment first: `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`), or select the `.venv` kernel in Jupyter. |
| Notebook asks for kernel "Python 3 (SIET demo)" | Choose any Python 3 kernel that has the packages installed. |
| V-JEPA model download is slow or fails | Run `python prefetch_models.py` once on a good connection. The models are cached in `~/.cache/huggingface`. |
| Notebook 3 is slow | Normal on a CPU-only machine: each video takes a few seconds. Reduce `N_THROWS` / `N_SEEDS` to 1. |
| `could not read frames` for your video | Make sure the file is in the same folder as the notebook and named exactly `my_video.mp4`. Try converting `.mov` to `.mp4`. |
| Apple Silicon errors mentioning MPS | The notebooks already set `PYTORCH_ENABLE_MPS_FALLBACK=1`. If needed, force the CPU by setting `device = "cpu"`. |

---

Questions, bugs, or a cool experiment to share? Open an issue on this repository.
