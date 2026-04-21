# StereoVision 3D — Building AI for the 3D World

> **From rigid geometric kernels to latent optimisation — we aren't just matching pixels anymore; we are learning the underlying structure of reality.**

An interactive, visually stunning demo application that explores the full depth of modern stereo vision: from the immutable physics of triangulation to the frontier of Vision-Language-Action models. Built for engineers, researchers, and anyone who wants to understand how machines perceive the 3D world.

---

## Quick Start

```bash
cd StereoVision

# Create and activate conda environment
conda create -n stereovision python=3.13 -y
conda activate stereovision

# Install dependencies
pip install -r requirements.txt

# Launch
python app.py
```

Open **http://localhost:5555** in your browser.

---

## The Theory — Grounded

### 1. The Immutable Physics: Triangulation

Stereo depth estimation rests on a single, non-negotiable equation derived from similar triangles:

```
        b . f
Z  =  -------
          d
```

| Symbol | Meaning | Unit |
|--------|---------|------|
| `Z` | Depth (distance to object) | metres |
| `b` | Baseline (distance between camera centres) | metres |
| `f` | Focal length | pixels |
| `d` | Disparity (horizontal pixel shift between left/right views) | pixels |

**Derivation.** Given two pinhole cameras separated by baseline `b`, a 3D point `P = (X, Y, Z)` projects onto the left image at `x_L = f*X/Z` and the right image at `x_R = f*(X - b)/Z`. The disparity is:

```
d = x_L - x_R = f*b / Z
```

Rearranging gives `Z = b*f / d`. This is geometry — it doesn't change. What changes is how we *estimate* `d`.

**Key insight:** As `d -> 0`, `Z -> infinity`. This is why stereo fundamentally struggles at long range — the signal-to-noise ratio of disparity vanishes. This is not a software limitation; it's physics.

---

### 2. The Paradigm Shift: From Geometry to Latent Optimisation

#### Legacy: Single-Pass Cost Volume (PSMNet, 2018)

The classical deep stereo pipeline:

1. Extract features `F_L, F_R` via a shared CNN backbone
2. Construct a **4D cost volume** `C(x, y, d) = concat(F_L(x,y), F_R(x-d, y))` for `d in [0, D_max]`
3. Regularise with **3D convolutions** over the `(H, W, D)` volume
4. Soft argmin to produce the final disparity

**The problem:** 3D convolutions over a `H x W x D x C` tensor are `O(H*W*D*C^2*k^3)` — computationally brutal. Worse, the fixed `D_max` creates an inductive bias that fails on out-of-distribution depth ranges.

#### Modern: Iterative Refinement (RAFT-Stereo, 2021+)

The key insight: treat disparity as a **latent state** refined through recurrent updates.

```
Initialise:  d_0 = 0  (or a coarse estimate)

For k = 0, 1, 2, ..., K:
    1. Look up correlation features:  corr_k = Lookup(C, d_k)
    2. Update hidden state:           h_k = ConvGRU(h_{k-1}, [corr_k, d_k, context])
    3. Predict residual:              delta_d = Head(h_k)
    4. Update disparity:              d_{k+1} = d_k + delta_d
```

This mimics the human visual system's ability to **converge** on a scene — your eyes don't compute depth in one shot; they iteratively adjust vergence until the scene "locks in."

**Why this works better:**
- **No fixed D_max** — the model can refine to arbitrary disparity ranges
- **Amortised computation** — early iterations handle coarse structure, later iterations refine edges
- **Graceful degradation** — you can trade compute for accuracy by adjusting `K`

---

### 3. The All-Pairs Correlation Volume

Instead of a concatenation-based cost volume, modern architectures compute a **correlation volume**:

```
C(i, j) = <F_L(i), F_R(j)>    for all (i, j) along the epipolar line
```

This is a dot-product similarity between every left feature and every right feature at the same scanline. The result is a `H x W x W` tensor (or `H x W x D_max` if truncated).

**The correlation lookup** at iteration `k`:
- Given current estimate `d_k` at pixel `(x, y)`
- Sample correlation values in a local neighbourhood around `(x, y, d_k)`
- This provides the gradient signal: "should I increase or decrease `d`?"

---

### 4. Cross-Attention: Stereo as Sequence-to-Sequence

The most elegant formulation treats stereo matching as a **sequence-to-sequence** problem. For each scanline, the left image features are queries and the right image features are keys/values:

```
Attention(Q_L, K_R, V_R) = Softmax( Q_L * K_R^T / sqrt(d_k) ) * V_R
```

Where:
- `Q_L = W_Q * F_L` — queries from left image (what am I looking for?)
- `K_R = W_K * F_R` — keys from right image (what's available?)
- `V_R = W_V * F_R` — values from right image (what information to transfer?)
- `d_k` — dimension of key vectors (for numerical stability)

**Why this is powerful:**

1. **Uniqueness constraint** — Softmax normalisation ensures each query pixel distributes attention across candidates, naturally enforcing soft uniqueness
2. **Global receptive field** — Unlike fixed correlation windows, attention can match features at any disparity
3. **Learned matching** — The projection matrices `W_Q, W_K, W_V` learn task-specific similarity metrics that go beyond raw dot-product correlation

---

### 5. The SGBM Algorithm (What This Demo Uses)

This demo implements **Semi-Global Block Matching (SGBM)**, the gold standard of classical stereo. The energy function:

```
E(d) = sum_p [ C(p, d_p) + sum_{q in N(p)} P1 * I(|d_p - d_q| = 1) + P2 * I(|d_p - d_q| > 1) ]
```

| Term | Meaning |
|------|---------|
| `C(p, d_p)` | Matching cost at pixel `p` with disparity `d_p` (Census/BT) |
| `P1` | Penalty for small disparity changes (smooth surfaces) |
| `P2` | Penalty for large disparity jumps (depth discontinuities) |

**Semi-Global** means we aggregate costs along **8 or 16 directions** (not just left-right), then take the minimum. This approximates 2D MRF optimisation at `O(W*H*D)` instead of NP-hard global optimisation.

The demo also applies **WLS (Weighted Least Squares) filtering** when available, which uses the left-right consistency check to fill holes and smooth the disparity map while preserving edges.

---

### 6. From Depth to 3D: Point Cloud Reconstruction

Given a disparity map `d(x, y)`, we reconstruct 3D coordinates:

```
Z(x, y) = b * f / d(x, y)
X(x, y) = (x - c_x) * Z(x, y) / f
Y(x, y) = (y - c_y) * Z(x, y) / f
```

Where `(c_x, c_y)` is the principal point (image centre). Each pixel becomes a coloured 3D point — the result is a **dense point cloud** that you can rotate and explore in the interactive 3D viewer.

---

### 7. The Bridge to Vision-Language-Action (VLA)

In the regime of Embodied AI, depth estimation has evolved from a narrow geometric task into a **fundamental survival metric**. The integration path:

```
Stereo Pair -> Disparity -> Dense Depth -> 3D Tokens -> VLA Backbone -> Action

token_action = f(token_vision, token_language, token_depth)
```

**Three capabilities that enable this:**

| Capability | What It Solves | Why It Matters |
|-----------|---------------|----------------|
| **Semantic Hallucination** | Predicts depth in occluded/specular regions using learned world priors | Classical geometry returns NaN for glass, shadows, reflections |
| **Zero-Shot Generalisation** | Deploys across domains without fine-tuning | Factory floors, kitchens, outdoor — one model |
| **Action-Space Grounding** | 60+ FPS dense 3D priors as VLA tokens | The LLM backbone "perceives" physical distance as a first-class citizen |

---

## Application Architecture

```
StereoVision/
├── app.py                  # Flask backend — stereo processing API
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Single-page application
└── static/
    ├── css/
    │   └── style.css       # Dark theme, glassmorphism, animations
    └── js/
        └── main.js         # Three.js 3D viewer, interactive controls
```

### Backend Pipeline (app.py)

```
Input Image
    |
    ├── Simulate Stereo Pair (horizontal shift)
    |       |
    |       ├── Left View
    |       └── Right View
    |
    ├── SGBM Disparity Computation
    |       |
    |       └── WLS Filtering (if ximgproc available)
    |
    ├── Disparity -> Depth Map (Z = b*f/d)
    |
    └── Depth -> 3D Point Cloud (back-projection)
            |
            └── JSON response -> Three.js renderer
```

### Frontend Sections

| Section | Maps to LinkedIn Post Section |
|---------|-------------------------------|
| Hero + Stats | Title + key metrics |
| Pipeline Cards | Paradigm Shift |
| Interactive Demo | Hands-on stereo processing |
| Triangulation Playground | Z = (b*f)/d |
| Architecture Timeline | PSMNet -> RAFT-Stereo -> Foundation Models |
| Cross-Attention Viz | The Math of Cross-Attention |
| VLA Bridge | Vision-Language-Action |

---

## Technical Notes

- **Stereo simulation:** Since we take a single image as input, we simulate a stereo pair by horizontally shifting the image. This is a simplification — real stereo uses two physically separated cameras. The disparity map will reflect the synthetic shift, but the pipeline (SGBM -> depth -> point cloud) is identical to real stereo.

- **SGBM parameters:**
  - `numDisparities`: Maximum disparity range to search. Higher = can detect closer objects, but slower.
  - `blockSize`: Size of the matching window. Larger = smoother but less detail.
  - `P1, P2`: Smoothness penalties. Set heuristically as `P1 = 8*3*blockSize^2`, `P2 = 32*3*blockSize^2`.

- **Point cloud density:** We subsample every 3rd pixel for performance. The Three.js viewer uses OrbitControls for interactive exploration.

---

## References

1. Chang & Chen, *"Pyramid Stereo Matching Network"*, CVPR 2018
2. Lipson et al., *"RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching"*, 3DV 2021
3. Li et al., *"Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation"*, TPAMI 2023
4. Hirschmuller, *"Stereo Processing by Semiglobal Matching and Mutual Information"*, TPAMI 2008
5. Vaswani et al., *"Attention Is All You Need"*, NeurIPS 2017

---

*Built as a companion demo for the LinkedIn post: Stereo Vision — Building AI for the 3D World*
