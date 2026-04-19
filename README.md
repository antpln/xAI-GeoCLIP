# Explainable Image Geo-localization — GeoCLIP Reimplementation

Reimplementation of **GeoCLIP** (Vivanco et al., NIPS 2023) in pure PyTorch,
extended with an interpretability pipeline to verify that the model relies on
genuine geographic visual cues rather than spurious correlations.

> Vivanco Cepeda, V., Nayak, G. K., & Shah, M. (2023).
> *GeoCLIP: Clip-Inspired Alignment between Locations and Images for Effective Worldwide Geo-localization.*
> NeurIPS 2023.

---

## What GeoCLIP does

GeoCLIP frames geo-localization as **image-to-GPS retrieval** via contrastive
learning. Given an image and a gallery of GPS coordinates, the predicted
location is the gallery point whose embedding is most similar to the image
embedding.

The architecture has two encoders:

- **Image encoder** — a CLIP Vision Transformer (ViT) that maps an image to a
  unit-norm vector in a shared embedding space.
- **Location encoder** — a GPS encoder that maps (lat, lon) coordinates to the
  same space, using Random Fourier Features (RFF) followed by an MLP.

Training minimizes a symmetric InfoNCE loss that pulls each image embedding
toward its correct GPS embedding while pushing it away from all other GPS
embeddings in the batch.

---

## Our reimplementation

Every component is written in PyTorch from scratch.
We do not import or call the official `geoclip` package at any point.

### Vision Transformer (`geoclip/models/vit.py`)

The paper calls `clip.load("ViT-L/14")` from the OpenAI CLIP package and
treats the visual encoder as a black box. We implement every layer ourselves:

| Module | What it does |
|---|---|
| `PatchEmbedding` | Strided `Conv2d(stride=patch_size)` extracts non-overlapping patches and projects them to `hidden_dim`. Equivalent to the linear patch projection in the ViT paper. |
| `MultiHeadSelfAttention` | Explicit `q_proj`, `k_proj`, `v_proj`, `out_proj` linear layers. Computes scaled dot-product attention `softmax(QKᵀ / √d_k) V`. Returns attention weights `[B, heads, S, S]` when requested. |
| `MLP` | Two-layer feedforward: `fc1 → GELU → fc2`. Hidden dimension is 4× the token dimension, matching the CLIP ViT specification. |
| `TransformerBlock` | Pre-norm: `x + MHSA(LN(x))`, then `x + MLP(LN(x))`. CLIP uses pre-norm, unlike the original "Attention is All You Need" which uses post-norm. |
| `VisionTransformer` | Assembles the above: learnable CLS token and positional embeddings, `pre_norm` applied once after positional embedding addition, `N` blocks, `post_norm` applied to the CLS token output. |

CLIP pre-trained weights are loaded into our architecture once at
initialisation via an explicit key-by-key mapping (`load_clip_weights` in
`vit.py`). HuggingFace is used only to fetch this state dict and is freed
immediately after. All forward passes run through our own modules.

The `output_attentions` flag is threaded through
`VisionTransformer → TransformerBlock → MultiHeadSelfAttention`, so every
layer's attention matrix `[B, heads, S, S]` is returned natively as a return
value — with no hooks, no monkey-patching, no side channels.

### Location encoder (`geoclip/models/gps_encoder.py`)

Follows the paper exactly:

1. Normalize (lat, lon) to `[-1, 1]` by dividing by (90, 180).
2. For each scale `k = 0, …, 9`: draw a fixed random matrix
   `B_k ~ N(0, 2^{2k} I)` of shape `[rff_dim, 2]`, registered as a
   non-trainable buffer.  Compute `[cos(x Bₖᵀ), sin(x Bₖᵀ)]` to produce
   `2 × rff_dim` features.
3. Concatenate all scales: total RFF dimension = `num_scales × rff_dim × 2 = 5120`.
4. Pass through a 3-layer MLP with GELU activations and a LayerNorm before
   the final projection.
5. L2-normalize the output.

### Contrastive objective (`geoclip/losses/infonce.py`)

Symmetric InfoNCE on the `[B, B]` image-GPS similarity matrix, with a
learnable log-temperature clamped to `[-4.6, 4.6]` after each optimizer step.

---

## Extensions beyond the paper

The paper provides no interpretability mechanism and uses the standard InfoNCE
loss with uniform random negatives. We add three targeted extensions:

### 1. Attention entropy regularization (`geoclip/losses/attention_entropy.py`)

An auxiliary training loss that penalizes high-entropy attention distributions
in the last transformer block:

```
L_attn = - Σ_h  Σ_s  a^{(h)}_{CLS,s}  log a^{(h)}_{CLS,s}
```

where `a^{(h)}_{CLS,s}` is the attention weight from the CLS token to patch `s`
in head `h`. Minimizing this encourages the model to concentrate attention on
a small number of patches rather than averaging uniformly over the image.

**Why it reinforces interpretability:** a model that attends diffusely produces
Grad-CAM and Attention Rollout maps that spread over the whole image, making
it impossible to identify which visual cues drive the prediction. Penalizing
entropy during training directly shapes the attention structure that the
interpretability tools read.

Total loss: `L = L_{InfoNCE} + λ · L_attn`, with `λ = 0.01` by default.
The regularization runs every `attn_reg_every = 4` steps to limit the memory
overhead of storing `L × [B, heads, S, S]` attention tensors.

### 2. Hard geographic negative mining (`geoclip/training/hard_negatives.py`)

Standard InfoNCE draws negatives uniformly from the batch. Many such negatives
are both geographically and visually dissimilar from the anchor, making them
trivially easy to separate — the model can exploit spurious cues (image
compression, camera white balance, sky color) without learning any geographic
content.

We replace a fraction of the GPS negatives with *hard* ones: for each anchor
image `i`, we find the batch sample `j ≠ i` whose GPS coordinate is closest
(by haversine distance), and with probability `swap_prob = 0.5` substitute
`j`'s GPS embedding as `i`'s negative in the image-to-GPS direction, provided
the two GPS points are within `min_distance_km = 500 km` of each other.

This is computed entirely in-batch from already-available `coords` — it adds
no data-loading overhead and requires no pre-mining pass.

**Why it reinforces interpretability:** if the model can be fooled by a GPS
point 200 km away from the correct one, it is not attending to fine-grained
geographic cues. Hard negatives force it to.

### 3. Three complementary interpretability methods

The paper provides no visualization. We implement three independent methods
that answer different questions:

| Method | File | Question answered | Requires gradients |
|---|---|---|---|
| **Grad-CAM** | `geoclip/interpretability/gradcam.py` | Which patches most increased the image-GPS similarity score? | Yes |
| **Attention Rollout** | `geoclip/interpretability/attention_rollout.py` | Which patches does the CLS token draw information from, accumulated across all 12 layers? | No |
| **Occlusion Sensitivity** | `geoclip/interpretability/occlusion.py` | Which patches, when masked out, shift the predicted GPS location the most? | No |

When Grad-CAM and occlusion sensitivity agree on which regions matter, and
when those regions correspond to recognizable geographic features (signage,
architecture, vegetation type, road markings), the model is behaving as
intended. When they disagree, or when they highlight backgrounds or
compression artifacts, a spurious correlation is likely present.

**Grad-CAM** hooks into the output of `vit.blocks[-1]` and back-propagates
the cosine similarity between the image embedding and the target GPS embedding.
Because `TransformerBlock` returns `(hidden, attn_weights)`, the hook captures
`hidden[:, 1:, :]` (the patch tokens) from the forward output and the
corresponding gradients from the backward pass, then computes
`cam = ReLU(mean_grad × activation)`.

**Attention Rollout** calls `model.image_encoder(images, output_attentions=True)`,
which returns all 12 attention matrices directly as return values. It then
accumulates `R = (Â_12 + I) ⊗ … ⊗ (Â_1 + I)` where each `Â_l` has residual
connection added and rows renormalized, and reads `R[CLS, 1:]` as the final
attribution over patch tokens.

**Occlusion sensitivity** runs `N_patches + 1` forward passes (one baseline
plus one per patch). For each patch `(r, c)`, it replaces the corresponding
`patch_size × patch_size` pixels with a constant fill value and measures
either (a) the cosine distance from the original embedding, or (b) the
haversine distance between the original and occluded GPS predictions. GPS mode
is more directly interpretable for the geo-localization task.

---

## Project structure

```
notebooks/
├── colab_train_eval.ipynb  # Colab notebook: training, evaluation, interpretability
└── 03_interpretability_demo.ipynb  # Local demo of all six interpretability methods
geoclip/
├── models/
│   ├── vit.py              # Vision Transformer from scratch
│   ├── image_encoder.py    # ViT + projection head + weight loading
│   ├── gps_encoder.py      # Multi-scale RFF + MLP
│   └── geoclip.py          # Top-level model + temperature
├── losses/
│   ├── infonce.py          # Symmetric InfoNCE
│   └── attention_entropy.py # Entropy regularization (extension)
├── training/
│   ├── trainer.py          # Training loop (AMP, hard negatives, attn reg)
│   ├── hard_negatives.py   # Hard geographic negative mining (extension)
│   ├── evaluator.py        # GCD metrics, recall@km
│   └── scheduler.py        # Cosine LR with linear warmup
├── interpretability/
│   ├── gradcam.py          # Grad-CAM (extension)
│   ├── attention_rollout.py # Attention Rollout (extension)
│   └── occlusion.py        # Occlusion sensitivity (extension)
├── data/
│   ├── dataset.py          # OSV-5M HuggingFace loader
│   ├── transforms.py       # CLIP normalization constants
│   └── gallery.py          # GPS gallery construction + embedding cache
└── utils/
    ├── geo_math.py         # Haversine distance
    ├── config.py           # YAML config dataclasses
    └── checkpoint.py       # Save / load checkpoints
scripts/
├── train.py                # Training entry point
├── evaluate.py             # Standalone evaluation
└── visualize.py            # Grad-CAM / Rollout / Occlusion overlay figures
configs/
├── default.yaml            # 50k samples, ViT-B/16
└── small_experiment.yaml   # 5k samples, ViT-B/32, for smoke tests
```

---

## Configuration

All hyper-parameters live in a YAML file passed via `--config`. The file is
split into five sections; any missing key falls back to the dataclass default.

### `model`

| Parameter | Default | What it controls |
|---|---|---|
| `clip_backbone` | `"ViT-B/16"` | ViT variant to load CLIP weights from. Options: `"ViT-B/32"`, `"ViT-B/16"`, `"ViT-L/14"`. |
| `freeze_layers` | `9` | Freeze the first N transformer blocks. `0` = fully fine-tune the ViT; `12` = freeze the whole backbone. Typical: freeze early layers (texture) and train later layers (semantics). |
| `rff_num_scales` | `10` | Number of frequency scales in the GPS encoder. Scale `k` uses `σ_k = 2^k`, so higher k captures finer spatial detail. |
| `rff_dim` | `256` | Random Fourier Features per scale. Total GPS feature dimension = `num_scales × rff_dim × 2`. |
| `mlp_hidden` | `1024` | Hidden size of the GPS encoder MLP. |
| `embedding_dim` | `512` | Shared embedding dimension for both encoders. Must match the CLIP projection head output. |

### `data`

| Parameter | Default | What it controls |
|---|---|---|
| `subset_size` | `50000` | Limit training to the first N samples. Set to `null` for the full 5M dataset. |
| `streaming` | `false` | Use HuggingFace streaming mode (no disk cache, slower, works with limited storage). |
| `num_workers` | `4` | DataLoader worker processes. |
| `pin_memory` | `true` | Pin CPU memory for faster GPU transfers. |

### `gallery`

The gallery is the set of GPS candidates used at retrieval time. The model picks the gallery point whose embedding is closest to the image embedding.

| Parameter | Default | What it controls |
|---|---|---|
| `size` | `10000` | Number of GPS points in the gallery. |
| `strategy` | `"train_sample"` | `"train_sample"` samples real OSV-5M coordinates (land-biased); `"uniform"` samples uniformly over the globe including oceans. |
| `cache_path` | `"gallery.pt"` | Where to cache the built gallery so it is not rebuilt every run. |

### `training`

| Parameter | Default | What it controls |
|---|---|---|
| `batch_size` | `128` | Samples per step. Larger batches = more negatives per step for InfoNCE, which is generally better. |
| `epochs` | `30` | Total training epochs. |
| `warmup_epochs` | `2` | LR rises linearly from 0 during these epochs, then follows cosine decay. Prevents large gradient steps while the projection head is still random. |
| `lr_clip` | `1e-5` | Learning rate for the ViT backbone. Low because it starts from CLIP pre-trained weights. |
| `lr_gps` | `1e-4` | Learning rate for the GPS encoder. Higher because it is trained from scratch. |
| `lr_temp` | `1e-3` | Learning rate for the learnable temperature `log σ`. Converges quickly. |
| `weight_decay` | `0.1` | AdamW weight decay. Applied to all parameter groups. |
| `amp` | `true` | Enable automatic mixed precision (FP16 forward, FP32 gradients). Recommended on any modern GPU. |
| `log_every` | `50` | Print loss + temperature to the progress bar every N steps. |
| `eval_every` | `1` | Run gallery-based evaluation every N epochs. |
| `checkpoint_dir` | `"checkpoints/"` | Directory for saved checkpoints. Each epoch writes `epoch_NNN.pt`; the best is also written to `best.pt`. |
| `hard_neg_swap_prob` | `0.5` | Probability of replacing a GPS negative with a geographically hard one. `0.0` disables the extension. |
| `hard_neg_min_dist_km` | `500` | Only GPS pairs closer than this are considered hard negatives. Pairs farther apart are easy enough without swapping. |
| `lambda_attn` | `0.01` | Weight of the attention entropy loss relative to InfoNCE. `0.0` disables the extension. |
| `attn_reg_every` | `4` | Run the entropy loss every N steps. Attention maps are memory-heavy (`L × [B, heads, S, S]`), so every step is prohibitive. |

### `evaluation`

| Parameter | Default | What it controls |
|---|---|---|
| `thresholds_km` | `[1, 25, 200, 750, 2500]` | Distance thresholds for recall@km metrics. A prediction counts as correct if it is within `t` km of the true location. |

---

## Quick start

**Google Colab** — open `notebooks/colab_train_eval.ipynb` directly in Colab.
It handles setup, training, evaluation, and a Grad-CAM sanity check, and
saves checkpoints to Google Drive so sessions can be resumed.

**Local**:

```bash
pip install -r requirements.txt

# Smoke test (~5 min, no GPU required)
python scripts/train.py --config configs/small_experiment.yaml --device cpu

# Full training
python scripts/train.py --config configs/default.yaml --device cuda

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best.pt \
                            --config configs/default.yaml

# Generate Grad-CAM + Attention Rollout + Occlusion overlays
python scripts/visualize.py --checkpoint checkpoints/best.pt \
                             --config configs/default.yaml \
                             --method both \
                             --output_dir outputs/
```

---

## Dataset

We use **OpenStreetView-5M** (OSV-5M), available on HuggingFace as
`osv5m/osv5m`. Images are street-level photographs with GPS coordinates.
The `subset_size` config parameter limits training to a manageable fraction;
set it to `null` to use the full dataset.

---

## Dependencies

| Package | Role |
|---|---|
| `torch`, `torchvision` | Model, training, transforms |
| `transformers` | Fetching CLIP pre-trained weights once at init |
| `datasets` | OSV-5M data loading |
| `matplotlib` | Visualization overlays |
| `scikit-learn` | K-means for gallery construction (optional) |
| `pyyaml` | Config loading |
