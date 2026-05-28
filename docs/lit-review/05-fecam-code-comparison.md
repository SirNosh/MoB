# FeCAM vs MoB — Paper, Codebase, and Gap Analysis

**Date**: 2026-04-19
**Author**: Nosh (via inline comparison during Phase-1 protocol review)
**Source paper**: Goswami et al., *FeCAM: Exploiting the Heterogeneity of Class
Distributions in Exemplar-Free Continual Learning*, NeurIPS 2023. arxiv
[2309.14062](https://arxiv.org/abs/2309.14062), [arxiv HTML v3](https://arxiv.org/html/2309.14062v3).
**Source repo**: [github.com/dipamgoswami/FeCAM](https://github.com/dipamgoswami/FeCAM)
**MoB reference implementation**: `contibualmob/prototype_store.py` at HEAD (2026-04-18).

---

## 1. Why this comparison exists

Breach's Phase-1 killer-gate protocol (`docs/protocols/fecam-router-gate.md`) defines
Arm B (FeCAM-Router) as the null-hypothesis mechanism: Arm A with β=γ=0. The protocol
§2.1 asserts:

> **Mahalanobis formulation**: FeCAM shrinkage + correlation-norm + Tukey transform
> **exactly per FeCAM paper**.

If the MoB codebase's Mahalanobis pathway deviates from FeCAM's published recipe,
Arm B is not actually FeCAM — it's a weakened variant. Either gate outcome becomes
unpublishable:
  - **PASS** → reviewers: "you beat a weakened FeCAM, not real FeCAM."
  - **TIE/FAIL** → we cannot distinguish "MoB's auction is epiphenomenal" from
    "our weakened-FeCAM Arm B was beatable by reasons unrelated to the auction."

This doc maps the deviation gap and scopes the remediation before freeze.

---

## 2. FeCAM's four architectural elements (from paper + repo)

FeCAM's contribution is a four-element recipe layered on top of a frozen feature
extractor. All four are load-bearing; the paper's ablations (Table 2) show ≥ 1–3pp
accuracy drops when any one is removed.

### 2.1 Per-class covariance (Σ_y, one per class)

Paper eq. (1):
```
𝒟_M(x, μ_y) = (x − μ_y)ᵀ Σ_y⁻¹ (x − μ_y)
```

FeCAM Table 1 (CIFAR-100, T=5): per-class Σ beats shared ("common") Σ by **2.1pp**
(70.9% vs 68.8%). Per-class is the headline configuration.

Storage: for d=768 (ViT-B/16), 100 classes × (768² × 4 bytes) = ~235 MB per task.
Not trivial but feasible.

### 2.2 Shrinkage (eq. 8)

```
Σ_s = Σ + γ₁·V₁·I + γ₂·V₂·(1−I)
```

Where:
- `V₁ = mean(diag(Σ))` — average diagonal variance
- `V₂ = mean(off-diag(Σ))` — average off-diagonal covariance
- `I` — identity matrix (so `(1-I)` is the off-diagonal indicator)
- `γ₁, γ₂` hyperparameters

**CIFAR-100 MSCIL**: γ₁ = γ₂ = 1. **FSCIL**: γ₁ = γ₂ = 100.

**This is NOT ridge regularization.** Ridge adds `ε·I` (constant). FeCAM shrinkage
adds BOTH a data-dependent multiple of identity AND a data-dependent scaling of the
off-diagonal structure — the diagonal gets boosted by the average variance and the
off-diagonal gets pushed toward the average off-diagonal covariance. This
preserves the class-specific correlation *pattern* while regularizing magnitude.

Repo implementation (`models/base.py :: shrink_cov`):
```python
def shrink_cov(self, cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0])
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_
```

### 2.3 Correlation normalization (eq. 7)

```
Σ̂_y(i,j) = Σ_y(i,j) / (σ_y(i) · σ_y(j))
```

Where `σ(i) = sqrt(Σ(i,i))`. This converts Σ to a correlation matrix: diagonals
become exactly 1, off-diagonals become Pearson correlations in [-1, 1].

**Why it matters**: different classes have different feature-magnitude scales
(a class with higher-variance features will have larger raw Σ entries). Comparing
raw Mahalanobis distances across classes then double-counts scale. Correlation
normalization strips the scale so distance comparisons are on the same footing.

Repo implementation (`models/base.py :: normalize_cov`):
```python
def normalize_cov(self):
    ...
    for cov in cov_mat:
        sd = torch.sqrt(torch.diagonal(cov))
        cov = cov/(torch.matmul(sd.unsqueeze(1), sd.unsqueeze(0)))
        norm_cov_mat.append(cov)
    return norm_cov_mat
```

### 2.4 Tukey's Ladder of Powers (eq. 9) — backbone-conditional

```
φ̃(x) = φ(x)^β       if β ≠ 0
     = log(φ(x))     if β = 0
```

Applied element-wise to both features AND class means. Default **β = 0.5**
(square-root transform).

**Why it matters (for ResNet-like backbones)**: FeCAM empirically shows that
ResNet penultimate features are skewed / non-Gaussian. Mahalanobis distance's
optimality rests on Gaussianity; the Tukey transform reduces skew and promotes
Gaussianity before the Mahalanobis assumption is invoked.

**Critical exception — ViT-B/16 (our Phase-1 backbone)**: FeCAM paper §7 explicitly
disables Tukey for ViT backbones:

> "Since ViT features contain negative values, we do not apply Tukey's
> transformation of Powers [...]"

Tukey with `β = 0.5 = 1/2` on a negative value is undefined (square root of a
negative real). The paper therefore sets Tukey OFF for all ViT-backbone experiments.
This is **not an ablation choice** — it is a mathematical constraint imposed by
the mixed-sign feature distribution. Our Phase-1 protocol (v1.2 §2.0) inherits
this: Tukey is OFF for Arm A and Arm B on ViT-B/16.

LAMDA-PILOT's `models/fecam.py` enforces this with a `tukey=false` flag in the
ViT config; `dipamgoswami/FeCAM`'s `exps/FeCAM_cifar100.json` sets `beta=0.5`
and `tukey=false` for the CIFAR-100 + ViT configuration.

**Implication for our gap analysis**: at S2 (CIFAR-100 + ViT-B/16), Tukey
being "absent" from our code is not a gap — the canonical FeCAM does not apply
it either. The gap list reduces to the other elements (per-class Σ, additive
two-parameter shrinkage, correlation normalization, L2 normalization) for our
specific backbone. Tukey becomes a gap only if we port MoB to a ResNet backbone
in any future scale variant.

Repo implementation (`models/base.py :: _tukeys_transform`):
```python
def _tukeys_transform(self, x):
    beta = self.args["beta"]
    x = torch.tensor(x)
    if beta == 0:
        return torch.log(x)
    else:
        return torch.pow(x, beta)
```

### 2.5 Mahalanobis distance with L2 normalization

The actual per-class scoring, with all four elements composed
(`models/base.py :: _mahalanobis`):

```python
def _mahalanobis(self, vectors, class_means, cov=None):
    if self.args["tukey"] and self._cur_task > 0:
        class_means = self._tukeys_transform(class_means)
    x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
    if cov is None:
        cov = torch.eye(self._network.feature_dim)
    inv_covmat = torch.linalg.pinv(cov).float().to(self._device)
    left_term = torch.matmul(x_minus_mu, inv_covmat)
    mahal = torch.matmul(left_term, x_minus_mu.T)
    return torch.diagonal(mahal, 0).cpu().numpy()
```

Note: FeCAM uses **L2-normalized features and class means** as inputs to the
(x − μ) computation. This is effectively cosine-geometry Mahalanobis — the
combination of L2 normalization + correlation-normalized Σ is what FeCAM's
heterogeneity story is built around.

---

## 3. What `contibualmob/prototype_store.py` actually implements

```python
def _recompute_inv_cov(self, allow_pinv: bool):
    if self.cov_count < self.MIN_SAMPLES_FOR_MAHALANOBIS or len(self.centroids) == 0:
        self.inv_cov = None
        return
    # Global mean
    total_sum = sum(self.class_sum[c] for c in self.class_sum)
    total_count = sum(self.class_count[c] for c in self.class_sum)
    global_mean = total_sum / total_count
    # Shared Σ = E[xx^T] - μμ^T
    cov = self.cov_sum / self.cov_count - global_mean.unsqueeze(1) * global_mean.unsqueeze(0)
    # Ridge regularization
    eps = 1e-4
    cov += eps * torch.eye(self.feature_dim, device=self.device)
    try:
        self.inv_cov = torch.linalg.inv(cov)
    except torch.linalg.LinAlgError:
        if allow_pinv:
            self.inv_cov = torch.linalg.pinv(cov)
        else:
            self.inv_cov = None
```

And distance:

```python
# Mahalanobis: d = sqrt(diff @ inv_cov @ diff^T)
diff = features.unsqueeze(1) - centroid_matrix.unsqueeze(0)  # raw, NOT L2-normalized
transformed = torch.matmul(diff, self.inv_cov)
distances = (transformed * diff).sum(dim=-1).clamp(min=0).sqrt()
```

So our implementation:
- Shared Σ per expert (FeCAM's "common covariance" variant — the 2.1pp-weaker config).
- Simple ridge: `Σ + 1e-4·I`. Constant, data-independent, no off-diagonal term.
- No Tukey transform.
- No correlation normalization.
- No L2 feature normalization.
- Incremental running-sum prototypes (vs FeCAM's one-shot-at-task-end — functionally equivalent for the mean, minor drift for the covariance).

---

## 4. Side-by-side gap table

| Element | FeCAM paper/repo | MoB `contibualmob/prototype_store.py` | Implementation gap | Severity for Phase-1 gate |
|---|---|---|---|---|
| **Per-class Σ** | Per-class `Σ_y`, one per class (primary config) | Shared Σ per expert (one Σ across all classes the expert has seen) | **MISSING per-class** | HIGH — FeCAM Table 1 shows 2.1pp gap on CIFAR-100 T=5 |
| **Shrinkage** | `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)` with γ₁=γ₂=1 | `Σ + 1e-4·I` | **WRONG SHRINKAGE FORMULA** | HIGH — this is the "F" (feature covariance) in FeCAM |
| **Correlation normalization** | `Σ̂(i,j) = Σ(i,j)/(σ(i)·σ(j))` | Not applied | **MISSING** | HIGH — paper's central claim for cross-class comparability |
| **Tukey β=0.5** | `φ^0.5` on features AND means (ResNet only; **OFF for ViT** per paper §7) | Not applied | N/A for ViT-B/16 (paper disables) / MISSING for ResNet | **N/A at S2** (our setting) / HIGH if future ResNet variant |
| **L2 normalization of features / means** | `F.normalize(x) − F.normalize(μ)` | `x − μ` (raw) | **MISSING** | MEDIUM — interacts with correlation norm; changes geometry |
| **Inverse method** | `torch.linalg.pinv` always | `torch.linalg.inv` → `pinv` fallback | Different but ≈ equivalent | LOW |
| **Prototype update cadence** | One-shot per class at task end | Incremental running mean | Different, functionally similar | LOW |
| **Minimum samples for Σ** | Implicit (covariance well-conditioned with shrinkage) | Hard 256-sample floor | Different | LOW |
| **Backbone assumption** | Frozen (ViT-B/16 or ResNet-18) | Trained (CNN per expert at S1) | Architectural | N/A for S1; binds S2 |

---

## 5. Implications

### 5.1 For Phase-1 gate (immediate)

Breach's §2.1 says Arm B must be "FeCAM **exactly per FeCAM paper**." The current
MoB Mahalanobis code implements 1 of 4 FeCAM elements (and that one — a shared Σ with
ridge — is the 2.1pp-weaker configuration). Running the gate with this code as Arm B
would be statistically honest but substantively a straw man.

**Required before gate pilot**: a faithful FeCAM implementation that Arm B invokes.
Arm A (MoB-Full) must use **the same** faithful FeCAM Mahalanobis core, differing
from Arm B only in the β·forget_cost and γ·conscience terms. This preserves Breach's
strict-paired-ablation invariant.

**Implementation options**:
1. **Port FeCAM's functions directly** from `dipamgoswami/FeCAM:models/base.py`
   (`_tukeys_transform`, `shrink_cov`, `normalize_cov`, `_mahalanobis`) into a new
   file `contibualmob/fecam_core.py` or `mob/gate/fecam_core.py`. License-check
   the FeCAM repo first; it's likely MIT/Apache.
2. **Use LAMDA-PILOT's FeCAM implementation**. LAMDA-PILOT
   ([github.com/sun-hailong/LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT))
   already includes FeCAM in its model zoo; Breach's protocol already commits to
   LAMDA-PILOT as the harness. This is the cleanest path — Arm B can call the
   harness's own FeCAM method, which is the community-canonical implementation.
3. **Reimplement**. Slow and error-prone. Avoid unless the above two fail a
   license check.

**Recommendation**: path 2 (LAMDA-PILOT's FeCAM). Single source of truth; passes
any reviewer's "is this really FeCAM?" challenge immediately.

### 5.2 For existing S1 (MNIST) results

The v2/v3 prototype-routing experiments (`results/experiments_v3/*`) were run against
the current simplified implementation. Those numbers are **not** comparable to FeCAM.
We have been reporting "MoB + prototype routing 86.7%" in `progress_report.md` as if
it were a FeCAM-class baseline. It is not — it's a weakened-FeCAM baseline. The
numbers are still internally consistent (MoB vs its own simplified prototype routing)
but the external framing must be corrected if any MNIST number appears in the paper.

**Action**: update `project.md` §6 and `timeline.md` historical entries to flag this.
Paper text should never call the current code "FeCAM" without the four-element recipe.

### 5.3 For Sage's theory

Sage's `docs/theory/polya-urn-conscience-proof.md` §3 cites the Ledoit-Wolf shrunk
covariance `Σ(λ) = (1−λ)Σ̂ + λ·(tr(Σ̂)/d)·I` as the FeCAM recipe. This is
inaccurate — FeCAM does not use Ledoit-Wolf; it uses the two-parameter
`γ₁·V₁·I + γ₂·V₂·(1−I)` additive shrinkage described above. The DSIC proof
(Prop 5.1) invokes "fixed public λ" from Astra; this generalizes to fixed public
(γ₁, γ₂), so the theorem survives, but Sage's theory writeup needs a one-line
correction to cite FeCAM's actual shrinkage formula.

**Action**: flag to Sage in the next round of theory revisions.

### 5.4 For Breach's v1.1 amendment cycle (currently running)

Breach's v1.1 work in progress does not touch Arm B's Mahalanobis implementation
(KAY/O's defects were statistical and methodology-level, not implementation-level).
So this gap is **orthogonal** to the v1.1 amendments and must be handled as a
separate pre-freeze concern.

**Action**: after Breach returns v1.1, commission Jett to wire the gate runner
against LAMDA-PILOT's FeCAM method (path 2 above), with the β=γ=0 toggle being
the only code-path difference between arms.

---

## 5.5 Resolution status (updated 2026-04-19 post-v1.2)

Breach landed a v1.1 → v1.2 amendment (§2.0, §4.6, §4.7, tests 17–19) addressing
the implementation gap identified in §5.1. The resolution diverged from path 2
in one load-bearing way:

**Composite binding, not pure LAMDA-PILOT.** Breach's v1.2 audit found that
LAMDA-PILOT's `models/fecam.py` at commit `7a6e904c5bc5cb7a4e1823b3434020be27469b63`
applies a **single-parameter** shrinkage `cov + 100·I`, NOT the paper's
two-parameter additive `Σ + γ₁·V₁·I + γ₂·V₂·(1−I)` from eq. 8. LAMDA-PILOT's
implementation is a simplification that deviates from the paper recipe on a
load-bearing element.

v1.2 therefore binds the gate against a **composite target**:
- Harness, trainer, ViT-B/16 backbone, ViT-Tukey-OFF enforcement →
  `sun-hailong/LAMDA-PILOT@7a6e904c`.
- Paper-canonical Mahalanobis recipe (additive shrinkage, correlation norm,
  L2 norm) → `dipamgoswami/FeCAM@e33f39d1`, `models/base.py`,
  `exps/FeCAM_cifar100.json` canonical config
  `{alpha1=1, alpha2=1, beta=0.5, per_class=true, full_cov=true, shrink=true, norm_cov=true}`.

This means path 2 from §5.1 as originally recommended — "use LAMDA-PILOT's
FeCAM method directly" — would have produced a non-FeCAM Arm B even after the
rebind, because LAMDA-PILOT's own shrinkage is wrong. The composite binding in
v1.2 is the technically correct target.

v1.2 tests 17, 18, 19 (protocol §8.4) CI-enforce this:
- Test 17 — byte-equivalent port fidelity to `dipamgoswami/FeCAM` upstream (≤ 1e-5).
- Test 18 — frozen_config.yaml matches paper §7 canonical values.
- Test 19 — L2 normalization produces unit vectors; Tukey is disabled on ViT.

### 5.6 For Sage's theory (updated 2026-04-19 post-v1.2)

The Ledoit-Wolf citation in `docs/theory/polya-urn-conscience-proof.md` §3
remains inaccurate (FeCAM does not use Ledoit-Wolf convex-combination shrinkage
`Σ(λ) = (1−λ)Σ̂ + λ·(tr/d)·I`; it uses additive two-parameter
`Σ + γ₁·V₁·I + γ₂·V₂·(1−I)`). Non-load-bearing corrections to references and
the bid-function recipe have been applied inline by Nosh.

The load-bearing corrections to §6.2 and §6.3 — the scale-specialization
γ_min bounds that depend on the condition number of the shrunk covariance —
**require Sage re-derivation**. Under convex-combination Ledoit-Wolf,
κ(Σ(λ)) ≤ 1/λ · tr(Σ̂)/d; under additive two-parameter shrinkage,
κ(Σ_s) ≤ (λ_max(Σ) + γ₁V₁ + γ₂V₂) / (λ_min(Σ) + min(γ₁V₁, γ₂V₂)), a different
functional form with different λ → 0 / γ → 0 limits. This affects the
§6.2 scaling `γ_min(λ) = O(α/(K√λ))` and the §6.3 analogous S3 bound.
Sage's core theorems (T1 collapse, T2 ergodicity) survive structurally; the
per-scale quantitative prescriptions need one pass of rework.

---

## 6. Sources

- FeCAM paper: [arxiv 2309.14062](https://arxiv.org/abs/2309.14062), [HTML v3](https://arxiv.org/html/2309.14062v3), [NeurIPS 2023 proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/file/15294ba2dcfb4521274f7aa1c26f4dd4-Paper-Conference.pdf)
- FeCAM code: [github.com/dipamgoswami/FeCAM](https://github.com/dipamgoswami/FeCAM)
- LAMDA-PILOT harness (includes canonical FeCAM): [github.com/sun-hailong/LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT)
- MoB current Mahalanobis: `contibualmob/prototype_store.py` (HEAD, 2026-04-18)
