# Kayo session log

## [2026-04-19 09:15] Red-team of Breach's Phase-1 killer-gate protocol (MOB-GATE-001 v1.0)
- **artifact**: C:/MoB Final/docs/protocols/fecam-router-gate-redteam.md
- **verdict**: APPROVE WITH AMENDMENTS (3 binding amendments required before 2026-04-26 freeze)
- **outcome**: Produced full adversarial review of Breach's preregistered gate protocol. Three critical defects found that each independently compromise the gate. D1 (outcome-branch test framework): §1.2 FAIL branch uses two-sided CI under one-sided primary test; FAIL is operationally almost unreachable at n=10. D2 (Fisher-match gate): the 2× cross-arm within-seed check does not address my R1 18× threat — R1 was about within-arm-across-seed Fisher variance; protocol checks a near-trivially-satisfied cross-arm invariant. D3 (power band): §6.5 "marginal" σ_Δ band (1.16–1.75pp) drops actual power to 46% at σ_Δ=1.75; the protocol would proceed on 50%-powered evidence while branding as "marginally powered." Six additional high/medium defects on α-calibration symmetry, RNG-state-at-first-bid equality (not just weight equality), DataLoader worker seed pinning, acceptance tests for arm-disagreement and loss equality at t=0, wall-clock trap, pilot n=3 χ²-upper-bound noise, and paper metric primacy. Acceptance test set expanded from 8 to 16. With the three binding amendments (clean ≤1 day diff), freeze can proceed on 2026-04-26.
- **top-3 binding defects**:
  - D1 §1.2/§6/§9: FAIL branch unreachable under one-sided primary test — rewrite as symmetric one-sided BCa tests (PASS upper and FAIL lower at ±Δ_practical).
  - D2 §4.4: add within-arm-across-seed Fisher-stratification check (`CV(log F̄^A) ≤ 0.5`); the R1 18× threat is unaddressed without it.
  - D3 §6.5: tighten σ_Δ escalation threshold from 1.75pp to ~1.30pp, or pre-commit n=20 backup; current band admits 46%-powered runs.
- **follow_up**:
  - Breach reviews and authors amendment diffs (<1 day effort); Nosh signs, KAY/O countersigns per §13.
  - Jett adds acceptance tests 9–16 (RNG-state, α-shared-prepass, c_forget-bitwise, grad-graph, arms-disagree, losses-at-t0, FLOP unit test, class-order logging) before pilot seed 42.
  - If Breach pushes back on D1 or D3, escalate to Nosh before freeze — these are not stylistic.
  - Post-PASS (if it happens): the Phase-1 gate licenses Phase 2/3 but does NOT refute R1 mechanistically. β-only and α-only ablations (promised in synthesis Phase 2) are still required to claim "auction is irreducible."

## [2026-04-18 20:30] Round 1 red-team on MoB path forward
- **artifact**: C:/MoB Final/docs/research-party/round1/kayo-position.md
- **outcome**: Delivered adversarial position written BEFORE reading teammates. Central threat: FeCAM equivalence -- the auction is epiphenomenal; argmin-distance on prototypes reduces to FeCAM except in a narrow regime where the EWC term straddles the Mahalanobis gap, and that regime is dominated by Fisher-magnitude seed variance (18x per project memory). Argued 87% Split-MNIST is debug-grade not frontier; CIFAR-100 ViT leakage renders "continual learning" claims ceiling-bounded; LLM MoE pitch is under-specified against DeepSeek-V3's bias-term routing and ignores covariance-cost at d=4096. Proposed one project-killer experiment: FeCAM-Router vs MoB v2 on CIFAR-100 20T, 10 seeds. Flagged arxiv 2512.10969 as blocking. Identified what each peer specialist (Astra/Chamber/Killjoy/Sage/Fade) will miss.
- **follow_up**: (1) Confirm ownership of arxiv 2512.10969 before any further design work. (2) Run the FeCAM-Router vs MoB v2 ablation on CIFAR-100 20T with 10 seeds before scaling. (3) After peer positions are read, return for round 2 rebuttal and statistical-power audit of each design proposal.
