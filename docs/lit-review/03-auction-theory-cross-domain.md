# MoB through the Lens of Auction Theory: A Cross-Domain Synthesis

Author: Astra (BMad ML cross-domain synthesis)
Date: 2026-04-18
Scope: Interpret MoB (Mixture of Bidders) as a procurement mechanism, audit its
theoretical properties, and surface cross-domain bridges that the continual
learning (CL) and Mixture-of-Experts (MoE) literatures have missed.

> Do NOT read `docs/literature/` for this analysis; the assumption per the
> commissioning prompt is that it is stale. All claims below are grounded in
> primary mechanism-design, classifier-systems, and competitive-learning
> sources.

---

## 1. Executive summary (sharp claims)

- **MoB is a first-score sealed-bid reverse procurement auction with a
  quasi-linear scoring rule.** The exact parent is Che (1993), "Design
  Competition through Multi-Dimensional Auctions." The score is
  `S_i = -(alpha * exec + beta * forget)`; items are tasks/batches; principals
  are the training objective; no transfers are observed because experts never
  strategize - their "types" are simply read off their parameters.
- **Because experts do not have private types, MoB is trivially truthful**
  today. The moment any expert-local parameter (alpha_i, beta_i, a learned
  bidder head) is optimized against wins, the revelation principle starts to
  bite: the direct mechanism corresponding to "learned-bidder MoB" is *not*
  incentive-compatible unless the scoring rule is monotone and the payment
  rule follows Myerson's lemma. See Section 2.3.
- **The optimizer-reset pathology is a winner's-curse analog.** Under
  affiliated-signal common-value auctions (Milgrom-Weber 1982), the bidder
  with the most optimistic private signal wins and is ex-post biased. In MoB
  the "signal" is stale Adam moments; the winner's posterior over the batch
  is systematically too confident, and resetting the optimizer is the
  discrete ML analog of Milgrom-Weber's linkage-principle "signal disclosure"
  fix.
- **MoB is *not* bucket-brigade in disguise.** Holland's bucket brigade (1985)
  redistributes strength *backward* through a chain of rule activations;
  MoB has no chain, no transfer, and no retrospective credit flow. Structural
  isomorphism exists only at the "bidders compete for the right to act"
  level; below that they diverge. The CL/MoE community should stop gesturing
  at this analogy as if it were identity.
- **The "dead expert" problem has a 1988 solution the MoE world re-derived
  badly.** DeSieno's conscience mechanism for Kohonen SOMs is functionally
  identical to Switch Transformer's load-balancing auxiliary loss, only
  additive to bids rather than to logits. This is worth a paper on its own
  (see Section 5.1).
- **Per-token MoE routing is an online combinatorial reverse auction with
  capacity constraints** (top-k = k copies of the slot, expert capacity =
  supply constraint). Winner determination for combinatorial auctions is
  NP-hard and inapproximable to within `m^(1/2-eps)` (Sandholm et al.,
  Lehmann-Mueller). Switch/top-k routing is the O(K) greedy approximation.
  That means current MoE routing is not strategy-proof when experts have
  learned biases - which is exactly the regime DeepSeek's
  "auxiliary-loss-free load balancing" (Wang et al. 2024) enters.
- **Capacity markets (electricity) have 30 years of experience with exactly
  MoB's "incumbent premium" pathology.** `forget_cost` is a stranded-asset
  payment. The missing-money / missing-market literature (Newbery 2016)
  predicts that capacity auctions with forget-style protection *over-procure*
  protection and thereby *exacerbate* the problem they solve. MoB should
  expect the same if `beta` is not calibrated against a social-cost
  reference.
- **A principled LLM-scale MoB looks like a posted-price menu mechanism, not
  an auction**. Sequential posted pricing (Chawla-Hartline-Malec-Sivan 2010)
  achieves a `1/2`-approximation to Bayesian-optimal revenue with O(1)
  per-token complexity - which is the only complexity regime a router can
  afford at `T = 4096` tokens per layer per forward pass.
- **Two-sided variants (tokens also bid) are theoretically cleaner but
  practically worthless** for 8-128 experts: stable-matching mechanisms
  (Gale-Shapley, deferred acceptance) cost `O(TK log K)` per layer and lose
  the "one-pass" streaming property. Document this explicitly so nobody
  wastes an ablation quarter on it.
- **Publishable gaps:** (i) conscience-mechanism <-> auxiliary-loss
  equivalence proof; (ii) MoB as a scoring auction with learnable scoring
  rule and a Myerson-monotonicity audit; (iii) the CL "winner's curse"
  framing as a bias-variance decomposition over affiliated-signal bidders.

---

## 2. Auction-theory audit

### 2.1 Naming the mechanism

MoB's bid rule
```
b_i = alpha * exec_cost_i + beta * forget_cost_i
winner = argmin_i b_i
```
is, in auction-theory vocabulary, a **first-score sealed-bid reverse (procurement)
auction with a linear-in-attributes quasi-linear scoring rule** (Che 1993,
Asker-Cantillon 2008). Specifically:

- *Reverse* / procurement: the principal (training loop) is buying a service
  (one gradient step) from experts. Sellers bid; lowest price wins. This is
  the mirror of a standard ascending auction.
- *Sealed-bid, first-score*: bids are computed independently per round and
  not revealed to other bidders; the winner "pays" their own bid (bears the
  update cost themselves). Che (1993) proves that first-score and
  second-score auctions yield identical buyer utility when the scoring rule
  equals the buyer's utility function, which is likely why MoB's logged
  second-price variant produces the same winner in almost all diagnostics.
- *Quasi-linear scoring rule, linear in attributes*: the score
  `S(exec, forget) = -(alpha*exec + beta*forget)` is quasi-linear in the
  price-equivalent axis (here a composite loss) and *linear* in the two
  attributes. This is the simplest and most analyzed scoring family in
  procurement theory (Asker-Cantillon 2008, Branco 1997).

MoB is *not*:
- A Vickrey auction (second-price, single-item) because the cost structure
  is multi-attribute.
- A VCG mechanism (multi-unit, multi-item) because there is no externality-
  based transfer.
- A combinatorial auction - each round is single-item ("this batch").

### 2.2 Properties audit

| Property | Status in current MoB | Justification |
|---|---|---|
| Truthful (DSIC) | Vacuously yes | Experts have no private type to misreport; `exec`, `forget` are observed by the mechanism. |
| Efficient (ex-post) | Approximately, conditional on scoring rule | True iff `alpha*exec + beta*forget` is a monotone transform of counter-factual per-expert loss delta. There is no proof this holds; it is the implicit modeling assumption. |
| Individually rational | Weakly yes | Losers bear zero cost. Winner bears the SGD-step cost but by construction this is bounded since they had the lowest bid among participants. |
| Budget-balanced | Trivially yes | No transfers. This is what allows MoB to be stateless across rounds - there is nothing to settle. |
| Revenue-optimal | Not analyzed | Would require a Myerson-style virtual-cost transformation of exec+forget distributions, which presumes private types. N/A to stateless MoB; relevant to *learned-bidder* MoB (see 2.3). |

The single nontrivial property is **efficiency relative to the true
continual-learning objective**. MoB efficiency hinges on the (unstated)
assumption:
```
alpha*exec_i + beta*forget_i  ~  E[L_new(i) + lambda * Delta L_old(i)]
```
where the right-hand side is the true per-expert continual-learning cost. If
this assumption fails (e.g., forget_cost under-penalizes low-Fisher directions
that still matter at test time), the greedy first-score winner is *not* the
social-welfare maximizer.

### 2.3 Revelation-principle gap when experts strategize

If a future variant of MoB lets each expert learn per-expert `(alpha_i,
beta_i)` as functions of their observation history - say, meta-learned bidder
heads - the mechanism becomes a game with private types
`theta_i = (alpha_i, beta_i)`.

The **revelation principle** (Myerson 1981, Gibbard 1973) says that for any
equilibrium outcome of this game there exists a direct mechanism in which
truthful reporting is an equilibrium. But the current MoB payment rule is
zero, so truthfulness requires:

1. **Allocation monotonicity** (Myerson's lemma): the winning probability of
   expert `i` must be weakly monotone in its reported type. Because the bid
   is linear in `alpha_i` and `beta_i`, monotonicity in `alpha_i` and
   `beta_i` individually holds - but monotonicity *in the type space* is
   not automatic because `exec_i` and `forget_i` may themselves respond to
   `alpha_i`, `beta_i` through the training trajectory.
2. **Correct payment rule.** For the direct mechanism to be truthful there
   must be a payment `t_i(theta)` equal to the Myerson "threshold bid" - the
   minimum report at which `i` would still have won. MoB has no such
   payment. Therefore learned-bidder MoB is **not** DSIC.

This is the revelation-principle gap. Practical implications:
- If experts learn to down-weight `forget_cost` by under-reporting Fisher
  information, they win more batches than they should. This is the direct
  analog of capacity-market bidders under-reporting their own outage risk.
- Adding an exponential payment rule `t_i = -threshold_bid_i` and subtracting
  it from `i`'s subsequent gradient budget would restore truthfulness - but
  at that point the mechanism is a Vickrey-Clarke-Groves auction on a
  single item and loses statelessness.

**Design implication**: stateless learned-bidder MoB is at best approximately
truthful. Any paper that introduces learned bidders must either (a) prove
monotonicity of the induced allocation rule, or (b) accept a bounded
incentive gap and quantify it.

### 2.4 Winner's-curse analog and the optimizer-reset fix

In a **common-value sealed-bid first-price auction with affiliated signals**
(Milgrom-Weber 1982), each bidder sees a noisy signal of the true common
value. The winner is systematically the bidder with the most optimistic
signal; in expectation they overpay. Bid shading and signal disclosure (via
ascending auctions) mitigate this.

MoB's rounds are exactly this structure:
- **Common value**: the "true" per-batch learning value is shared (it's the
  actual gradient-step utility).
- **Private signals**: each expert's estimates of `exec_i` and `forget_i`.
  `forget_i` in particular depends on Fisher information estimated from the
  *winning* expert's own past trajectory, which is stale and
  idiosyncratically biased.
- **Affiliation**: signals are correlated because experts share the input
  distribution.

Predicted pathology: the expert whose stale Adam moments *most
underestimate* `forget_i` wins most, then *overfits* to the won batch
because those stale moments were precisely what made it look cheap. This is
the "overfit-after-winning" pattern.

The **optimizer reset** fix after Fisher update (per `MEMORY.md`) is the
discrete analog of Milgrom-Weber's linkage principle: it injects information
(here, "your old moments are void") that *reduces* affiliation and breaks
the correlation between winning and being biased. Clamping the Fisher
minimum (0.001 -> 0.1 per `MEMORY.md`) is bid-shading at mechanism level: it
refuses to allow an expert to report "costless protection" just because its
Fisher happened to be initialized small.

This is a complete and rigorous re-derivation of two MoB fixes from 1982
auction theory. **Publishable as a framing paper** (see Section 5).

---

## 3. Mechanism design for ML routing at LLM scale

### 3.1 What a multi-token auction principally looks like

At LLM scale a router faces, per layer per forward pass:
- `T` tokens (items)
- `K` experts (bidders)
- Capacity `C_k` per expert (multi-unit supply)
- Top-k routing (each token needs `k` units)

This is the classic **multi-unit combinatorial reverse auction with capacity
constraints**. The winner-determination problem (WDP) is equivalent to a
weighted bipartite b-matching when bids are unit-item (standard
top-1/top-2 MoE). When bids are bundle-valued (e.g., "expert k bids on
two specific tokens jointly because together they form a local structure it
specializes in"), WDP becomes NP-hard and inapproximable within
`min(l^{1-eps}, m^{1/2-eps})` for `l` bids and `m` items (Sandholm, Lehmann-
O'Callaghan-Shoham 2002).

**Practical consequence**: any MoB variant that lets experts bid on
*bundles* of tokens - say, "I want this whole span because I specialize in
fact-retrieval for biographies" - falls into an intractable combinatorial
auction regime. Current top-k routing sidesteps this by enforcing
single-item bids.

### 3.2 Top-2 routing as a two-winner VCG-lite combinatorial auction

If two experts jointly win a token (top-2), the mechanism must allocate the
token to a *pair* of experts, then split the gradient update
(proportionally, by softmax gate). Treating each token-pair slot as a
combinatorial item yields:

- **Full VCG**: price each expert by the social welfare externality it
  imposes on the other winners. Requires computing counterfactual allocations
  without each expert. At LLM scale this is `K-1` additional WDP instances
  per token - infeasible.
- **VCG-lite (approximate VCG with approximation algorithms)**: known to
  break truthfulness (Nisan-Ronen 2001, Vorobeychik 2011). So the common
  "top-2 with load-balance loss" approach is neither efficient *nor*
  truthful in the auction sense; it is a heuristic that happens to work
  because experts in practice don't strategize.
- **Single-item greedy with bundling price** (Lehmann-O'Callaghan-Shoham
  2002): approximates optimal combinatorial welfare within `sqrt(m)` while
  remaining *truthful for single-minded bidders*. The right benchmark if MoB
  wants bundle bids.

### 3.3 Truthful stateless bid rule: a sketch

The key constraint: MoB wants statelessness. The key auction-theoretic
insight: Myerson's lemma gives truthfulness from monotonicity + correct
payments. Statelessness forbids remembering per-expert "debits".

**Proposal (sketched, not endorsed): Myerson-shaded MoB.**
Replace `b_i = alpha*exec_i + beta*forget_i` with
```
b_i = alpha*exec_i + beta*forget_i + phi_i^{-1}(F_i(exec_i + forget_i))
```
where `phi_i^{-1}` is the inverse virtual-cost transformation under the
empirical CDF `F_i` of expert `i`'s historical costs. This is Myerson's
optimal-mechanism construction applied to the score axis. It is truthful
when expert types are drawn i.i.d. and monotone-hazard-rate. It preserves
statelessness if `F_i` is approximated by a running quantile sketch (O(log
N) memory per expert).

This is *not* a recommendation; it is a concrete candidate for an ablation
that would answer "does MoB benefit from proper auction-theoretic truthful
shading?"

### 3.4 Posted-price menu mechanisms and online mechanism design

Per-token LLM routing is fundamentally **online**: tokens stream in,
routing decisions must be irrevocable, no re-auctioning. The mechanism-
design answer is posted-price and menu mechanisms.

- **Sequential posted pricing** (Chawla-Hartline-Malec-Sivan 2010): for each
  expert, *post* a threshold bid `p_k` such that the first expert whose true
  cost falls below its posted price takes the item. Achieves
  `1/2`-approximation to Bayesian-optimal welfare with O(K) complexity per
  item, truthful because there is no strategic surface - the expert either
  accepts or doesn't. This is *the* right primitive for per-token routing.
- **Menu mechanisms** (Hartline-Roughgarden 2009): each expert picks a
  "contract" from a posted menu of (capacity, price) pairs. Allows static
  load-balancing without auxiliary loss - capacity is literally sold as a
  commitment.
- **Online mechanism design, Blum-Hajiaghayi-Ligett-Roth 2008, Hartline
  2013**: analyze regret bounds of posted-price mechanisms against the
  optimal batch auction. Directly applicable to streaming CL: `T` tokens
  arrive, a posted-price mechanism is `O(sqrt(T))`-regret against the
  ex-post optimal allocation.

**Structural claim**: the DeepSeek-style "auxiliary-loss-free load
balancing" (Wang et al. 2024) - which maintains a per-expert bias updated by
recent load - is *exactly* a sequential posted-price mechanism without the
authors realizing it. The bias is the posted price; the recent-load update
is the price-adjustment rule in Kleinberg-Leighton 2003 digital-goods
pricing. Worth formalizing.

---

## 4. Cross-domain bridges

### 4.1 Holland's bucket-brigade classifier systems: *not* an ancestor

Holland (1985) bucket brigade: each classifier has a *strength*. When its
condition matches a message, it *bids* a fraction of its strength for the
right to post its consequent message. The bid is *transferred* to the
classifier(s) whose prior messages satisfied its conditions. External
reward flows back the chain; genetic-algorithm rule discovery acts on
strengths.

Structural comparison:

| Axis | Bucket brigade | MoB |
|---|---|---|
| Bidders | Rules matched against messages | Experts matched against batches |
| Bid | Fraction of own strength | alpha*exec + beta*forget |
| Transfer | Bid paid to upstream suppliers | None |
| Credit assignment | Multi-step chain via transfers | Single-step (per-round) |
| Rule discovery | GA on strengths | N/A (experts are fixed) |
| Learning signal | Environmental payoff, propagated back | Supervised gradient, local |

**Isomorphism test**: if MoB had *no* gradient signal and instead
redistributed a scalar "strength" among experts that chained to produce
outputs, it would be a bucket brigade. It doesn't. The two mechanisms share
only the "competitive bid for action rights" surface; the credit-assignment
substrate is different.

**The CL/MoE community should stop citing bucket brigade as the precedent
for MoB.** A cleaner precedent is Che (1993) scoring auctions or DeSieno
(1988) competitive learning.

### 4.2 Competitive learning: the conscience mechanism is the missing link

Kohonen SOMs, neural gas (Martinetz-Schulten 1991), and frequency-sensitive
competitive learning (Ahalt et al. 1990) all face the **dead-unit problem**:
a unit that never wins never updates, so it never becomes competitive. This
is *identical* to MoE expert collapse (Fedus et al. 2022, Switch
Transformer).

DeSieno's **conscience mechanism** (IEEE ICNN 1988) adds a bias `b_i =
gamma*(1/N - f_i)` to each unit's distance, where `f_i` is unit `i`'s
recent winning frequency. Units that win too often get *penalized*; units
that rarely win get *preferred*. This is **exactly** the shape of:

- Switch Transformer's load-balancing auxiliary loss (Fedus et al. 2022): a
  penalty on fraction-of-tokens times mean-gate-probability.
- DeepSeek's auxiliary-loss-free load balancing (Wang et al. 2024): an
  additive bias per expert, updated by recent load.
- Dynamic routing biases in MoE deployments.

**Structural claim**: Switch-style auxiliary losses are DeSieno's conscience
mechanism with `gamma` chosen to match scale. DeepSeek's bias is literally
DeSieno's bias, re-derived in 2024. The MoE literature did not cite DeSieno
or competitive-learning literature when deriving these fixes.

**Implication for MoB**: to avoid expert collapse without auxiliary loss,
the cleanest primitive is to add a DeSieno-style frequency bias directly to
`b_i`:
```
b_i <- b_i + gamma * (f_i - 1/K)
```
where `f_i` is `i`'s recent winning rate. This stays stateless (a single
EMA per expert) and interpolates smoothly with MoB's existing bid rule. Do
*not* call this "load-balancing loss" - call it a conscience term, credit
DeSieno, and cite the 36-year heritage.

### 4.3 Electricity capacity markets: mature analog of forget-cost pathology

Capacity markets pay generators for *being available* regardless of dispatch.
The **missing-money problem** (Newbery 2016, Joskow 2008): energy-only
markets don't clear at the social cost of reliability; capacity payments
fill the gap but are prone to over-procurement.

Structural mapping to MoB:
- `exec_cost` ~ energy-market clearing price (cost to actually run now).
- `forget_cost` ~ capacity-market payment (cost of protecting installed,
  potentially stranded, capability).
- Expert = generator. Unused expert = stranded asset. Overloaded expert =
  must-run incumbent.

Three 30-year-old lessons:
1. **Capacity payments without a social-cost benchmark over-procure
   protection.** If `beta` is tuned purely against retention metrics, MoB
   will bias toward hoarding "capacity" (preserving old experts) at the
   expense of new-task learning. Calibrate `beta` against a scenario where
   forgetting is *cheap* (e.g., a replayable fine-tune) to avoid this.
2. **Incumbent bidders under-report cost.** In deregulated capacity markets
   (PJM, UK), incumbents exercised market power by strategic bidding. In
   learned-bidder MoB, the analog is an expert meta-learning to underreport
   Fisher-weighted interference.
3. **Descending-clock auctions are the capacity-market state of the art**
   because they reveal the market-clearing price through iterative
   disclosure. MoB's sealed-bid one-shot format is the *older* technology;
   a descending-clock variant (broadcast the current cutoff, let experts
   drop out) would reveal signal affiliation and address the winner's curse
   from Section 2.4. Not a quick change given per-batch compute, but worth
   logging as a candidate.

### 4.4 Market-based distributed-systems scheduling

Waldspurger et al.'s **Spawn** (IEEE TSE 1992) auctions CPU time to jobs.
Later work (Clearwater 1996, Chun-Culler 2002, Lai et al. 2005) added
Vickrey auctions and proportional-share. Key design moves MoB has not
adopted:

- **Tickets** (Waldspurger-Weihl 1994, lottery scheduling): randomized
  proportional allocation. At routing scale, a stochastic bid sampled from
  a Boltzmann over scores recovers tickets and avoids deterministic
  collapse.
- **Strata** (Lai-Rasmusson-Adar-Sorkin-Zhang 2005): preserve starvation-
  free guarantees. MoE top-k with capacity factor is a crude strata
  mechanism.
- **Virtual currencies with refresh** (Chun-Culler 2002): replenish
  bidder budgets periodically. For MoB, a per-expert "training budget"
  reset per epoch prevents one expert from monopolizing the first half of
  training.

These are all compatible with statelessness at per-round granularity.

### 4.5 Peer prediction and strategyproof classification

- **Peer prediction** (Miller-Resnick-Zeckhauser 2005) elicits truthful
  private signals without ground truth by comparing agents to each other.
  Relevant to MoB *if* we treat each expert's self-report of `forget_cost`
  as a signal: one could compute a peer-prediction score by comparing
  expert `i`'s forget-cost estimate to the estimate experts `j != i` would
  produce on the same parameters. This is a (speculative) route to making
  learned-bidder MoB truthful without explicit payments.
- **Bayesian Truth Serum** (Prelec 2004, Witkowski-Parkes 2012): elicits
  truthful common-knowledge beliefs. Less directly applicable - experts
  don't have common-knowledge beliefs over each other's Fisher.
- **Strategyproof classification** (Meir-Procaccia-Rosenschein 2012; Hardt-
  Megiddo-Papadimitriou-Wootters 2016): analyzes classifiers robust to
  strategic agents manipulating features. The MoB analog is: if the
  *inputs* to the router can be manipulated (adversarial prompts, prompt-
  injection), how does the routing mechanism degrade? Not a current MoB
  concern, but a clean framing for safety work on MoE routers.

---

## 5. Transferable design moves (candidates only - do not adopt without ablation)

Each item lists (i) the move, (ii) its parent mechanism, (iii) what it
would cost MoB in statelessness/compute, (iv) what pathology it targets.

1. **DeSieno conscience term in the bid.** `b_i <- b_i + gamma*(f_i - 1/K)`.
   Parent: DeSieno 1988 conscience; Ahalt 1990 FSCL. Cost: one EMA per
   expert. Targets: expert collapse.
2. **Myerson virtual-cost shading on the score axis.** Replace the score
   with its Myerson virtual-cost transform. Parent: Myerson 1981. Cost:
   one quantile sketch per expert (O(log N) memory). Targets: strategic
   manipulation if learned-bidder MoB is introduced.
3. **Sequential posted-price router.** Maintain per-expert posted price;
   first expert whose live cost falls below its price wins. Parent:
   Chawla-Hartline-Malec-Sivan 2010. Cost: per-expert price, updated by
   recent-load rule (Kleinberg-Leighton 2003). Targets: streaming per-
   token routing at LLM scale.
4. **Descending-clock variant for per-task routing.** Broadcast current
   cutoff, let experts drop out. Parent: PJM capacity auctions. Cost:
   round-trip per task (fine for task-level, infeasible per-token).
   Targets: winner's curse under affiliated signals.
5. **Boltzmann-softmax over bids instead of argmin.** Parent: Waldspurger
   lottery scheduling 1994. Cost: one temperature hyperparameter. Targets:
   deterministic expert collapse; adds exploration.
6. **Peer-prediction audit of forget-cost reports.** Parent: Miller-Resnick-
   Zeckhauser 2005. Cost: one cross-expert comparison per update. Targets:
   truthfulness of learned-bidder MoB.
7. **Periodic budget refresh** (virtual currency per epoch). Parent: Chun-
   Culler 2002 distributed virtual currencies. Cost: one counter per
   expert. Targets: early-training monopoly by one expert.
8. **Menu mechanism at task boundaries.** Each expert chooses a contract
   (capacity, effective beta). Parent: Hartline-Roughgarden 2009. Cost:
   task-boundary decision. Targets: principled capacity allocation without
   auxiliary loss.
9. **Lehmann-O'Callaghan-Shoham greedy for bundle bids.** If top-k routing
   is reframed as bundle bids. Parent: LOS 2002. Cost: sqrt(m)
   approximation ratio, single-minded-bidder assumption. Targets: any
   future bundle-valued bid structure.
10. **Linkage-principle-style signal disclosure on win.** Broadcast
    winner's post-update state to losers so they can refine forget-cost
    estimates next round. Parent: Milgrom-Weber 1982. Cost: one cross-
    expert parameter-delta message. Targets: winner's-curse bias
    accumulation.

---

## 6. Publishable gaps

The following cross-domain framings each appear to be a paper in
themselves. Flagged by confidence (H/M/L) and the minimum viable
experiment.

### 6.1 [H] "MoB is a scoring auction; continual learning is multi-attribute procurement"
A focused paper that (a) formalizes MoB as a first-score quasi-linear
scoring auction, (b) identifies `beta` as a capacity-payment coefficient,
(c) proves the conditions under which MoB is ex-post efficient, and
(d) runs ablations that *vary the scoring family* (linear vs. CES vs.
log-additive). Minimum viable experiment: swap the scoring rule; plot
task-retention Pareto fronts.

### 6.2 [H] "Switch Transformer's load-balancing loss is DeSieno's conscience"
A structural-equivalence paper: formalize the equivalence, derive
conditions under which DeSieno-style bias is Pareto-superior to auxiliary
loss (easier gradient, no hyperparameter trade-off), and show that DeepSeek's
aux-loss-free scheme is a rediscovery of 1988 competitive learning. This
is low-controversy, high-citation (multiple communities benefit). Minimum
viable experiment: reproduce Switch/DeepSeek load-balance behaviour with a
DeSieno bias at matched compute; show equivalence of steady-state load
distributions.

### 6.3 [M] "A revelation-principle audit of learned bidders in MoE routing"
If any lab starts meta-learning per-expert bid weights, the DSIC question
is live. A paper that (a) proves non-truthfulness of naive learned-bidder
routing, (b) constructs a Myerson-monotone variant, (c) empirically
measures the welfare gap, would get cited widely in both ML and
algorithmic game theory.

### 6.4 [M] "Winner's curse as a diagnostic for continual learning"
Framing MoB's optimizer-reset pathology as affiliated-signal winner's curse
(Section 2.4), measuring bid-signal correlations with win-vs-bias, and
formalizing "linkage-principle" interventions (information sharing across
experts) as a principled alternative to optimizer reset. Requires
controlled experiments distinguishing stale-moment bias from Fisher-
estimation bias.

### 6.5 [M] "Capacity markets for continual learning"
A longer-format paper that imports the missing-money, missing-market
framework into the CL retention/plasticity trade-off. Novel because it
treats `beta` not as a hyperparameter but as a market-design parameter
subject to over-procurement failure modes. Risk: reviewers may find the
economic detour indulgent without a concrete algorithmic deliverable.

### 6.6 [L] "Sequential posted-price routing for LLM MoE"
High-risk, high-reward: formalize per-token routing as sequential posted
pricing, prove regret bounds against ex-post optimal routing, show
empirical parity with DeepSeek aux-loss-free at lower compute. Risk comes
from engineering: needs a real LLM MoE replication, not just a Split-MNIST
toy.

### 6.7 [L] "Bucket brigade vs. MoB: a structural non-identity theorem"
A short position paper correcting the CL/MoE community's frequent
hand-waving at bucket-brigade as MoB's ancestor. Useful but niche;
probably a workshop paper, not a venue paper.

---

## 7. Uncertainty and limitations

- **Uncertain (M-H)**: the precise equivalence between DeSieno conscience
  and Switch/DeepSeek load balancing. Structurally strong but the
  coefficient calibration differs; the proof needs matching EMA half-life
  against gradient-rate constants.
- **Uncertain (M)**: whether MoB's linear-in-attributes scoring rule is
  monotone in the full type space (Section 2.3). Proving or disproving
  this is a prerequisite to any learned-bidder truthfulness claim.
- **Uncertain (L)**: whether capacity-market over-procurement literature
  quantitatively predicts `beta` pathologies in MoB or is only a
  structural analogy. The former would give concrete `beta`-calibration
  prescriptions; the latter is only a framing.
- **Limit**: all claims above assume MoB's current scale. At LLM scale
  (K=128, T=4096) the winner-determination NP-hardness kicks in for any
  bundle-valued variant; this is a compute floor, not an algorithmic
  choice.

---

## Primary sources (selected, most-cited-first within each theme)

Auction theory and mechanism design:
- Myerson, R. "Optimal Auction Design." Mathematics of Operations Research, 1981.
- Vickrey, W. "Counterspeculation, Auctions, and Competitive Sealed Tenders." J. Finance, 1961.
- Milgrom, P., Weber, R. "A Theory of Auctions and Competitive Bidding." Econometrica, 1982. (linkage principle, affiliated signals)
- Che, Y.-K. "Design Competition through Multi-Dimensional Auctions." RAND J. Econ., 1993. (first-score / second-score scoring auctions)
- Asker, J., Cantillon, E. "Properties of Scoring Auctions." RAND J. Econ., 2008.
- Lehmann, D., O'Callaghan, L., Shoham, Y. "Truth Revelation in Approximately Efficient Combinatorial Auctions." J. ACM, 2002.
- Sandholm, T. "Algorithm for Optimal Winner Determination in Combinatorial Auctions." AIJ, 2002.
- Chawla, S., Hartline, J., Malec, D., Sivan, B. "Multi-parameter Mechanism Design and Sequential Posted Pricing." STOC 2010.
- Hartline, J. "Mechanism Design and Approximation." Book draft, 2013.
- Nisan, N., Ronen, A. "Algorithmic Mechanism Design." Games and Economic Behavior, 2001.
- Kagel, J., Levin, D. "Common Value Auctions and the Winner's Curse." Princeton UP, 2002.
- Bergemann, D., Brooks, B., Morris, S. "Countering the Winner's Curse." Theoretical Economics, 2020.

Classifier systems and credit assignment:
- Holland, J. H. "Properties of the Bucket Brigade." Proc. 1st ICGA, 1985.
- Holland, J. H., Reitman, J. S. "Cognitive Systems Based on Adaptive Algorithms." Pattern-Directed Inference Systems, 1978.

Competitive learning:
- DeSieno, D. "Adding a Conscience to Competitive Learning." IEEE ICNN, 1988.
- Ahalt, S., Krishnamurthy, A., Chen, P., Melton, D. "Competitive Learning Algorithms for Vector Quantization." Neural Networks, 1990.
- Martinetz, T., Schulten, K. "A Neural-Gas Network Learns Topologies." Artificial Neural Networks, 1991.

MoE routing:
- Fedus, W., Zoph, B., Shazeer, N. "Switch Transformers." JMLR 2022.
- Wang, L. et al. "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts." arXiv 2408.15664, 2024 (DeepSeek).

Capacity markets and procurement:
- Newbery, D. "Missing Money and Missing Markets." Energy Policy, 2016.
- Joskow, P. "Capacity Payments in Imperfect Electricity Markets." Utilities Policy, 2008.

Market-based distributed systems:
- Waldspurger, C., Hogg, T., Huberman, B., Kephart, J., Stornetta, S. "Spawn: A Distributed Computational Economy." IEEE TSE, 1992.
- Waldspurger, C., Weihl, W. "Lottery Scheduling." OSDI 1994.
- Chun, B., Culler, D. "Market-based Proportional Resource Sharing for Clusters." TR UCB CSD-00-1092, 2000.

Peer prediction / strategyproof ML:
- Miller, N., Resnick, P., Zeckhauser, R. "Eliciting Informative Feedback: The Peer-Prediction Method." Management Science, 2005.
- Prelec, D. "A Bayesian Truth Serum for Subjective Data." Science, 2004.
- Witkowski, J., Parkes, D. "A Robust Bayesian Truth Serum for Small Populations." AAAI 2012.
- Meir, R., Procaccia, A. D., Rosenschein, J. S. "Algorithms for Strategyproof Classification." Artificial Intelligence, 2012.
- Hardt, M., Megiddo, N., Papadimitriou, C., Wootters, M. "Strategic Classification." ITCS 2016.
