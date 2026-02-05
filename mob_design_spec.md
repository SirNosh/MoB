# MoB: Mixture of Bidders - Design Specification

## Document Purpose

This specification defines what MoB (Mixture of Bidders) is intended to be, its theoretical foundations, architectural requirements, and success criteria. This serves as the reference for implementation and evaluation.

---

## 1. Executive Summary

**MoB** is a continual learning framework that replaces learned gating in Mixture-of-Experts architectures with a VCG (Vickrey-Clarke-Groves) auction mechanism. The key insight is that auction-based routing is **stateless** and therefore immune to catastrophic forgetting at the routing level.

### Core Innovation
Traditional MoE systems use a learned gater network to route inputs to experts. When trained on sequential tasks, this gater forgets how to route previous tasks - even if the experts themselves retain their knowledge. MoB eliminates this "gater forgetting" by using an economic mechanism that requires no learning and no state.

### One-Sentence Summary
> MoB uses game-theoretic auction mechanisms to achieve forgetting-free expert routing in continual learning.

---

## 2. Problem Statement

### 2.1 The Continual Learning Challenge

**Setting:** Data arrives as a sequence of tasks T = {T1, T2, ..., TK}. Each task Tk has a dataset Dk = {(xi, yi)}. The model must:
1. Learn each new task effectively
2. Retain performance on all previous tasks (no catastrophic forgetting)
3. Operate without explicit task labels at test time

**Metric:** Average accuracy across all tasks after training completes:
```
ACC = (1/K) * Σk ACC(Tk)
```

**Forgetting Metric:**
```
Forgetting = (1/(K-1)) * Σk max(0, ACC_after_task_k(Tk) - ACC_final(Tk))
```

### 2.2 The Gater Forgetting Problem

In standard Mixture-of-Experts:
```
output = Σi g(x)_i * E_i(x)
```

Where g(x) is a learned gating network. The problem:

1. Train gater on Task 1: g learns to route digits 0-1 to Expert A
2. Train gater on Task 2: g learns to route digits 2-3 to Expert B
3. **But updating g on Task 2 corrupts the Task 1 routing logic**
4. Even if Expert A perfectly remembers digits 0-1, g no longer sends them there

**This is gater forgetting** - catastrophic forgetting at the routing level, separate from expert-level forgetting.

### 2.3 Why Existing Solutions Fail

| Approach | Why It Fails for Routing |
|----------|-------------------------|
| EWC on gater | Still learns, still forgets partially |
| Freeze gater | Can't learn to route new tasks |
| Replay for gater | Requires storing routing examples |
| Task-specific gaters | Requires task labels at test time |

**MoB's Solution:** Don't learn routing at all. Use a stateless economic mechanism.

---

## 3. Theoretical Framework

### 3.1 The VCG Mechanism

The Vickrey-Clarke-Groves mechanism is a auction design that achieves **truthfulness** - each participant's dominant strategy is to bid their true cost, regardless of what others do.

**Properties:**
- **Dominant Strategy Incentive Compatible (DSIC):** Truthful bidding is optimal
- **Allocatively Efficient:** Selects the socially optimal outcome
- **Individually Rational:** No participant is worse off by participating

**For MoB:**
- Each expert bids its "cost" to process a batch
- The expert with the lowest bid wins
- Winner pays the second-lowest bid (Vickrey/second-price rule)

### 3.2 The Bidding Function

Each expert i computes a bid for batch (x, y):

```
bid_i(x, y) = α * execution_cost_i(x, y) + β * forgetting_cost_i(x, y)
```

Where:
- **α** = weight for execution cost (how well can I handle this batch?)
- **β** = weight for forgetting cost (how much would training damage my knowledge?)
- **α + β need not equal 1** (they're scaling factors, not probabilities)

### 3.3 Execution Cost (α signal)

**Definition:** The predicted loss if this expert processes the batch.

```python
def execution_cost(expert, x, y):
    logits = expert.forward(x)
    return cross_entropy(logits, y)
```

**Interpretation:**
- Low loss → Expert is already good at this data → Low execution cost → Competitive bid
- High loss → Expert would struggle → High execution cost → Uncompetitive bid

**Key Property:** This creates natural specialization. Experts that have learned a task will bid low for similar data.

### 3.4 Forgetting Cost (β signal)

**Definition:** The estimated damage to previously learned knowledge if this expert trains on the batch.

**Theoretical formulation (gradient interference):**
```
forgetting_cost(x, y) = Σ_params F_i * (∂L/∂θ_i)²
```

Where:
- F_i = Fisher Information for parameter i (importance for past tasks)
- ∂L/∂θ_i = gradient from current batch

**Interpretation:**
- If the gradient from (x, y) would modify important parameters, forgetting cost is high
- If the gradient is orthogonal to important parameters, forgetting cost is low

**Key Property:** This protects experts from being assigned data that would damage their knowledge, even if they could technically learn it.

### 3.5 Why VCG Ensures Truthfulness

**Theorem (Vickrey):** In a second-price auction, truthful bidding is a dominant strategy.

**Proof sketch:**
- If you bid lower than your true cost and win, you might pay more than you're worth
- If you bid higher than your true cost, you might lose when you should have won
- Bidding exactly your true cost maximizes expected utility regardless of others' bids

**For MoB:** Each expert's "true cost" is α*exec + β*forget. The VCG mechanism ensures experts have no incentive to misrepresent this cost.

---

## 4. Architecture Specification

### 4.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        MoB System                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │  Expert 1   │   │  Expert 2   │   │  Expert N   │        │
│  │  ┌───────┐  │   │  ┌───────┐  │   │  ┌───────┐  │        │
│  │  │ Model │  │   │  │ Model │  │   │  │ Model │  │        │
│  │  └───────┘  │   │  └───────┘  │   │  └───────┘  │        │
│  │  ┌───────┐  │   │  ┌───────┐  │   │  ┌───────┐  │        │
│  │  │ EWC   │  │   │  │ EWC   │  │   │  │ EWC   │  │        │
│  │  │Engine │  │   │  │Engine │  │   │  │Engine │  │        │
│  │  └───────┘  │   │  └───────┘  │   │  └───────┘  │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Bid Collection Layer                    │    │
│  │   bid_1 = α*exec_1 + β*forget_1                     │    │
│  │   bid_2 = α*exec_2 + β*forget_2                     │    │
│  │   ...                                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              VCG Auction Mechanism                   │    │
│  │   winner = argmin(bids)                             │    │
│  │   payment = second_lowest_bid                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│                    Winner trains on batch                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Expert Component

Each expert contains:

```python
class MoBExpert:
    model: nn.Module           # The neural network
    ewc_engine: EWCEngine      # Fisher information + optimal params
    expert_id: int             # Unique identifier
    
    def compute_bid(self, x, y) -> float:
        """Compute truthful bid for batch (x, y)"""
        exec_cost = self.compute_execution_cost(x, y)
        forget_cost = self.compute_forgetting_cost(x, y)
        return self.alpha * exec_cost + self.beta * forget_cost
    
    def train_on_batch(self, x, y, optimizer):
        """Train with EWC regularization"""
        loss = cross_entropy(self.model(x), y) + self.ewc_engine.penalty()
        loss.backward()
        optimizer.step()
    
    def update_ewc(self, dataloader):
        """Update Fisher information after task completion"""
        self.ewc_engine.compute_fisher(dataloader)
```

### 4.3 EWC Engine

```python
class EWCEngine:
    fisher: Dict[str, Tensor]        # Fisher information per parameter
    optimal_params: Dict[str, Tensor] # Parameters at task completion
    lambda_ewc: float                 # Regularization strength
    
    def compute_fisher(self, dataloader, num_samples=200):
        """Compute diagonal Fisher Information Matrix"""
        # F_i = E[(∂log p(y|x)/∂θ_i)²]
        
    def penalty(self) -> Tensor:
        """EWC regularization term"""
        # L_ewc = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²
        
    def compute_forgetting_cost(self, x, y) -> float:
        """Estimate forgetting if trained on (x, y)"""
        # Cost = Σ_i F_i * (∂L/∂θ_i)²
```

### 4.4 VCG Auction

```python
class VCGAuction:
    def run_auction(self, bids: np.ndarray) -> Tuple[int, float]:
        """
        Execute VCG auction
        
        Returns:
            winner: Index of winning expert (lowest bid)
            payment: Second-lowest bid (VCG payment)
        """
        winner = np.argmin(bids)
        payment = np.partition(bids, 1)[1]  # Second lowest
        return winner, payment
```

---

## 5. Algorithm Specification

### 5.1 Training Algorithm

```
Algorithm: MoB Training

Input: Task sequence T = {T1, ..., TK}, Expert pool E = {E1, ..., EN}
Output: Trained experts

for each task Tk in T:
    for each epoch:
        for each batch (x, y) in Tk:
            # Step 1: Collect bids
            bids = []
            for each expert Ei in E:
                bid_i = α * Ei.execution_cost(x, y) + β * Ei.forgetting_cost(x, y)
                bids.append(bid_i)
            
            # Step 2: Run VCG auction
            winner = argmin(bids)
            payment = second_min(bids)
            
            # Step 3: Train winner
            E[winner].train_on_batch(x, y)
    
    # Step 4: Update EWC for experts that trained on this task
    for each expert Ei that won batches in Tk:
        Ei.update_ewc(Tk.dataloader)

return E
```

### 5.2 Inference Algorithm (CRITICAL - Must Not Use Labels)

```
Algorithm: MoB Inference

Input: Test batch x, Expert pool E
Output: Predictions

# Option A: Confidence-based routing (RECOMMENDED)
confidences = []
all_logits = []
for each expert Ei in E:
    logits = Ei.forward(x)
    confidence = softmax(logits).max(dim=-1).mean()
    confidences.append(confidence)
    all_logits.append(logits)

winner = argmax(confidences)
predictions = all_logits[winner].argmax(dim=-1)

# Option B: Ensemble averaging
all_logits = [Ei.forward(x) for Ei in E]
avg_logits = mean(all_logits)
predictions = avg_logits.argmax(dim=-1)

# FORBIDDEN: Using true labels for routing
# winner = argmin([cross_entropy(Ei(x), y) for Ei in E])  # THIS IS CHEATING

return predictions
```

---

## 6. Correctness Requirements

### 6.1 Bidding Requirements

| Requirement | Description | Validation |
|-------------|-------------|------------|
| **BID-1** | Execution cost must be computed using the batch (x, y) | exec_cost depends on x |
| **BID-2** | Forgetting cost must be input-dependent | forget_cost depends on x, y |
| **BID-3** | Forgetting cost = 0 when no Fisher exists | First task has no forgetting |
| **BID-4** | Bids must be non-negative | bid >= 0 always |

### 6.2 EWC Requirements

| Requirement | Description | Validation |
|-------------|-------------|------------|
| **EWC-1** | Fisher computed from model's own predictions | Use sampled_y or true y |
| **EWC-2** | Optimal params stored at task completion | Don't overwrite previous |
| **EWC-3** | Fisher accumulated across tasks | F_total = F_1 + F_2 + ... |
| **EWC-4** | Penalty uses correct reference point | (θ - θ*) for correct θ* |

### 6.3 Evaluation Requirements

| Requirement | Description | Validation |
|-------------|-------------|------------|
| **EVAL-1** | **No true labels used for routing at test time** | Critical for validity |
| **EVAL-2** | Same evaluation protocol for all methods | Fair comparison |
| **EVAL-3** | Multiple seeds for statistical significance | n >= 5 seeds |
| **EVAL-4** | Report mean ± std | Include variance |

### 6.4 Auction Requirements

| Requirement | Description | Validation |
|-------------|-------------|------------|
| **AUC-1** | Winner is minimum bid | winner = argmin(bids) |
| **AUC-2** | Payment is second-lowest bid | VCG mechanism |
| **AUC-3** | Auction is stateless | No learning in auction |

---

## 7. Expected Behavior

### 7.1 Natural Expert Specialization

Over training, experts should naturally specialize:

```
Task 1 (digits 0-1): Expert A wins most batches
  → Expert A becomes good at 0-1
  → Expert A bids low for 0-1 data
  
Task 2 (digits 2-3): Expert B wins most batches
  → Expert B becomes good at 2-3
  → Expert B bids low for 2-3 data
  → Expert A bids HIGH for 2-3 (high forgetting cost)
```

### 7.2 Forgetting Protection

When Task 2 data arrives:
- Expert A (Task 1 specialist) should have HIGH forgetting cost for Task 2 data
- This makes Expert A uncompetitive for Task 2 batches
- Expert A is protected from training on data that would damage its knowledge

### 7.3 Bid Component Balance

The α and β parameters control the tradeoff:
- **High α, Low β:** Prioritize immediate performance (risk forgetting)
- **Low α, High β:** Prioritize knowledge protection (risk poor new task learning)
- **Balanced:** Optimal continual learning

**Expected bid diagnostics:**
```
Forgetting/Execution ratio should be meaningful (0.1x - 1.0x)
NOT negligible (0.001x) - indicates β signal is too weak
```

---

## 8. Success Criteria

### 8.1 Quantitative Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Average Accuracy | > Monolithic EWC | MoB should beat single-model CL |
| Average Accuracy | > Gated MoE | MoB should beat learned gating |
| Forgetting | < 10% | Minimal catastrophic forgetting |
| Statistical Significance | p < 0.05 | Results must be significant |

### 8.2 Qualitative Criteria

| Criterion | Validation |
|-----------|------------|
| Expert specialization emerges | Different experts win different tasks |
| EWC penalty activates | ewc_penalty > 0 for experts on new tasks |
| Forgetting cost influences bids | β signal is non-negligible |
| No label leakage at test time | Evaluation uses confidence/ensemble only |

### 8.3 Baseline Ordering

Expected performance ranking:
```
MoB > Monolithic EWC > Random Assignment ≥ Gated MoE >> Naive
```

**Rationale:**
- MoB: Best (auction + EWC + multi-expert)
- Monolithic EWC: Good (EWC but single model, limited capacity)
- Random Assignment: Decent (multi-expert + EWC, but random routing)
- Gated MoE: Poor (gater forgetting destroys routing)
- Naive: Worst (complete forgetting)

---

## 9. Experimental Protocol

### 9.1 Dataset: Split-MNIST (Proof of Concept)

| Task | Classes | Train Samples | Test Samples |
|------|---------|---------------|--------------|
| 1 | 0, 1 | ~12,000 | ~2,000 |
| 2 | 2, 3 | ~12,000 | ~2,000 |
| 3 | 4, 5 | ~12,000 | ~2,000 |
| 4 | 6, 7 | ~12,000 | ~2,000 |
| 5 | 8, 9 | ~12,000 | ~2,000 |

### 9.2 Hyperparameters

```python
config = {
    'num_experts': 4,           # Fewer than tasks to force sharing
    'num_tasks': 5,
    'alpha': 0.5,               # Execution cost weight
    'beta': 0.5,                # Forgetting cost weight
    'lambda_ewc': 5000,         # EWC regularization strength
    'learning_rate': 0.001,
    'epochs_per_task': 4,
    'batch_size': 32,
}
```

### 9.3 Evaluation Protocol

```python
def evaluate_mob(expert_pool, test_loader):
    """
    CORRECT: Confidence-based routing (no labels)
    """
    all_preds = []
    all_labels = []
    
    for x, y in test_loader:
        # Compute confidence for each expert
        confidences = []
        logits_list = []
        for expert in expert_pool:
            logits = expert.model(x)
            conf = softmax(logits).max(dim=-1).values.mean()
            confidences.append(conf)
            logits_list.append(logits)
        
        # Route to most confident expert
        winner = np.argmax(confidences)
        preds = logits_list[winner].argmax(dim=-1)
        
        all_preds.append(preds)
        all_labels.append(y)
    
    accuracy = (cat(all_preds) == cat(all_labels)).float().mean()
    return accuracy
```

### 9.4 Baselines

| Baseline | Purpose |
|----------|---------|
| Naive Fine-tuning | Lower bound (maximum forgetting) |
| Random Assignment | Isolates auction contribution |
| Monolithic EWC | Isolates multi-expert contribution |
| Gated MoE | Demonstrates gater forgetting problem |

---

## 10. Implementation Checklist

### 10.1 Core Components

- [ ] Expert model with forward pass
- [ ] EWC engine with Fisher computation
- [ ] EWC engine with penalty computation
- [ ] Execution cost estimator
- [ ] Forgetting cost estimator (INPUT-DEPENDENT)
- [ ] VCG auction mechanism
- [ ] Expert pool manager

### 10.2 Training Pipeline

- [ ] Bid collection from all experts
- [ ] Auction to select winner
- [ ] Winner training with EWC penalty
- [ ] Fisher update after task (only for participants)
- [ ] Logging of bid components

### 10.3 Evaluation Pipeline

- [ ] Confidence-based routing (NO LABELS)
- [ ] Per-task accuracy computation
- [ ] Forgetting metric computation
- [ ] Multi-seed execution
- [ ] Statistical significance tests

### 10.4 Diagnostics

- [ ] Bid component breakdown (α signal, β signal)
- [ ] Expert win distribution
- [ ] Fisher statistics per expert
- [ ] EWC penalty activation logging

---

## 11. Appendix: Mathematical Derivations

### A.1 VCG Truthfulness Proof

**Claim:** In a second-price auction, bidding true value is dominant strategy.

**Setup:** 
- Your true value: v
- Your bid: b
- Highest other bid: h

**Case 1: v > h (you should win)**
- If b >= h: You win, pay h. Utility = v - h > 0 ✓
- If b < h: You lose. Utility = 0 ✗

**Case 2: v < h (you should lose)**
- If b <= h: You lose. Utility = 0 ✓
- If b > h: You win, pay h. Utility = v - h < 0 ✗

**Conclusion:** Bidding b = v is optimal regardless of h. □

### A.2 Forgetting Cost Derivation

**Goal:** Estimate damage to previous knowledge from training on (x, y).

**EWC penalty:** L_ewc = (λ/2) * Σ_i F_i * (θ_i - θ*_i)²

**After gradient step:** θ'_i = θ_i - η * ∂L/∂θ_i

**Change in EWC penalty:**
```
ΔL_ewc ≈ Σ_i F_i * (θ'_i - θ*_i)² - F_i * (θ_i - θ*_i)²
       ≈ Σ_i F_i * 2(θ_i - θ*_i) * (-η * ∂L/∂θ_i) + F_i * (η * ∂L/∂θ_i)²
```

**For small η, dominant term:**
```
forgetting_cost ∝ Σ_i F_i * (∂L/∂θ_i)²
```

This is the gradient interference: how much the update direction conflicts with important parameters.

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-02-02 | Initial specification |

---

## 13. References

1. Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting in neural networks" - EWC
2. Vickrey (1961). "Counterspeculation, Auctions, and Competitive Sealed Tenders" - VCG
3. Shazeer et al. (2017). "Outrageously Large Neural Networks" - Sparse MoE
4. Fedus et al. (2022). "Switch Transformers" - Efficient MoE routing
