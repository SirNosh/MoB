### Phase 1 Baselines: Isolating the Core Idea

Your core hypothesis has two parts:
1.  Using multiple experts helps fight forgetting.
2.  Using a **market-based auction** to route data is a robust and effective way to make them specialize.

Your baselines should be designed to test these ideas independently, culminating in a direct showdown with the very architecture MoB aims to replace.

#### Baseline 1: The Sanity Check - "Naive" Fine-tuning

This is the simplest possible baseline and establishes the "lower bound" of performance.

*   **What it is:** A **single CNN** (with the same architecture as one of your experts) that is trained sequentially on the tasks from Split-MNIST without any continual learning strategies.
*   **Implementation:**
    ```python
    # A single model and a single optimizer
    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Loop through tasks and train the same model on new data
    for task in benchmark.train_stream:
        train(model, task.dataset, optimizer)
    ```
*   **What it Proves:** This model will experience maximum catastrophic forgetting. Its performance provides the floor that any viable CL method must beat.
*   **Question Answered:** *Does the MoB framework do anything at all to mitigate catastrophic forgetting?*

#### Baseline 2: The Market Test - "Random Assignment"

This baseline uses the exact same multi-expert architecture as MoB but removes the intelligent auction mechanism. This isolates the value of the market itself.

*   **What it is:** The same `ExpertPool` of 4 CNNs. However, for each batch of data, instead of running an auction, you assign it to an expert **chosen uniformly at random**. Each expert still uses the EWC penalty when it trains.
*   **Implementation:**
    ```python
    # Inside your training loop
    import random
    
    for x, y in train_dataloader:
        # Instead of an auction...
        winner_id = random.randint(0, self.config['num_experts'] - 1)
        
        # Train the randomly chosen expert
        winner = self.expert_pool.experts[winner_id]
        winner.train_on_batch(x, y, self.optimizers[winner_id])
    ```
*   **What it Proves:** This tests whether your sophisticated bidding system is providing a more useful routing signal than pure chance.
*   **Question Answered:** *Is the intelligent, bid-based routing mechanism genuinely better than random assignment?*

#### Baseline 3: The Architecture Test - "Monolithic EWC"

This is a crucial comparison. It pits the MoB *architecture* against the core CL *strategy* (EWC) being used on a standard, single model.

*   **What it is:** A **single CNN** (same architecture as one expert) trained sequentially on all tasks, but it **uses the same EWC penalty** that your MoB experts use. This is a very strong and common CL baseline.
*   **Implementation:**
    ```python
    # A single expert agent using the same training logic
    expert = MoBExpert(expert_id=0, model=SimpleCNN())
    optimizer = torch.optim.Adam(expert.model.parameters(), lr=0.001)

    for task in benchmark.train_stream:
        # Train the single expert on the whole task
        for x, y in task.dataset:
            expert.train_on_batch(x, y, optimizer)
        
        # Update its EWC parameters after the task
        expert.update_after_task(task.dataset)
    ```
*   **What it Proves:** This provides an "apples-to-apples" comparison of the CL strategy. It isolates the architectural benefit of MoB (a pool of specialists) versus a single, EWC-protected generalist.
*   **Question Answered:** *Does the MoB architecture of auction-driven specialization provide benefits **on top of** the underlying EWC strategy?*

#### Baseline 4: The "Knockout" Test - "Gated MoE" (The Central Planner)

This baseline directly confronts the "strategic flaw" that MoB is designed to solve. It replaces the auction with the very component we hypothesize is the problem: a centralized, learnable gater.

*   **What it is:** The exact same 4-expert CNN architecture as MoB. However, the VCG auction is removed. In its place, you implement a **standard, centralized, learnable gater network** (e.g., a small linear layer that takes batch data and outputs a softmax probability over the experts).
*   **Implementation:**
    ```python
    class GatedMoE(nn.Module):
        def __init__(self, num_experts, expert_model_factory):
            super().__init__()
            self.experts = nn.ModuleList([expert_model_factory() for _ in range(num_experts)])
            
            # The "monolithic dictator" gater network
            # This example assumes flattened input for the gater
            self.gater = nn.Sequential(
                nn.Linear(28*28, 128),
                nn.ReLU(),
                nn.Linear(128, num_experts)
            )

    # In the training loop...
    for x, y in train_dataloader:
        # Gater decides which expert to use
        gating_logits = gater(x.view(x.size(0), -1))
        
        # Top-1 routing: choose expert with highest score
        winner_id = torch.argmax(gating_logits, dim=-1)
        
        # ...route data to the chosen expert and train BOTH the expert and the gater...
    ```
*   **What it Proves:** This baseline directly tests the hypothesis of **"gater-level catastrophic forgetting."** The gater itself is a small neural network. As it is fine-tuned on the data from later tasks (e.g., Task 5), it is expected to forget how to correctly route data for earlier tasks (e.g., Task 1). This should cause a systemic failure, proving the robustness of MoB's stateless auction.
*   **Question Answered:** *Is a decentralized, stateless auction mechanism more robust to catastrophic forgetting than a centralized, stateful, learnable gater?*

### Summary and Phase 1 Success Criteria

Your Phase 1 validation is a definitive success if you can demonstrate the following hierarchy of performance, measured by Average Accuracy at the end of all tasks:

**MoB > Monolithic EWC > Gated MoE >= Random Assignment >> Naive**

*   **MoB >> Naive:** Confirms your system is fundamentally working.
*   **MoB > Random Assignment:** Confirms your auction mechanism is intelligent.
*   **MoB > Monolithic EWC:** Demonstrates an architectural advantage over a strong standard baseline.
*   **MoB > Gated MoE:** This is the **knockout result**. It provides direct evidence that replacing the centralized gater with a market mechanism is a superior design choice for continual learning, validating the core thesis of the entire project.
