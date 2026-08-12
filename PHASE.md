You are tasked with executing a rigorous, exhaustive, zero-shortcut audit and refactoring of this entire codebase to make it production-ready for training (`heist`).

DO NOT assume any existing code or math is correct. DO NOT give high-level summaries without doing the underlying work. You must explicitly verify every formula, execute tool commands, and show your work at every step.

Follow this multi-phase workflow strictly:

---

### PHASE 1: Baseline Health, Linting & Formatting
1. Run `ruff check .` and `ruff format .` across the entire codebase. Fix ALL errors and warnings.
2. Run `pytest` to establish the current baseline test results. 
3. Record all failing tests or syntax/type issues before moving forward.

---

### PHASE 2: Rigorous Theoretical & Mathematical Audit
1. Audit all papers, documents, and LaTeX/markdown specs inside the `paper/` directory.
2. Extract EVERY mathematical equation, algorithm, loss function, theoretical claim, or parameter definition from the `paper/` dir.
3. Write down a breakdown mapping:
   - **Theory / Paper Formula** -> **Target Code File & Line Numbers**
4. Perform an explicit line-by-line mathematical audit. Specifically check for:
   - Sign errors (+/-), missing normalization constants, missing log/exp epsilons for numerical stability.
   - Matrix dimension/tensor shape mismatches (e.g., transposed multiplications, batch dimension handling).
   - Off-by-one errors in indexing, sequence lengths, or sliding windows.
   - Incorrect loss scaling, gradient clipping implementation, or optimizer step adjustments.
5. If the implementation differs from the paper in ANY way, fix the code or document if it was an intentional deviation (and confirm if it is mathematically valid).

---

### PHASE 3: Implementation vs. Theory Alignment & Code Audit
1. Deep-dive into all Python/C++/CUDA scripts outside of `paper/`.
2. Trace the data flow from data loading -> model forward pass -> loss calculation -> backward pass -> optimization step.
3. Verify that edge cases are handled (e.g., empty batches, NaNs/Infs in activations/gradients, zero-division, padding masks).
4. Verify hyperparameter wiring: Ensure parameters defined in config files or scripts actually get passed to the underlying theoretical models without silent overrides or hardcoded default drops.

---

### PHASE 4: Dead Code Pruning & Refactoring
1. Scan the repository for unused files, dead functions, orphaned variables, deprecated helper scripts, and unused imports.
2. Explicitly list the files and functions marked for deletion before removing them.
3. Remove all unused code/files. Ensure the directory structure is clean, lean, and minimal.

---

### PHASE 5: Performance Optimization
1. Review the computational pipeline for bottlenecks:
   - Check tensor operations: Replace inefficient Python loops with vectorized PyTorch/NumPy operations.
   - Check GPU/CPU memory efficiency (e.g., unnecessary tensor allocations, unreleased memory, missing `torch.no_grad()` in evaluation/inference paths).
   - Check DataLoader efficiency: verify `pin_memory`, `num_workers`, and prefetching settings where applicable.
2. Refactor performance bottlenecks without altering mathematical accuracy.

---

### PHASE 6: Test Suite Expansion
1. Audit the existing tests for coverage gaps, especially around core theoretical modules and loss functions.
2. Add new `pytest` unit and integration tests:
   - Tests checking exact mathematical correctness using known synthetic inputs/outputs.
   - Shape-checking tests for all neural network layers/modules.
   - Edge case tests (NaN checks, zero-inputs, max sequence lengths, boundary conditions).
3. Re-run `pytest` and verify that **100% of tests pass cleanly**.

---

### PHASE 7: Production Readiness & Verification for `train.zsh`
1. Inspect `train.zsh` and all underlying scripts invoked by it.
2. Verify environment variables, path references, CLI flags, hardware/GPU target configurations, and logging output paths.
3. Perform a dry-run or a minimal 1-step iteration test (if possible without training full model) to ensure running `train.zsh` will execute smoothly with ZERO runtime errors, warnings, or missing dependencies.

---

### PHASE 8: Version Control & Clean Exit
1. Run `git status` and run ruff format and check one final time to guarantee clean execution.
2. Run `git add .`.
3. Commit the changes with a highly detailed commit message containing:
   - Summary of mathematical/theoretical fixes (with file and line numbers).
   - List of pruned files/code.
   - Optimizations and performance upgrades added.
   - Summary of new tests added and full test pass status.
4. Run `git push`.

Start with Phase 1 now. Show every command output and detail every code change.

---

### 1. Mathematical Adjustments & Theory Alignment
* **MAPPO+CAR Synchronization:** I found that `train_comm.py` and `train_coma.py` were computing the Counterfactual Affordance Reward (CAR) using the next state's value ($V(s_{t+1})$) rather than the masking delta ($\Delta \text{Mask}_j(s_t, a_{i,t})$) defined in Section 3.4 of the paper and implemented correctly in `train_mappo.py`. I have synchronized them to compute exactly $\sum \Delta \text{Mask}$.
* **Paper Normalization Correction:** I updated the formulation of Macro Weighting ($\Omega_t$) in `paper/06_dynamic_skill_routing.qd` and `COOP.md` from $\exp(-\alpha A_t)$ to $\exp\left(-\alpha \frac{A_t}{A_{max}}\right)$ to properly reflect the normalization logic deployed in the codebase (so $A_t = 100$ doesn't erroneously flatten gradients to zero).

### 2. Missing Training Arguments (`eval_every` Crash Prevention)
* The Hierarchical MARL trainers (`train_charm.py`, `train_mahiro.py`, `train_roma.py`, `train_lrs.py`, and `train_coop.py`) utilized `args.eval_every` in their logging loops, but the parameter was entirely missing from their respective `Args` dataclass and `parse_args()`.
* I successfully backfilled `--eval-every` into their dataclasses to prevent `AttributeError` crashes upon hitting evaluation milestones.

### 3. Missing Checkpoint Transfer Logic
* H-MARL models were missing the stage-to-stage `ppo_utils.get_previous_stage_checkpoint` curriculum transfer logic that `ippo`, `mappo`, `qmix`, and `coma` possessed.
* I wired the `load_matching_weights` pipeline correctly into `train_charm.py`, `train_mahiro.py`, `train_roma.py`, `train_lrs.py`, and `train_coop.py` so `--resume` and `--from-stage` CLI flags in `train.zsh` will correctly warm-start them. 

### 4. PyTorch Safety Fix
* Upgraded all `torch.load` calls (like in `ppo_utils.py` and `run_eval.py`) to include `weights_only=True` to fix the `FutureWarning` and improve load security.

