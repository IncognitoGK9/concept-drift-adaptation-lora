# concept-drift-adaptation-lora
This repository contains the experimental framework and code for analyzing conditional adaptation using LoRA under concept drift.
Controlled study of LoRA-based adaptation under concept drift in conversational AI, demonstrating non-additive and interaction-dependent behavior.

# Concept Drift Adaptation with LoRA

This repository provides a complete experimental framework for studying how transformer-based conversational models behave under concept drift and how LoRA adaptation affects performance.

The experiments show that adaptation is not automatic, but depends on the interaction between training signal, data structure, and model capacity.

---

## 1. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Experiment Overview

This project follows a structured workflow:

1. Train a baseline model on stable data  
2. Test the model under concept drift → observe failure  
3. Apply weak adaptation → still fails  
4. Apply strong adaptation → partial recovery  
5. Run ablation experiments → isolate contributing factors  

---

## 3. Running the Experiments (Full Workflow)

Follow the steps below in order.

---

### Step 1 — Train Baseline Model

Run:

```bash
python train_baseline.py
```

What this does:

- trains the base model on stable data  
- learns three classes:
  - process (class 0)
  - complaints (class 1)
  - general (class 2)

Output:

```
./baseline_model
```

---

### Step 2 — Test Baseline Under Drift

Run:

```bash
python test_drift.py
```

What this does:

- automatically loads all available models  
- evaluates them on:
  - baseline inputs (known patterns)
  - drift inputs (new patterns)

Expected result:

- high accuracy on baseline  
- near-zero accuracy under drift  

This demonstrates failure under concept drift.

---

### Step 3 — Apply Weak Adaptation

Run:

```bash
python train_lora_weak.py
```

Then run:

```bash
python test_drift.py
```

What this does:

- applies LoRA with:
  - small dataset
  - low rank (r=8)

Output:

```
./lora_weak_model
```

Expected result:

- no improvement under drift  

This shows weak adaptation is insufficient.

---

### Step 4 — Apply Strong Adaptation

Run:

```bash
python train_lora_strong.py
```

Then run:

```bash
python test_drift.py
```

What this does:

- applies LoRA with:
  - stronger training signal
  - balanced class structure
  - higher rank (r=16)

Output:

```
./lora_strong_model
```

Expected result:

- partial recovery under drift  

This shows adaptation requires sufficient conditions.

---

### Step 5 — Run Ablation Experiments

Run:

```bash
python train_ablation_rank.py
python train_ablation_data.py
python train_ablation_balance.py
```

Then evaluate:

```bash
python test_drift.py
```

Outputs:

```
./rank_only_model
./data_only_model
./balanced_only_model
```

Expected result:

- all fail under drift  

This demonstrates that no single factor enables adaptation.

---

## 4. How Testing Works

The file `test_drift.py`:

- tests all models automatically  
- runs once per model  
- skips missing models  
- prints results clearly  

Example output:

```
========== Baseline ==========
Baseline accuracy: 1.0
Drift accuracy: 0.0

========== Weak LoRA ==========
Baseline accuracy: 1.0
Drift accuracy: 0.0

========== Strong LoRA ==========
Baseline accuracy: ...
Drift accuracy: 0.67
```

---

## 5. Generating Figures

### Main Results

Run:

```bash
python plot_results.py
```

This generates:

- accuracy transition  
- training loss convergence  
- gradient comparison  

Saved in:

```
results/
```

---

### Ablation Plot

Run:

```bash
python plot_ablation.py
```

This generates:

- performance comparison across adaptation conditions  

---

## 6. Project Structure

```
train_baseline.py           baseline training

train_lora_weak.py          weak adaptation
train_lora_strong.py        strong adaptation

train_ablation_rank.py      rank-only experiment
train_ablation_data.py      data-only experiment
train_ablation_balance.py   balanced structure experiment

test_drift.py               evaluates all models automatically

plot_results.py             main results
plot_ablation.py            ablation results
```

---

## 7. Key Findings

- Baseline fails under drift  
- Weak adaptation fails  
- Single-factor changes fail  
- Adaptation is non-additive  
- Recovery requires multiple conditions  

---

## 8. Important Notes

- Experiments are deterministic  
- Testing runs once per model  
- Results are reproducible  
- Missing models are skipped automatically  

---

## 9. Summary

Effective adaptation under concept drift is not guaranteed by fine-tuning alone. It emerges only when training signal, data structure, and model capacity work together.
