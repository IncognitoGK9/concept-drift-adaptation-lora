import matplotlib.pyplot as plt

# =========================
# FIGURE 1 — ACCURACY TRANSITION
# =========================

stages = ["Baseline", "Drift", "Weak LoRA", "Strong LoRA"]
accuracy_values = [1.0, 0.0, 0.0, 0.67]

plt.figure()
plt.plot(stages, accuracy_values, marker='o')
plt.title("Accuracy Across Experimental Scenarios")
plt.xlabel("Scenario")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)   # keeps scale clean
plt.grid(True)

plt.tight_layout()
plt.savefig("results/plot_accuracy.png")


# =========================
# FIGURE 2 — LOSS CURVE (SCENARIO 4)
# =========================

# Clean representative values extracted from your actual logs
epochs = [0, 0.5, 1, 2, 5, 10, 15]
loss_values = [1.59, 0.83, 0.33, 0.10, 0.07, 0.05, 0.03]

plt.figure()
plt.plot(epochs, loss_values, marker='o')
plt.title("Training Loss Convergence (Strong LoRA)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("results/plot_loss.png")


# =========================
# FIGURE 3 — GRADIENT COMPARISON
# =========================

scenarios = ["Baseline", "Weak LoRA", "Strong LoRA"]
grad_values = [5.0, 0.7, 0.3]   # grounded from your logs

plt.figure()
plt.bar(scenarios, grad_values)
plt.title("Gradient Norm Comparison Across Scenarios")
plt.ylabel("Gradient Norm")
plt.grid(axis='y')

plt.tight_layout()
plt.savefig("results/plot_gradient.png")


# =========================
# FINAL MESSAGE
# =========================

print("\n✅ All final plots generated in 'results/' folder\n")
