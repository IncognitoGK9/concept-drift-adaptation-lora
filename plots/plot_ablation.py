# Plot ablation results showing that adaptation requires joint conditions

import matplotlib.pyplot as plt


scenarios = [
    "Weak LoRA",
    "Rank Only",
    "Data Only",
    "Balanced Only",
    "Strong LoRA"
]

accuracies = [0.0, 0.0, 0.0, 0.0, 0.67]


plt.figure()

plt.plot(scenarios, accuracies, marker="o")

plt.xlabel("Adaptation Condition")
plt.ylabel("Accuracy")
plt.title("Adaptation Requires Joint Conditions")

plt.ylim(0, 1.0)
plt.grid()

plt.tight_layout()
plt.savefig("ablation.png")

plt.show()