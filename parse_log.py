import re
from collections import defaultdict

with open("/home/fuddle/.gemini/antigravity-cli/brain/b7007ceb-f77e-49ae-814f-d320a67d2a2b/.system_generated/tasks/task-463.log") as f:
    lines = f.readlines()

data = defaultdict(lambda: defaultdict(str))

algos = [
    "ippo", "mappo", "mappo_car", "mappo_cir",
    "comm", "comm_cir", "comm_cir_car", "qmix",
    "coma", "coma_cir", "ate", "loo", "macca",
    "marc", "marc_no_shielding", "marc_no_macro", "marc_no_affordance"
]

current_stage = None
for line in lines:
    if "EVALUATING STAGE" in line:
        m = re.search(r"EVALUATING STAGE (\d+)", line)
        if m:
            current_stage = int(m.group(1))
    else:
        parts = line.strip().split()
        if len(parts) >= 7 and parts[0] in algos and parts[1] == "s0":
            algo = parts[0]
            win = float(parts[2])
            term = float(parts[3])
            ret = float(parts[6])
            data[current_stage][algo] = f"{win:.3f} / {term:.3f} / {ret:.2f}"

print("| Model | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |")
print("|-------|---------|---------|---------|---------|---------|")

for algo in algos:
    row = [f"**{algo}**"]
    for stage in [0, 1, 2, 3, 4]:
        val = data[stage].get(algo, "N/A")
        row.append(val)
    print(" | ".join(row))
