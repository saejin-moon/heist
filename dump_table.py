import json

with open("results/run009/summary.json") as f:
    data = json.load(f)

print("| Model | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |")
print("|-------|---------|---------|---------|---------|---------|")

algos = [
    "ippo", "mappo", "mappo_car", "mappo_cir",
    "comm", "comm_cir", "comm_cir_car", "qmix",
    "coma", "coma_cir", "ate", "loo", "macca",
    "marc", "marc_no_shielding", "marc_no_macro", "marc_no_affordance"
]

def fmt(metrics):
    if not metrics: return "N/A"
    return f"{metrics.get('win_rate',0):.3f} / {metrics.get('terminal_rate',0):.3f} / {metrics.get('mean_return',0):.2f}"

for algo in algos:
    row = [f"**{algo}**"]
    for stage in [0, 1, 2, 3, 4]:
        stage_key = f"stage_{stage}"
        # We average over seeds if multiple exist, or just take seed 0 for brevity. Let's just take seed 0.
        entries = data.get(stage_key, {}).get(algo, [])
        if not entries:
            row.append("N/A")
        else:
            # take seed 0
            m = [e for e in entries if e['seed'] == 0]
            if m:
                m = m[0]['metrics']
            else:
                m = entries[0]['metrics']
            row.append(fmt(m))
    print(" | ".join(row))
