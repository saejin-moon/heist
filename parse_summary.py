import json
with open("results/run020/summary_new.json") as f:
    data = json.load(f)["stage_0"]
for model in ["lrs", "roma", "mahiro", "charm", "coop", "coop_fixed", "coop_no_car", "coop_top_down"]:
    runs = data.get(model, [])
    if not runs: continue
    print(f"\n{model}:")
    for r in runs:
        m = r["metrics"]
        print(f"Seed {r['seed']}: Win {m['win_rate']:.2f}, Alarm {m['mean_alarm']:.1f}, Length {m['mean_length']:.1f}, Term {m['terminal_rate']:.2f}, Ext {m['extraction_rate']:.2f}")
