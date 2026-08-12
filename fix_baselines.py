import os
import re

files = [
    "src/train_charm.py",
    "src/train_mahiro.py",
    "src/train_roma.py",
    "src/train_lrs.py"
]

for file in files:
    algo = os.path.basename(file).replace("train_", "").replace(".py", "")
    with open(file, 'r') as f:
        content = f.read()

    # 1. Add import time if not present
    if "import time" not in content:
        content = content.replace("import os\n", "import os\nimport time\n")

    # 2. Add start_time
    if "start_time = time.time()" not in content:
        content = content.replace("global_step = 0", "start_time = time.time()\n    global_step = 0")

    # 3. Replace hardcoded loss coefficients
    content = content.replace(
        "loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss",
        "loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss"
    )

    # 4. Fix logging and saving block at the end
    
    # We find the block:
    #         if update % args.eval_every == 0 or update == num_updates:
    #             print(...)
    #     if args.save_model: ...
    # And replace it with the new properly indented block.
    
    pattern = r"        if update % args.eval_every == 0 or update == num_updates:\n            print\(f\"\[\{run_name\}\] Update \{update\}/\{num_updates\}\"\)\n\n    if args.save_model:\n        os.makedirs\(f\"checkpoints/\{run_name\}\", exist_ok=True\)\n        torch.save\(agent.state_dict\(\), f\"checkpoints/\{run_name\}/" + algo + r"\.pt\"\)\n        write_completion\(run_name, \"" + algo + r"\", args.total_timesteps, global_step\)"
    
    replacement = f"""        if update % args.eval_every == 0 or update == num_updates:
            sps = int(global_step / (time.time() - start_time))
            mean_reward = w_rewards.sum(0).mean().item()
            win_rate = ((w_rewards[:, 0, :] > 5.0).any(dim=0).float().mean().item())
            print(f"[{{run_name}}] update={{update}} step={{global_step}} sps={{sps}} win_rate={{win_rate:.3f}} mean_reward={{mean_reward:.3f}}")
            if args.save_model:
                os.makedirs(f"checkpoints/{{run_name}}", exist_ok=True)
                torch.save(agent.state_dict(), f"checkpoints/{{run_name}}/{algo}.pt")

    if args.save_model:
        write_completion(run_name, "{algo}", args.total_timesteps, global_step)"""
    
    content = re.sub(pattern, replacement, content)
    
    with open(file, 'w') as f:
        f.write(content)

print("Baselines fixed.")
