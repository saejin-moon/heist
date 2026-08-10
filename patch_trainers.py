import glob
import re

for file_path in glob.glob("src/train_*.py"):
    with open(file_path, "r") as f:
        content = f.read()

    if "win_rate = " in content and "dim=0" in content:
        continue # Already patched
    
    # Replace avg_reward calculation to include win_rate calculation right after it, matching the exact indentation
    def replace_avg_reward(match):
        indent = match.group(1)
        original_avg_reward = match.group(2)
        return f'{indent}{original_avg_reward}\n{indent}win_rate = (buffers[list(AGENTS)[0]]["rewards"] > 5.0).any(dim=0).float().mean().item() if hasattr(buffers[list(AGENTS)[0]]["rewards"], "dim") else 0.0'
    
    content = re.sub(
        r'^([ \t]*)(avg_reward\s*=\s*np\.mean\(\[buffers\[a\]\["rewards"\]\.mean\(\)\.item\(\)\s*for\s*a\s*in\s*AGENTS\]\))',
        replace_avg_reward,
        content,
        flags=re.MULTILINE
    )
    
    # Replace print to include win_rate
    def replace_print(match):
        prefix = match.group(1)
        mean_reward_part = match.group(2)
        suffix = match.group(3)
        return f'{prefix}win_rate={{win_rate:.3f}} {mean_reward_part}{suffix}'

    content = re.sub(
        r'(print\(\s*f"[^"]*?)sps=\{sps\}([^"]*)(mean_reward=\{[^}]+\})([^"]*"\s*\))',
        r'\1sps={sps}\2win_rate={win_rate:.3f} \3\4',
        content,
        flags=re.MULTILINE
    )
    
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Patched {file_path}")
