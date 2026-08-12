import re
import os

def remove_lines(filepath, patterns):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    skip = False
    for line in lines:
        match = False
        for p in patterns:
            if re.search(p, line):
                match = True
                break
        if not match:
            new_lines.append(line)
            
    with open(filepath, 'w') as f:
        f.writelines(new_lines)

# 1. src/env.py
remove_lines("src/env.py", [
    r"metadata\s*=\s*",
    r"def _get_obs",
    r"def observation_space",
    r"def action_space",
    r"self.breach_pos"
])

# Wait, the `def _get_obs` and others are multiline functions! I need a better way to remove multiline blocks.
