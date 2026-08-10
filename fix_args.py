import re

updates = {
    "src/train_coma.py": [
        "    cir_coef: float = 0.0",
        "    save_model: bool = True",
        "    load_checkpoint: str = \"\"",
        "    no_cuda: bool = False",
        "    use_rnd: bool = False",
        "    rnd_coef: float = 0.05"
    ],
    "src/train_ippo.py": [
        "    save_model: bool = True",
        "    load_checkpoint: str = \"\"",
        "    no_cuda: bool = False",
        "    use_rnd: bool = False",
        "    rnd_coef: float = 0.05"
    ],
    "src/train_mappo.py": [
        "    cir_coef: float = 0.0",
        "    target_kl: float = 0.0",
        "    save_model: bool = True",
        "    load_checkpoint: str = \"\"",
        "    no_cuda: bool = False",
        "    use_rnd: bool = False",
        "    rnd_coef: float = 0.05"
    ],
    "src/train_qmix.py": [
        "    load_checkpoint: str = \"\"",
        "    no_cuda: bool = False",
        "    use_rnd: bool = False",
        "    rnd_coef: float = 0.05"
    ]
}

for filepath, lines_to_add in updates.items():
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find the end of the Args class. It's usually a block of indented lines.
    # We will find the line containing `class Args:` and then skip until we hit a blank line or `def `.
    lines = content.split('\n')
    out_lines = []
    in_args = False
    args_end_inserted = False
    
    for line in lines:
        if line.startswith("class Args:"):
            in_args = True
            out_lines.append(line)
            continue
            
        if in_args:
            # Check if we have exited the Args class (unindented line or def)
            if line.strip() == "" and len(out_lines) > 0 and out_lines[-1].strip() == "":
                # Wait, it might have multiple blank lines. We insert before def.
                pass
            if line.startswith("def ") or (line.strip() != "" and not line.startswith("    ")):
                in_args = False
                for new_l in lines_to_add:
                    out_lines.append(new_l)
                out_lines.append("")
                
        out_lines.append(line)
        
    with open(filepath, 'w') as f:
        f.write("\n".join(out_lines))
        
print("done")
