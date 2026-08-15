import os
import re

def remove_ablations():
    src_dir = '/home/bae/heist/src'
    
    # Files to modify
    eval_stage = os.path.join(src_dir, 'eval_stage.py')
    run_eval = os.path.join(src_dir, 'run_eval.py')
    
    # 1. Clean eval_stage.py
    with open(eval_stage, 'r') as f:
        content = f.read()
    
    # Remove specific dictionary entries for ablations
    content = re.sub(r'\s*"marc_no_[^"]+":\s*\{[^}]+\},?\n', '\n', content)
    content = re.sub(r'\s*"coop_(fixed|no_car|top_down)":\s*\{[^}]+\},?\n', '\n', content)
    
    with open(eval_stage, 'w') as f:
        f.write(content)
        
    # 2. Clean run_eval.py
    with open(run_eval, 'r') as f:
        content = f.read()
        
    content = re.sub(r'algo in \["coop", "coop_fixed", "coop_no_car"\]', 'algo == "coop"', content)
    content = re.sub(r'if algo == "coop_top_down":.*?(elif|else|if)', r'\1', content, flags=re.DOTALL)
    content = re.sub(r'\s*"coop_top_down",\n', '\n', content)
    
    with open(run_eval, 'w') as f:
        f.write(content)

    print("Ablations removed from eval_stage.py and run_eval.py")

if __name__ == '__main__':
    remove_ablations()
