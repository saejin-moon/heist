import os
import re

paper_dir = 'paper'
qd_files = [f for f in os.listdir(paper_dir) if f.endswith('.qd')]

for file in qd_files:
    filepath = os.path.join(paper_dir, file)
    with open(filepath, 'r') as f:
        content = f.read()

    # Fix block math: replace isolated $$ with $$$ (but avoid $$$ -> $$$$)
    # Actually, the rule says multiline block math must use $$$ not $$.
    # So if there's any $$, replace it with $$$.
    # Be careful not to replace $$$ with $$$$.
    content = re.sub(r'(?<!\$)\$\$(?!\$)', '$$$', content)

    # Fix inline math spaces
    # Find all inline math $ ... $ and ensure spaces
    # It's tricky because of $$$ block math.
    
    # We can split by $$$ first
    parts = content.split('$$$')
    
    for i in range(len(parts)):
        # If it's outside block math (even index)
        if i % 2 == 0:
            # find all $...$ and add spaces
            # $foo$ -> $ foo $
            # using a regex that finds $ followed by anything except $ and ends with $
            def add_spaces(match):
                inner = match.group(1).strip()
                return f'$ {inner} $'
                
            parts[i] = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', add_spaces, parts[i])

    new_content = '$$$'.join(parts)
    
    with open(filepath, 'w') as f:
        f.write(new_content)

print("Math formatting fixed.")
