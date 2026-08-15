import os
import glob
import re

def clean_parsers():
    src_dir = '/home/bae/heist/src'
    train_files = glob.glob(os.path.join(src_dir, 'train_*.py'))
    
    for file in train_files:
        with open(file, 'r') as f:
            content = f.read()
            
        # Remove argparse lines for car-coef and cir-coef
        content = re.sub(r'[ \t]*[a-zA-Z_0-9]+\.add_argument\("--(car|cir)-coef",.*?\n', '', content)
        
        # Write back if changed
        with open(file, 'w') as f:
            f.write(content)

    print("Parser arguments --car-coef and --cir-coef removed.")

if __name__ == '__main__':
    clean_parsers()
