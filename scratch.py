import ast
import glob
import os

for f in sorted(glob.glob("src/train_*.py")):
    with open(f) as file:
        tree = ast.parse(file.read())
        
    args_class = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            args_class = node
            break
            
    if not args_class:
        continue
        
    dataclass_attrs = set()
    for node in args_class.body:
        if isinstance(node, ast.AnnAssign):
            dataclass_attrs.add(node.target.id)
            
    parser_args = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value.startswith("--"):
                    # Extract typical dest (replace '-' with '_')
                    dest = arg.value.lstrip("-").replace("-", "_")
                    parser_args.add(dest)
            # check kwargs for dest=...
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    parser_args.add(kw.value.value)
                    
    # The default behavior of argparse is to use the first long option, stripping -- and converting - to _
    # My simple logic above catches the dest override and the default dest.
    # We should see which parser args are not in dataclass_attrs
    missing_in_dataclass = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            dest = None
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    dest = kw.value.value
            if not dest:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value.startswith("--"):
                        dest = arg.value.lstrip("-").replace("-", "_")
                        break
            if dest and dest not in dataclass_attrs:
                missing_in_dataclass.add(dest)
                
    if missing_in_dataclass:
        print(f"{os.path.basename(f)}: missing {missing_in_dataclass}")

