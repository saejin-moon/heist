import ast
import glob
import os
import pytest

def get_args_dataclass_attrs(tree):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Args":
            attrs = set()
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    attrs.add(stmt.target.id)
            return attrs
    return None

def get_parser_dests(tree):
    dests = set()
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
            if dest:
                dests.add(dest)
    return dests

def test_all_trainers_args_consistency():
    train_scripts = sorted(glob.glob("src/train_*.py"))
    assert len(train_scripts) > 0, "No train scripts found."
    
    inconsistencies = []
    
    for script in train_scripts:
        with open(script) as f:
            tree = ast.parse(f.read())
            
        args_attrs = get_args_dataclass_attrs(tree)
        if args_attrs is None:
            continue
            
        parser_dests = get_parser_dests(tree)
        
        missing_in_args = parser_dests - args_attrs
        if missing_in_args:
            inconsistencies.append(f"{os.path.basename(script)} is missing args in Args dataclass: {missing_in_args}")
            
    if inconsistencies:
        pytest.fail("\n".join(inconsistencies))
