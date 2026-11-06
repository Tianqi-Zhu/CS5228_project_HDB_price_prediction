"""Isolated MLP commands; does not import main's analysis implementation."""
from pathlib import Path
import runpy
import sys

HERE=Path(__file__).resolve().parent
COMMANDS={"fullfit":"mlp_fullfit_submission","recency-search":"mlp_recency_experiment",
          "intervals":"neural_ensemble_uncertainty","cleaning":"cleaned_neural_ensemble"}

def main():
    args=sys.argv[1:]
    if not args or args[0] in ("-h","--help"):
        print("MLP: verify | test | fullfit | recency-search | intervals | cleaning")
        print("Training commands require --output-dir; fullfit optionally accepts --half-life-months.")
        return
    command=args.pop(0)
    if command=="verify":
        runpy.run_path(str(HERE/"verify.py"),run_name="__main__");return
    sys.path.insert(0,str(HERE/"runtime"))
    if command=="test":
        import unittest,torch
        torch.set_num_threads(1)
        suite=unittest.defaultTestLoader.discover(str(HERE/"runtime/tests"),top_level_dir=str(HERE/"runtime"))
        result=unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if command not in COMMANDS:raise SystemExit("Unknown MLP command: "+command)
    sys.argv=[COMMANDS[command],*args]
    runpy.run_module("analysis.experiments."+COMMANDS[command],run_name="__main__")

if __name__=="__main__":main()
