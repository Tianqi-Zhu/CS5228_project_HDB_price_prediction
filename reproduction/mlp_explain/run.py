"""Isolated MLP explanation experiments; help and verify never train."""
from pathlib import Path
import runpy
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COMMANDS={'verify':'verify_mlp_delivery','intervals':'neural_ensemble_uncertainty',
          'cleaning':'cleaned_neural_ensemble','risk':'historical_risk_correction','tail':'recent_tail_correction'}
def main():
    args=sys.argv[1:]
    if not args or args[0] in ('-h','--help'):
        print('verify | diagnose | test | intervals | cleaning | risk | tail')
        print('verify/diagnose use archived evidence; other experiment commands train with explicit output directories.')
        return
    cmd=args.pop(0)
    sys.path.insert(0,str(HERE/'runtime'))
    if cmd=='diagnose':
        sys.argv=['diagnose',*args]
        runpy.run_path(str(HERE/'diagnose.py'),run_name='__main__');return
    if cmd=='test':
        import unittest,torch
        torch.set_num_threads(1)
        suite=unittest.defaultTestLoader.discover(str(HERE/'runtime/tests'),top_level_dir=str(HERE/'runtime'))
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    if cmd not in COMMANDS:raise SystemExit('Unknown command: '+cmd)
    sys.argv=[cmd,*args]
    runpy.run_module('analysis.experiments.'+COMMANDS[cmd],run_name='__main__')
if __name__=='__main__':main()
