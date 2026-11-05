"""Curated main experiment entry points; help/list never train."""
from __future__ import annotations
import argparse
from pathlib import Path

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command')
    planned=commands.add_parser('planned',help='Run main requirement-card audits and frozen-prediction diagnostics; no training')
    planned.add_argument('--output-dir',type=Path)
    concat=commands.add_parser('concat',help='Concat Transformer: verify, replay, prepare or explicitly train')
    concat.add_argument('concat_args',nargs=argparse.REMAINDER)
    commands.add_parser('list',help='List experiments without running them')
    commands.add_parser('verify',help='Verify shipped file hashes without training')
    eda=commands.add_parser('eda',help='Run EDA only')
    eda.add_argument('--scope',choices=('main','auxiliary','overview','all'),default='all')
    eda.add_argument('--output-dir',type=Path)
    baseline=commands.add_parser('baseline',help='Train fixed C7 baselines')
    baseline.add_argument('--output-dir',type=Path)
    compare=commands.add_parser('compare',help='Train the fixed traditional-model comparison')
    compare.add_argument('--output-dir',type=Path)
    compare.add_argument('--baseline-dir',type=Path)
    ladder=commands.add_parser('ladder',help='Refit the fixed feature ladder')
    ladder.add_argument('--stage',choices=('outer','june'),default='outer')
    common=commands.add_parser('common-models',help='Refit frozen common-input model selections')
    common.add_argument('--output-dir',type=Path)
    common.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args(argv)
    if args.command=='planned':
        from analysis.validation.main import run
        return run(args.output_dir)
    if args.command=='concat':
        import sys
        from analysis.experiments.concat_transformer.__main__ import main as concat_main
        sys.argv=[sys.argv[0]]+args.concat_args
        return concat_main()
    if args.command in (None,'list','verify'):
        from review import main as review_main
        return review_main([args.command or 'list'])
    if args.command=='eda':
        from analysis.eda.run import main as run
        return run(scope=args.scope,output_dir=args.output_dir)
    if args.command=='baseline':
        from analysis.experiments.baseline import main as run
        return run(output_dir=args.output_dir)
    if args.command=='compare':
        from analysis.pipeline.config import EXPERIMENT_REPORTS
        from analysis.experiments.baseline import main as prepare
        from analysis.experiments.compare_models import main as run
        directory=args.baseline_dir or EXPERIMENT_REPORTS/'baseline_mean_catboost'
        if not (directory/'results.json').exists():
            prepare(output_dir=directory)
        return run(output_dir=args.output_dir,baseline_dir=directory)
    if args.command=='ladder':
        from analysis.experiments.report_selection_trials import ladder as run
        return run(args.stage)
    if args.command=='common-models':
        from analysis.experiments.common_feature_models import OUT,run
        return run(args.output_dir or OUT,args.device)

if __name__=='__main__': main()
