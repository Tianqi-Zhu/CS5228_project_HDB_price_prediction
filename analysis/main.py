"""Curated main experiment entry points; help/list never train."""
from __future__ import annotations
import argparse
import runpy
import sys

ENTRIES = {
    'eda':'analysis.eda.run',
    'baseline':'analysis.experiments.baseline',
    'compare':'analysis.experiments.compare_models',
    'ladder':'analysis.experiments.report_selection_trials',
    'common-models':'analysis.experiments.common_feature_models',
}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list','verify',*ENTRIES], nargs='?',default='list')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.command in ('list','verify'):
        from review import main as review_main
        return review_main([args.command])
    if args.command=='ladder' and (not args.arguments or args.arguments[0]!='ladder'):
        args.arguments=['ladder',*args.arguments]
    sys.argv=[ENTRIES[args.command],*args.arguments]
    runpy.run_module(ENTRIES[args.command],run_name='__main__')

if __name__=='__main__': main()
