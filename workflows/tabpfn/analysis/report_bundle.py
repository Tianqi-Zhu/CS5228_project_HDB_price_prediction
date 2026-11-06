"""Refresh a small reading bundle from local experiment outputs, without training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil

from analysis.pipeline.config import ROOT

RUNS = ROOT / 'output/experiment_runs'
READING_FILES = {
    '01_final/README.md': RUNS / '00_report_and_final/FINAL_MODEL.md',
    '01_final/EXPERIMENT_REPORT.md': RUNS / '00_report_and_final/REPORT.md',
    '01_final/submission_comparison.csv': RUNS / '00_report_and_final/final_submission_comparison.csv',
    '01_final/submission_selection.json': RUNS / '00_report_and_final/FINAL_SELECTION.json',
    '02_feature_engineering/BASELINE_27_FEATURES.md': RUNS / 'foundation_feature_review/BASELINE_REPORT.md',
    '02_feature_engineering/FEATURE_ABLATION.md': RUNS / 'features_ablation_evidence/README.md',
    '02_feature_engineering/feature_dictionary.csv': RUNS / 'foundation_feature_review/feature_dictionary.csv',
    '02_feature_engineering/single_feature_effects.csv': RUNS / 'foundation_feature_review/single_feature_effects.csv',
    '02_feature_engineering/group_feature_effects.csv': RUNS / 'foundation_feature_review/group_feature_effects.csv',
    '02_feature_engineering/controlled_feature_ladder.csv': RUNS / '00_report_and_final/controlled_feature_ladder.csv',
    '03_model_comparison/PROTOCOL.md': RUNS / 'comparison_same_27_features/README.md',
    '03_model_comparison/same_input_models.csv': RUNS / 'comparison_same_27_features/model_metrics.csv',
    '03_model_comparison/recommended_models.csv': RUNS / '00_report_and_final/recommended_models.csv',
    '03_model_comparison/model_catalog.csv': RUNS / '00_report_and_final/model_catalog.csv',
    '03_model_comparison/model_feature_comparison.png': RUNS / '00_report_and_final/report_comparison.png',
}
ROUND_NAMES = (
    'round_01_history_target_search', 'round_02_anniversary_history',
    'round_03_tree_transformer_ensemble', 'round_04_macro_spatial_diagnostics',
    'round_05_feature_interactions', 'round_06_deep_models',
    'round_07_pretrained_transformers', 'round_08_feature_pruning_seeds',
    'round_09_grouped_blending', 'round_10_plr_transformer_final',
)
for folder in ROUND_NAMES:
    READING_FILES['04_optimization_history/' + folder.removeprefix('round_') + '.md'] = RUNS / folder / 'README.md'


def rewrite_links(body, source, destination):
    preferred = {str(path.resolve()): ROOT / 'experiment' / name for name, path in READING_FILES.items()}
    # The full original report has its own navigation page in the reading bundle.
    preferred[str((RUNS / '00_report_and_final/README.md').resolve())] = ROOT / 'experiment/01_final/README.md'
    preferred[str((RUNS / 'foundation_feature_review/README.md').resolve())] = ROOT / 'experiment/02_feature_engineering/README.md'
    preferred[str((ROOT / 'project_report.md').resolve())] = ROOT / 'experiment/02_feature_engineering/BASELINE_27_FEATURES.md'
    def replace(match):
        target = match.group(1)
        if target.startswith(('https:', 'http:', 'mailto:', '#', 'data:')):
            return match.group()
        local, mark, fragment = target.partition('#')
        if not local:
            return match.group()
        resolved = (source.parent / local).resolve()
        for old, new in ((ROOT / 'reports/experiments', RUNS),
                         (ROOT / 'reports/foundation_feature_review', RUNS / 'foundation_feature_review'),
                         (ROOT / 'reports/maintenance', ROOT / 'docs/maintenance')):
            if resolved.is_relative_to(old):
                resolved = new / resolved.relative_to(old)
                break
        chosen = preferred.get(str(resolved), resolved)
        relative = Path(os.path.relpath(chosen, destination.parent)).as_posix()
        return '](' + relative + (mark + fragment if mark else '') + ')'
    return re.sub(r'\]\(([^)]+)\)', replace, body)


def refresh():
    evidence = []
    for name, source in READING_FILES.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = ROOT / 'experiment' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == '.md':
            body = rewrite_links(source.read_text(encoding='utf-8-sig'), source, destination)
            destination.write_text(body, encoding='utf-8', newline='\n')
        else:
            shutil.copy2(source, destination)
        evidence.append({'report': destination.relative_to(ROOT).as_posix(),
                         'source': source.relative_to(ROOT).as_posix()})
    manifest = ROOT / 'output/checks/records/reading_bundle_sources.json'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'mode': 'Copy selected existing results; no training', 'files': evidence},
                                   indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Refreshed {len(evidence)} reading files; raw trials remain in output/experiment_runs.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    refresh()


if __name__ == '__main__':
    main()
