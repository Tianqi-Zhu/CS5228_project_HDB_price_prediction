"""Maintain appendix decision cards and manifest after deliberate evidence changes.

This is a maintainer command, not a result-generating or training command.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import os
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

CARDS=[
('X01','archived','特征冗余与空间关联','历史特征是否重复编码？设施之间的关联是否受房型/月份混杂？',
 '固定模型删列/组，与完整输入比较；空间调整模型另列，不把相关当预测增量。',
 '删除时间与历史的联合变化远大于单组；MRT×商场交互补充主线单变量关联。重叠组效应不可相加。',
 '保留支持历史信息的消融和完整空间控制；没有证据据此推导设施因果效应。','M01/M03/M04',
 ['experiment/appendix/X01/feature_evidence/group_feature_effects.csv','experiment/appendix/X01/spatial_controls/reports/spatial_features/amenity_interactions/mrt_mall_interaction_summary_zh.md']),
('X02','archived','清洗敏感性与极端标签','低租金或重复输入是否应删除？',
 'MLP/TabM/hybrid 各与自己的未清洗版本比较，保留全部验证行；ReLU/IQR 另有协议。',
 '轻度清洗只改善部分模型；更强删除未稳定降低 RMSE，不能凭低租金就断定标签错误。',
 '主线保留全部记录，附录记录删除代价、实际行数与独立校准设置。','M01/M04',
 ['experiment/appendix/X06/mlp_explain/cleaning/comparison_metrics.csv','experiment/appendix/X06/mlp_explain/cleaning/protocol.json','experiment/appendix/X02/relu_historical/README.md']),
('X03','archived','另两提交与信息条件','为何从历史/宏观/结构三份中保留结构版？',
 '三份原配方、原 CSV 和 Y/J 开发指标；显示分数只作档案，宏观分支查询期实际观测需单列。',
 '三份均可 CPU 重组；三选一不等于所有消融中普遍最优。',
 '另两版本完整保留为对照，不把宏观额外信息的增益解释成架构增益。','M06/M09',
 ['reproduction/three_submissions/comparison.csv','reproduction/three_submissions/README.md']),
('X04','archived','种子、覆盖分组与结构负结果','复杂结构单独好、加入融合就一定好么？',
 '第8/9/10轮的固定规则、不同成员集合、早期锁定和后期探索分开；内部bagging不作外部重复。',
 'PLR 相对无新增对照年度 RMSE +0.164455、近期 -0.124548；同一模型可给出窗口依赖的结论。',
 '保留年度更好的无新增/PLE对照和负结果；禁止只报告挑中的最终分数。','M04/M05',
 ['experiment/appendix/X04/round_10_plr_transformer_final/blend_comparison.csv','experiment/appendix/X04/round_09_grouped_blending/README.md']),
('X05','archived','时间序列与动态租金面替代路线','把市场动态压缩成 latent 或 PCA 租金面能替代现有历史表示么？',
 'RNN/LSTM 对同记录的无latent基线，月更新；动态面固定原点，并为丢失2021拟合行设置2022+ CatBoost控制。',
 'RNN/LSTM 未稳定改善匹配基线；PCA历史重建好不等于未来面预测好，残差hybrid增益随折变化。',
 '作为替代表示和失败/条件增益归档；月更新、面诊断、逐行固定原点三种成绩不混排。','M03/M05',
 ['experiment/appendix/X05/external_studies/rnn_lstm/docs/rnn_lstm_report.md','experiment/appendix/X05/external_studies/dynamic_surface/reports/factored_rental_surface/report.md','experiment/appendix/X05/concat_explorations/README.md']),
('X06','archived','神经模型、基础模型与误差解释','不同深度方法在各自实际输入和预算下提供什么证据？',
 'MLP原协议、Concat同输入架构对照、TabPFN各自匹配实验分别阅读；官方TabM与自定义style分开命名。',
 'MLP 有低分歧却高误差的案例；Concat/TabPFN探索包含预算与输入限制，不能汇总成跨协议架构排名。',
 '保留远端已有的MLP三组图、原始逐行预测和运行入口；模型优点必须限定于实际对照。','M04/M05',
 ['experiment/appendix/X06/mlp_explain/README.md','experiment/appendix/X06/concat_explorations/README.md','experiment/appendix/T05/README.md']),
('X07','planned','匹配外部种子鲁棒性','两终点的小幅差距能否跨完整训练随机种子保持？',
 'F1/F2 配方冻结，Y/J各42/43/44；已有合格42可复用，43/44不重选超参和权重规则。',
 '当前不存在这组匹配完整重复。子成员四种子平均和 MLP 自身多种子都不能替代。',
 '保留为后续计算任务；需F2年度原点资料、checkpoint、明确GPU时间/显存，再做完整重复。','M08',
 ['docs/planning/appendix-execution-status.md']),
('X08','partial','设施时间/来源与特征敏感性','STATUS=open 是否能证明历史时点已开通？',
 '来源快照坐标去重，与仅排除JRL/CRL-only坐标的特征级对照；没有修改旧模型输入。',
 '34个坐标；最近站点受影响训练20557/测试7153行，复现原审计；距离增量并非RMSE增益。',
 '采用明确快照场景；缺完整开通日期，严格历史重训保持未完成，不认证剩余站点。','M01/M08',
 ['experiment/appendix/X08/evaluation/availability_audit.json','experiment/appendix/X08/evaluation/feature_sensitivity.csv']),
('X09','partial','预测跨度与共同月份边界','近期10个月验证能说明全部查询跨度么？',
 '按原始日期审计全量上下文，保留既有独立长窗参考；F1/F2同2023-12原点L仍须重拟合。',
 '380条查询与最新标签同月，18077条查询超过已配对的10个月范围；并无这些行的真实标签。',
 '只报告样本覆盖与未验证边界；不能从Y/J删月制造新L，不能把重叠窗口当独立测试。','M08',
 ['experiment/appendix/X09/evaluation/query_horizons.csv','experiment/appendix/X09/evaluation/cutoff_checks.json','experiment/appendix/X09/existing_multi_window_reference']),
('X10','archived','公开目标重合与无效排名排除','公开来源匹配到的租金属于预测还是重建？',
 '独立公开快照和已知训练目标重建；覆盖、歧义与估算条件分开。',
 '181.25为目标重建、173.41为估算口径，不能与F1/F2未来预测成绩排名。',
 '保留来源审计，不将公开同目标月份租金回填训练或查询预测。','M00/M08',
 ['experiment/appendix/X10/source_audit/README.md']),
('X11','partial','F2实际组件互补与消融','固定等权融合为何比两个实际成员的RMSE更低？',
 '相同28995行上比较tree7、TabPFN87、等权F2；F1只作独立管线对照，不是F2成员。',
 'F2 RMSE440.95196低于tree7的443.86305及TabPFN的441.95927；TabPFN单独MAE更低。误差相关0.98255。',
 '核验MSE交叉矩恒等式，按月/支持组解释；不以低相关或新权重搜索证明机制；Y未算。','M05/M08',
 ['experiment/appendix/X11/evaluation/member_ablation.csv','experiment/appendix/X11/evaluation/error_moments.json','experiment/appendix/X11/evaluation/group_metrics.csv']),
('X12','partial','配对差值描述性区间','近期F2−F1差值对月份或楼栋构成有多敏感？',
 '冻结J预测；seed42；3个月非循环区块与楼栋簇各3000次，保留簇内记录和抽取重数。',
 'RMSE差-0.88258；月份区间[-1.19267,-0.48572]，楼栋区间[-1.56254,-0.17751]。',
 '以固定预测为条件的描述，不能覆盖训练/开发选模不确定性。Y缺F2不计算，区间跨0也可验收。','M08',
 ['experiment/appendix/X12/evaluation/paired_intervals.csv','experiment/appendix/X12/evaluation/resampling_spec.json']),
('X13','partial','复现层次与环境边界','文件身份、组件重放、模型推理和从零训练分别证明什么？',
 '三份原CSV独立重组；F2保存树/TabPFN输出重建，对照六位小数原候选。',
 'F2的50000行可在半个最末小数单位容差内重组；不是本轮新GPU推理。',
 '保留命令与来源链；新Conda安装和完整重训分别记录待做，不能从pip解析成功推导训练成功。','M09',
 ['experiment/appendix/X13/evaluation/component_replay.json','reproduction/three_submissions/README.md','reproduction/appendix_diagnostics/source_manifest.json'])
]

def write_cards():
    records=[]
    for id,status,purpose,question,control,observation,decision,main,evidence in CARDS:
        directory=ROOT/'experiment/appendix'/id;dest=directory/'README.md'
        old=subprocess.check_output(['git','show','444811aaea2f62ab2801395a8295fda011f441a4:'+dest.relative_to(ROOT).as_posix()],cwd=ROOT).decode('utf-8')
        tail=old[old.index('\n## '):] if '\n## ' in old else ''
        if '\n## 本轮解释图表' in tail:tail=tail[:tail.index('\n## 本轮解释图表')]
        # Preserve established upstream subsection navigation, not generic old cards.
        lines=[f'# {id} {purpose}','',f'当前状态：**{status}**。来源与运行范围见[执行状态](../../../docs/planning/appendix-execution-status.md)。','',
               '| 推理环节 | 内容 |','|---|---|',f'| 问题/假设 | {question} |',f'| 对照与口径 | {control} |',
               f'| 观察 | {observation} |',f'| 取舍依据 | {decision} |',f'| 服务 main | {main}；方法质量60%、解释清晰度30%、来源/复现10% |','',
               '## 证据入口','']
        for path in evidence:
            rel=Path(os.path.relpath(ROOT/path,directory)).as_posix()
            lines.append(f'- [{Path(path).name}]({rel})')
        if id in ['X11','X12']:
            image='complementarity.png' if id=='X11' else 'paired_intervals.png'
            lines += ['', '## 本轮解释图表','',f'![{purpose}](evaluation/{image})','',
                      '图表由 `python -m analysis.appendix_review run --output-dir output/appendix-review/new-run` 从冻结预测重新生成；不训练、不重新选权。']
        dest.write_bytes(('\n'.join(lines)+'\n'+tail).replace('\r\n','\n').encode('utf-8'))
        records.append(dict(id=id,status=status,purpose=purpose,question=question,control=control,observation=observation,
                            decision=decision,main_support=main,evidence=';'.join(evidence)))
    path=ROOT/'docs/planning/appendix-decision-chains.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    (ROOT/'experiment/appendix/decision-catalog.json').write_bytes((json.dumps(records,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def manifest():
    path=ROOT/'review/manifest.json';m=json.loads(path.read_text(encoding='utf-8'))
    current={x['path']:x for x in m['files']}
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    additions=[]
    for pattern in ['analysis/appendix*.py','tests/test_appendix*.py','reproduction/appendix_diagnostics/**/*',
                    'docs/planning/appendix*','docs/guides/appendix*','experiment/appendix/**/*','review/checks/appendix-refresh/**/*']:
        additions.extend(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern) if p.is_file() and '__pycache__' not in p.parts)
    excluded={'review/manifest.json'}
    for relative in sorted(set(tracked+additions)-excluded-{''}):
        p=ROOT/relative
        if not p.is_file():raise ValueError('Missing registered file: '+relative)
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if relative in current:
            row=current[relative]
            if row['sha256']!=h:
                row.setdefault('pre_appendix_refresh_sha256',row['sha256'])
                row['sha256']=h;row['adapted_after_copy']=True
        else:
            current[relative]={'path':relative,'source':None,'source_commit':None,
                'source_kind':'appendix-explanatory-analysis-or-pinned-evidence','sha256':h,'source_sha256':h,'adapted_after_copy':False}
    for id,status,purpose,*_ in CARDS:
        item=next(x for x in m['experiments'] if x['id']==id)
        item.update(status=status,purpose=purpose)
    baseline=json.loads(subprocess.check_output(['git','show','444811aaea2f62ab2801395a8295fda011f441a4:review/manifest.json'],cwd=ROOT))
    original_order=[row['path'] for row in baseline['files']]
    original_rows=[current.pop(relative) for relative in original_order]
    m['files']=original_rows+[current[k] for k in sorted(current)]
    m['f2_status']='J component identity and full saved-prediction reconstruction verified; annual predictions/schema/policy/screenshot gaps remain'
    m['appendix_refresh']={'base_appendix_commit':'444811aaea2f62ab2801395a8295fda011f441a4',
        'main_evidence_commit':'d075c81a4719cad57f8f46d290760324afa14c6c','execution_status':'docs/planning/appendix-execution-status.md',
        'decision_catalog':'experiment/appendix/decision-catalog.json','scientific_models_fitted':0,
        'archived_execution':'review/checks/appendix-refresh/execution.json'}
    path.write_bytes((json.dumps(m,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'files_registered':len(m['files']),'decision_cards':len(CARDS)}))

def audit():
    m=json.loads((ROOT/'review/manifest.json').read_text(encoding='utf-8'))
    failures=[];syntax=0;links=0
    for item in m['files']:
        p=ROOT/item['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            failures.append({'path':item['path'],'reason':'missing/hash'})
        if p.suffix=='.py':
            try:ast.parse(p.read_bytes(),filename=item['path']);syntax+=1
            except (SyntaxError,ValueError) as error:failures.append({'path':item['path'],'reason':str(error)})
    active=[ROOT/'README.md',ROOT/'review/CHECKS.md',ROOT/'experiment/appendix/README.md',
            ROOT/'docs/planning/appendix-execution-status.md',ROOT/'docs/guides/appendix_review.md',
            *[ROOT/'experiment/appendix'/c[0]/'README.md' for c in CARDS]]
    for path in active:
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            links+=1
            if not (path.parent/target.split('#')[0]).exists():failures.append({'path':str(path.relative_to(ROOT)),'reason':'broken navigation','target':target})
    for c in CARDS:
        for e in c[-1]:
            if not (ROOT/e).exists():failures.append({'path':e,'reason':'missing decision evidence'})
    result={'files_checked':len(m['files']),'python_files_parsed':syntax,'active_links_checked':links,'failures':failures,'all_passed':not failures}
    print(json.dumps(result,ensure_ascii=False,indent=2));return int(bool(failures))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['cards','manifest','audit']);a=p.parse_args()
    if a.command=='cards':write_cards();return 0
    if a.command=='manifest':manifest();return 0
    return audit()

if __name__=='__main__':raise SystemExit(main())
