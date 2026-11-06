"""Recompute diagnostic claims from frozen row-level predictions, without fitting."""
import argparse,csv,json,math
from pathlib import Path
from analysis.experiments.verify_mlp_delivery import verify,rows
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'experiment/appendix/X06/mlp_explain'
def score(data):
    errors=[float(r['prediction_sgd'])-float(r['actual_sgd']) for r in data]
    n=len(errors);sse=sum(e*e for e in errors)
    return dict(n=n,rmse=math.sqrt(sse/n),mae=sum(map(abs,errors))/n,bias=sum(errors)/n,sse=sse,
        coverage90=sum(float(r['pi90_lower_sgd'])<=float(r['actual_sgd'])<=float(r['pi90_upper_sgd']) for r in data)/n)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve()
    if not out.is_relative_to(ROOT/'output'):raise ValueError('Use a fresh directory under repository output/')
    if out.exists():raise FileExistsError(out)
    validation=verify(BUNDLE,ROOT)
    data=list(rows(BUNDLE/'validation/uncleaned_hybrid_outer.csv'));overall=score(data)
    sq=lambda r:(float(r['prediction_sgd'])-float(r['actual_sgd']))**2
    top=sorted(data,key=sq,reverse=True)[:math.ceil(.05*len(data))]
    concentration=sum(map(sq,top))/overall['sse']
    archived=json.loads((BUNDLE/'uncertainty_diagnostics/summary.json').read_text())
    assert abs(overall['rmse']-archived['rmse'])<1e-8
    assert abs(concentration-archived['top_5pct_sse_share'])<1e-8
    groups=[]
    for label,select in [('<=1900',lambda y:y<=1900),('1900..3800',lambda y:1900<y<=3800),('>3800',lambda y:y>3800)]:
        result=score([r for r in data if select(float(r['actual_sgd']))])
        result.update(group=label,sse_share=result['sse']/overall['sse']);groups.append(result)
        reference=next(r for r in archived['rent_bands'] if r['band']==label)
        assert abs(result['sse_share']-reference['share_sse'])<1e-8
    monthly=[dict(month=m,**score([r for r in data if r['month']==m])) for m in sorted({r['month'] for r in data})]
    case=list(rows(BUNDLE/'uncertainty_diagnostics/indistinguishable_inputs_proof_case.csv'))
    y=[float(r['MONTHLY_RENT']) for r in case];mean=sum(y)/len(y)
    minimum=math.sqrt(sum((v-mean)**2 for v in y)/len(y))
    proof=json.loads((BUNDLE/'uncertainty_diagnostics/indistinguishable_inputs_proof.json').read_text())
    assert abs(minimum-proof['minimum_group_rmse'])<1e-8
    risk=list(rows(BUNDLE/'validation/risk_corrected_hybrid_outer.csv'))
    large=[r for r in risk if abs(float(r['prediction_sgd'])-float(r['actual_sgd']))>=1000]
    confident=sum(float(r['ensemble_std_sgd'])<100 for r in large)
    assert (len(large),confident)==(1422,1182)
    out.mkdir(parents=True)
    for name,records in [('rent_bands.csv',groups),('monthly.csv',monthly)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");w.writeheader();w.writerows(records)
    result={'training_performed':False,'archive_verification':validation,'uncleaned_hybrid':overall,
        'worst_5pct_sse_share':concentration,'risk_corrected_large_errors':len(large),
        'risk_corrected_large_errors_with_std_below_100':confident,
        'case_optimal_constant':mean,'case_minimum_rmse':minimum,
        'limits':'Development-period, retrospective diagnostics. Case bound is local only. Encoded equality is inherited evidence; no checkpoints are replayed here. True-rent groups are unavailable at inference.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
