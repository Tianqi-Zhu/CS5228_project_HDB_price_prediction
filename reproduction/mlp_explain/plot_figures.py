from pathlib import Path
import hashlib,json,zipfile,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'experiment/appendix/X06/mlp_explain'
OUT=ROOT/'output/figures/mlp_explain'
OUT.mkdir(parents=True,exist_ok=True)
PDF=ROOT/'output/pdf/mlp_explain'
PDF.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':16,'axes.titlesize':18,'axes.labelsize':17,'xtick.labelsize':14,'ytick.labelsize':14,'legend.fontsize':14,'axes.linewidth':1.5,'xtick.major.width':1.3,'ytick.major.width':1.3,'xtick.major.size':6,'ytick.major.size':6,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
BLUE,ORANGE,GREEN='#265d90','#d27a34','#337f74'
inputs=[]
def load(rel):
 p=B/rel;inputs.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return pd.read_csv(p)
mlp=load('validation/uncleaned_mlp_outer.csv')
hybrid=load('validation/uncleaned_hybrid_outer.csv')
clean=load('validation/obvious_low_mlp_outer.csv')
for other in [hybrid,clean]:
 assert np.array_equal(mlp.source_row,other.source_row)
 assert np.array_equal(mlp.actual_sgd,other.actual_sgd)
assert len(mlp)==35158
mlp['error']=mlp.prediction_sgd-mlp.actual_sgd
stats={'validation_period':'2024-04 to 2025-03','fitting_end':'2023-12','n':len(mlp),'training_performed':False}
def figure():
 fig,ax=plt.subplots(1,2,figsize=(16.8,5.0));fig.subplots_adjust(left=.065,right=.91,bottom=.22,top=.84,wspace=.38)
 for a in ax:a.spines[['top','right']].set_visible(False);a.tick_params(pad=6)
 return fig,ax
def save(fig,name,note):
 fig.text(.065,.015,textwrap.fill(note, width=195),fontsize=9,color='#444444',va='bottom')
 for ext in ['png','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)
 svg=OUT/f'{name}.svg'
 svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 fig.savefig(PDF/f'{name}.pdf')
 plt.close(fig)
# Figure 1: reference-style temporal curves and hexagonal density.
fig,(a,b)=figure()
for d,label,color in [(mlp,'MLP ensemble',BLUE),(clean,'MLP + low-rent filtering',ORANGE),(hybrid,'MLP + TabM equal blend',GREEN)]:
 monthly=d.assign(sq=(d.prediction_sgd-d.actual_sgd)**2).groupby('month').sq.mean().pow(.5)
 a.plot(range(12),monthly.values,'o-',color=color,lw=2.5,ms=6,label=label)
 monthly.rename(label).to_csv(OUT/(label.split(' ')[0]+('_clean' if d is clean else '_hybrid' if d is hybrid else '')+'_monthly.csv'))
a.set_xticks(range(12),['Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar'],rotation=45)
a.set_ylabel('RMSE (SGD)');a.set_title('(a) Monthly development error',pad=14)
a.legend(frameon=False,loc='upper right',fontsize=12);a.set_ylim(410,550)
h=b.hexbin(mlp.actual_sgd,mlp.prediction_sgd,gridsize=55,extent=(0,8000,0,8000),bins='log',mincnt=1,cmap='Blues',linewidths=0)
b.plot([0,8000],[0,8000],'--',color=ORANGE,lw=2)
b.set(xlim=(0,8000),ylim=(0,8000),xlabel='Actual rent (SGD)',ylabel='Predicted rent (SGD)')
b.set_title('(b) MLP prediction versus observation',pad=14)
cb=fig.colorbar(h,ax=b,pad=.035,fraction=.045);cb.set_label('Transactions / hexagon',labelpad=10)
rmse=np.sqrt(np.mean(mlp.error**2));stats['mlp_rmse']=float(rmse)
save(fig,'01_mlp_prediction_diagnostics','Development: Apr 2024-Mar 2025; n = 35,158. Models fit through Dec 2023. Density plot: unfiltered MLP ensemble; dashed line: perfect prediction.')
# Figure 2: quantify concentration and overconfident ensemble agreement.
fig,(a,b)=figure()
sq=np.sort(mlp.error.to_numpy()**2)[::-1];x=np.arange(1,len(sq)+1)/len(sq)*100;y=np.cumsum(sq)/sq.sum()*100
n5=int(np.ceil(.05*len(sq)));top5=y[n5-1]
a.plot(np.r_[0,x],np.r_[0,y],color=BLUE,lw=2.5);a.plot([0,100],[0,100],'--',color='0.6',lw=1.4)
a.scatter([5],[top5],color=ORANGE,s=65,zorder=4)
a.annotate(f'Worst 5%: {top5:.1f}% of squared error',xy=(5,top5),xytext=(17,30),fontsize=14,arrowprops={'arrowstyle':'->','color':ORANGE,'lw':1.4})
a.set(xlim=(0,100),ylim=(0,100),xlabel='Transactions, ranked by absolute error (%)',ylabel='Cumulative squared error (%)')
a.set_title('(a) MLP error concentration',pad=14)
std=mlp.ensemble_std_sgd.to_numpy();err=np.abs(mlp.error.to_numpy());rho=float(spearmanr(std,err).statistic)
# Full data displayed: bounds rounded above observed maxima.
xmax=np.ceil(std.max()/50)*50;ymax=np.ceil(err.max()/500)*500
h=b.hexbin(std,err,gridsize=55,extent=(0,xmax,0,ymax),bins='log',mincnt=1,cmap='Blues',linewidths=0)
b.axvline(100,color=ORANGE,lw=1.7,ls='--');b.axhline(1000,color=GREEN,lw=1.7,ls='--')
b.set(xlim=(0,xmax),ylim=(0,ymax),xlabel='MLP ensemble disagreement (SGD)',ylabel='Absolute prediction error (SGD)')
b.set_title('(b) Low disagreement, large errors',pad=14)
large=err>=1000;hidden=large&(std<100)
b.text(.97,.96,f'Spearman correlation: {rho:.3f}\n{hidden.sum():,} / {large.sum():,} large errors\nhave disagreement < 100',ha='right',va='top',transform=b.transAxes,fontsize=12,bbox={'facecolor':'white','edgecolor':'none','alpha':.9})
cb=fig.colorbar(h,ax=b,pad=.035,fraction=.045);cb.set_label('Transactions / hexagon',labelpad=10)
stats.update(mlp_worst5_sse_share=float(top5/100),mlp_disagreement_error_spearman=rho,mlp_errors_at_least_1000=int(large.sum()),mlp_large_errors_std_below100=int(hidden.sum()))
save(fig,'02_mlp_error_and_uncertainty','Unfiltered MLP ensemble, n = 35,158. Large error: absolute error >= 1,000 SGD. Disagreement is the standard deviation across three independent networks.')
# Figure 3: empirical local restriction + matched cleaning sensitivity.
fig,(a,b)=figure()
case=load('uncertainty_diagnostics/indistinguishable_inputs_proof_case.csv')
actual=case.MONTHLY_RENT.to_numpy();prediction=case.prediction_sgd.to_numpy();optimal=actual.mean()
a.scatter(np.arange(3),actual,color=BLUE,s=90,label='Observed rent',zorder=3)
a.axhline(optimal,color=GREEN,lw=2.2,ls='--',label=f'Best constant: {optimal:,.0f}')
a.plot(np.arange(3),prediction,color=ORANGE,lw=2,marker='s',ms=6,label=f'Current blend: {prediction[0]:,.0f}')
for i,(v,p) in enumerate(zip(actual,prediction)):
 a.vlines(i,min(v,p),max(v,p),color='0.65',lw=1.3);a.annotate(f'{v:,.0f}',(i,v),xytext=(0,12 if v>=5000 else -22),textcoords='offset points',ha='center',fontsize=14)
a.set_xticks(range(3),['Record 1','Record 2','Record 3']);a.set(xlim=(-.45,2.45),ylim=(3500,6600),ylabel='Monthly rent (SGD)')
a.set_title('(a) Identical inputs, different rents',pad=14)
a.legend(frameon=False,loc='upper left',fontsize=12)
a.text(.98,.045,'Group RMSE: current 816.99; best 816.50',transform=a.transAxes,ha='right',fontsize=12)
metrics=load('cleaning/comparison_metrics.csv')
policies=['uncleaned_baseline','obvious_low','highest_only'];labels=['Unfiltered','Low-rent\nfiltering','Highest-only\nfiltering'];x=np.arange(3);width=.24
for i,(name,label,color) in enumerate([('mlp','MLP',BLUE),('tabm','TabM',ORANGE),('hybrid','Equal blend',GREEN)]):
 vals=[float(metrics[(metrics.policy==p)&(metrics.model==name)].rmse_sgd.iloc[0]) for p in policies]
 b.bar(x+(i-1)*width,vals,width,color=color,label=label)
 for xx,v in zip(x+(i-1)*width,vals):b.text(xx,v+.9,f'{v:.1f}',ha='center',fontsize=11)
b.set_xticks(x,labels);b.set_ylabel('RMSE (SGD)');b.set_ylim(0,560);b.legend(frameon=False,loc='upper left',ncol=3,fontsize=12)
b.set_title('(b) Cleaning sensitivity',pad=14)
save(fig,'03_identical_inputs_and_cleaning','Panel (a): archived MLP + TabM blend, three Nov 2024 records at 1E Cantonment Road. Local case only. Panel (b): all 35,158 validation rows retained; only fitting rows filtered.')
(OUT/'figure_statistics.json').write_text(json.dumps(stats,indent=2)+'\n')
(OUT/'sources.json').write_text(json.dumps(inputs,indent=2)+'\n')
(OUT/'README.txt').write_text('MLP explanatory experiment figures\n\nLayout follows error_analysis.pdf supplied by the user: white background, two panels, blue/orange/teal palette, line charts and logarithmic hexbin density.\n\n01: annual development monthly RMSE; MLP density plot.\n02: MLP squared-error concentration and disagreement/error relation.\n03: archived blend identical-input case; matched cleaning experiment across MLP, TabM and equal blend.\n\nPNG: 5040 x 1500 pixels at 300 dpi. SVG and PDF: vector text and graphics.\nAll figures use archived predictions; no model fitting performed. Baseline fitting ends Dec 2023; Jan-Mar 2024 are interval calibration; Apr 2024-Mar 2025 are development validation. Repeatedly inspected development results are not independent future testing.\nThe local best-constant case is not a global or population RMSE lower bound. Ensemble disagreement is not a probability of correctness.\nSource paths and SHA256 hashes are in sources.json. Reproduce from the appendix repository root with reproduction/mlp_explain/plot_figures.py.\n')
with zipfile.ZipFile(OUT/'mlp_explanatory_figures.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
 for p in sorted(PDF.glob('*.pdf')):z.write(p,'pdf/'+p.name)
print(json.dumps(stats,indent=2))
