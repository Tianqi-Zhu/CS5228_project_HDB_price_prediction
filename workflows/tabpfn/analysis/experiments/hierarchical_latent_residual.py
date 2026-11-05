"""Sparse Gaussian MAP estimates of crossed building and submarket effects."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr


def latest_available_residuals(cases, origin):
    origin=str(pd.Period(origin,'M'))
    if cases.empty: return cases.copy()
    if (cases.origin>cases.month).any(): raise ValueError('Residual prediction precedes its own context cutoff incorrectly')
    eligible=cases.loc[(cases.month<origin)&(cases.origin<origin)].copy()
    eligible=eligible.sort_values(['row_id','origin'],kind='stable').drop_duplicates('row_id',keep='last')
    eligible=eligible.sort_values(['month','row_id']).reset_index(drop=True)
    if len(eligible):
        if not np.isfinite(eligible.weight).all() or (eligible.weight<=0).any(): raise ValueError('Invalid sampling weights')
        eligible['fit_weight']=eligible.weight/eligible.weight.mean()
    return eligible


def apply_correction(base, correction, alpha):
    base=np.asarray(base,float); correction=np.asarray(correction,float)
    if base.shape!=correction.shape or not np.isfinite(correction).all() or not 0<=alpha<=1:
        raise ValueError('Invalid correction shape, values or strength')
    return base.copy() if alpha==0 else base+alpha*correction


def weighted_metrics(actual,predicted,weights=None):
    y,p=np.asarray(actual,float),np.asarray(predicted,float)
    w=np.ones(len(y)) if weights is None else np.asarray(weights,float)
    if len(y)==0 or not np.isfinite(p).all() or (w<=0).any(): raise ValueError('Invalid evaluation rows')
    error=p-y
    return dict(rmse_sgd=float(np.sqrt(np.average(error**2,weights=w))),
                mae_sgd=float(np.average(np.abs(error),weights=w)),bias_sgd=float(np.average(error,weights=w)))


class HierarchicalResidual:
    """r = mu + building + town/type, Gaussian priors centered at zero.

    Penalties are ratios sigma^2/tau^2 in effective observed-case units.
    Weights are normalized to the actual number of residual cases so population
    sampling weights cannot manufacture thousands of independent observations.
    """
    def __init__(self,block_penalty=100.,group_penalty=100.):
        if block_penalty<=0 or group_penalty<=0: raise ValueError('Positive Gaussian prior precision required')
        self.block_penalty=float(block_penalty); self.group_penalty=float(group_penalty)

    def _system(self,frame,origin):
        origin=pd.Period(origin,'M')
        months=pd.PeriodIndex(frame.month,freq='M')
        if not len(frame) or months.max()>=origin: raise ValueError('Latent fitting labels must precede forecast origin')
        if frame.row_id.duplicated().any(): raise ValueError('Each physical source observation may enter fitting once')
        if not np.isfinite(frame.residual).all(): raise ValueError('Nonfinite historical residual')
        buildings=pd.Index(sorted(frame.building.astype(str).unique()))
        groups=pd.Index(sorted(frame.submarket.astype(str).unique()))
        bi=buildings.get_indexer(frame.building); gi=groups.get_indexer(frame.submarket)
        n,nb,ng=len(frame),len(buildings),len(groups)
        columns=np.column_stack([np.zeros(n,int),1+bi,1+nb+gi])
        x=sparse.csr_matrix((np.ones(3*n),(np.repeat(np.arange(n),3),columns.ravel())),shape=(n,1+nb+ng))
        precision=np.r_[0.,np.full(nb,self.block_penalty),np.full(ng,self.group_penalty)]
        penalty=sparse.diags(np.sqrt(precision),format='csr')
        w=np.asarray(frame.get('fit_weight',frame.get('weight',np.ones(n))),float)
        if not np.isfinite(w).all() or (w<=0).any(): raise ValueError('Invalid fitting weights')
        w=w/w.mean()
        return x,penalty,w,dict(buildings=buildings,groups=groups,bi=bi,gi=gi,origin=origin,
            n=n,nb=nb,ng=ng,months=months)

    def fit(self,frame,origin):
        started=time.perf_counter(); x,penalty,w,meta=self._system(frame,origin)
        a=sparse.vstack([x.multiply(np.sqrt(w)[:,None]),penalty],format='csr')
        rhs=np.r_[np.sqrt(w)*frame.residual.to_numpy(float),np.zeros(penalty.shape[0])]
        solution=lsmr(a,rhs,atol=1e-10,btol=1e-10,maxiter=4000)
        if solution[1] not in [0,1,2,4,5]: raise RuntimeError(f'Sparse MAP solver did not converge: {solution[1]}')
        self.coef_=solution[0]
        self.buildings_=meta['buildings']; self.groups_=meta['groups']; self.origin_=str(meta['origin'])
        self.n_buildings_=meta['nb']; self.n_groups_=meta['ng']
        self.building_counts_=np.bincount(meta['bi'],minlength=meta['nb'])
        self.group_counts_=np.bincount(meta['gi'],minlength=meta['ng'])
        self.building_weights_=np.bincount(meta['bi'],weights=w,minlength=meta['nb'])
        self.group_weights_=np.bincount(meta['gi'],weights=w,minlength=meta['ng'])
        self.last_fitting_month_=str(meta['months'].max())
        self.fit_rows_=len(frame); self.solver_iterations_=int(solution[2]); self.solver_stop_=int(solution[1])
        self.system_shape_=a.shape; self.system_nnz_=a.nnz
        self.training_rmse_=float(np.sqrt(np.average((np.asarray(x@self.coef_).ravel()-frame.residual.to_numpy())**2,weights=w)))
        self.fit_seconds_=time.perf_counter()-started
        assert np.isfinite(self.coef_).all()
        return self

    def predict_components(self,query):
        periods=pd.PeriodIndex(query.month,freq='M')
        if (periods<pd.Period(self.origin_,'M')).any(): raise ValueError('Cannot forecast back into fitted target months')
        bi=self.buildings_.get_indexer(query.building); gi=self.groups_.get_indexer(query.submarket)
        known_b=bi>=0; known_g=gi>=0
        a=np.zeros(len(query)); c=np.zeros(len(query)); count=np.zeros(len(query),int)
        # Completely unseen submarket falls back to global mu, per protocol.
        usable_b=known_b&known_g
        a[usable_b]=self.coef_[1+bi[usable_b]]
        c[known_g]=self.coef_[1+self.n_buildings_+gi[known_g]]
        count[known_b]=self.building_counts_[bi[known_b]]
        result=pd.DataFrame(dict(global_effect=np.full(len(query),self.coef_[0]),building_effect=a,
            submarket_effect=c,dynamic_effect=np.zeros(len(query)),residual_building_count=count,
            known_building=known_b,known_submarket=known_g),index=query.index)
        result['total_correction']=result.global_effect+result.building_effect+result.submarket_effect
        return result


class DynamicHierarchicalResidual(HierarchicalResidual):
    """Joint static/dynamic Gaussian MAP with a shared stationary AR(1) prior.

    The innovation precision multiplies (d_t-rho*d_{t-1})^2. Initial precision
    is lambda_state*(1-rho^2), corresponding to the stationary initial prior.
    Only historical states through origin-1 are estimated. Future states are
    forecasts, never extra fitted evaluation-month parameters.
    """
    def __init__(self,block_penalty=100.,group_penalty=100.,rho=.5,state_penalty=100.):
        super().__init__(block_penalty,group_penalty)
        if not abs(rho)<1 or state_penalty<=0: raise ValueError('Stable shared AR persistence and positive innovation precision required')
        self.rho=float(rho); self.state_penalty=float(state_penalty)

    def _system(self,frame,origin):
        x,regularizer,w,meta=super()._system(frame,origin)
        first=meta['months'].min(); nt=(meta['origin']-1).ordinal-first.ordinal+1
        n,ng=meta['n'],meta['ng']; start=x.shape[1]; nd=ng*nt
        t=meta['months'].asi8-first.ordinal
        state_x=sparse.csr_matrix((np.ones(n),(np.arange(n),meta['gi']*nt+t)),shape=(n,nd))
        x=sparse.hstack([x,state_x],format='csr')
        top=sparse.hstack([regularizer,sparse.csr_matrix((regularizer.shape[0],nd))],format='csr')
        rr=[]; cc=[]; vv=[]; root=np.sqrt(self.state_penalty)
        for group in range(ng):
            for month in range(nt):
                row=group*nt+month; col=start+row
                rr.append(row); cc.append(col); vv.append(root*np.sqrt(1-self.rho**2) if month==0 else root)
                if month:
                    rr.append(row); cc.append(col-1); vv.append(-root*self.rho)
        temporal=sparse.csr_matrix((vv,(rr,cc)),shape=(nd,start+nd))
        meta.update(first=first,nt=nt)
        return x,sparse.vstack([top,temporal],format='csr'),w,meta

    def fit(self,frame,origin):
        super().fit(frame,origin)
        self.state_first_month_=str(pd.PeriodIndex(frame.month,freq='M').min())
        self.state_last_month_=str(pd.Period(origin,'M')-1)
        self.state_nmonths_=pd.Period(self.state_last_month_,'M').ordinal-pd.Period(self.state_first_month_,'M').ordinal+1
        offset=1+self.n_buildings_+self.n_groups_
        self.states_=self.coef_[offset:].reshape(self.n_groups_,self.state_nmonths_)
        return self

    def predict_components(self,query):
        result=super().predict_components(query)
        gi=self.groups_.get_indexer(query.submarket); known=gi>=0
        h=pd.PeriodIndex(query.month,freq='M').asi8-pd.Period(self.state_last_month_,'M').ordinal
        dynamic=np.zeros(len(query))
        dynamic[known]=self.states_[gi[known],-1]*self.rho**h[known]
        result['dynamic_effect']=dynamic
        result['total_correction']=result.global_effect+result.building_effect+result.submarket_effect+result.dynamic_effect
        return result
