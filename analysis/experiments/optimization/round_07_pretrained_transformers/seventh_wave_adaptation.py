"""Small supervised adaptation of the author TabDPT inference backbone.

This project trainer is not the author's original pretraining/fine-tuning recipe.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from analysis.experiments.frontier_search import score
from analysis.experiments.sixth_wave_deep import sample_nca_context


def set_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in list(model.transformer_encoder[-2:])+list(model.reg_y_encoders[-2:])+[model.head]:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    # SDPA's scalar `scale` is an inference hyperparameter, not a differentiable input.
    for name,parameter in model.named_parameters():
        if name.endswith('kappa'):
            parameter.requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def adapt(regressor,target,query,bundle,offset,spec,out,epochs=None):
    from tabdpt.utils import pad_x
    model=regressor.model;model.eval()
    parameters=set_trainable(model)
    optimizer=torch.optim.AdamW(parameters,lr=3e-5,weight_decay=.01)
    source=torch.as_tensor(regressor.X_train,dtype=torch.float32,device='cuda')
    if source.shape[1]>regressor.max_features:
        raise ValueError('Adaptation assumes no PCA feature reduction')
    target=torch.as_tensor(target,dtype=torch.float32,device='cuda')
    target_mean=target.mean();target_std=target.std()+1e-6
    normalized=(target-target_mean)/target_std
    history=[];best_state=None;best_epoch=0;stale=0;best=float('inf')
    # An exact matched frozen control uses the same input, BF16, views, and context.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        initial=np.asarray(regressor.predict(query,n_ensembles=2,context_size=2048,batch_size=2048,seed=42),float)+offset
    initial_scores=score(bundle['y_val'],initial)
    best=initial_scores['rmse_sgd']
    if epochs is None:
        best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    for epoch in range(1,(epochs if epochs is not None else 8)+1):
        order=torch.randperm(len(target),device='cuda')
        losses=[]
        for begin in range(0,len(order),256):
            rows=order[begin:begin+256]
            context=sample_nca_context(len(target),rows,512)
            indices=torch.cat([context,rows])
            permutation=torch.randperm(source.shape[1],device='cuda')
            x=pad_x(source[indices][:,permutation][None],regressor.max_features)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(x_src=x,y_src=normalized[context][None],is_cls=False)
                logits=output[:,0,model.n_out:].float()
                expected=regressor._expectation_from_regression_logits(logits)
                loss=(expected-normalized[rows]).square().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            losses.append(float(loss.detach()))
        if epochs is None:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=np.asarray(regressor.predict(query,n_ensembles=2,context_size=2048,batch_size=2048,seed=42),float)+offset
            rmse=score(bundle['y_val'],prediction)['rmse_sgd']
            history.append({'epoch':epoch,'normalized_train_mse':float(np.mean(losses)),'rmse_sgd':rmse})
            if rmse<best-.02:
                best=rmse;best_epoch=epoch;stale=0
                best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            else:
                stale+=1
            print(f'  TabDPT adaptation epoch {epoch}: RMSE {rmse:.3f}, best {best:.3f}',flush=True)
            if stale>=3:
                break
        else:
            print(f'  fixed TabDPT adaptation epoch {epoch}/{epochs}',flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {'training_epochs':int(best_epoch if epochs is None else epochs),'training_history':history,
        'matching_frozen_control':initial_scores,'trainable_parameters':sum(parameter.numel() for parameter in parameters),
        'supervised_rows_per_epoch':len(target),'training_context':512,'training_query_batch':256,
        'training_note':'Project supervised adapter; head and final two attention/FFN blocks plus their y encoders updated. Attention scale kappa fixed. Each fitting row is a query once per epoch; its own label is excluded from context. Original author source unchanged.'}


def replay_saved(stage):
    """Rebuild a fitted adapter from recipe + learned weights, without retraining."""
    import gc
    import joblib
    import pandas as pd
    from tabdpt import TabDPTRegressor
    from analysis.experiments.third_wave_training import make_data
    from analysis.experiments.seventh_wave_transformers import OUT,ART,ROOT,frame_inputs
    directory=ART/stage/'tabdpt13_adapt_residual'
    recipe=joblib.load(directory/'recipe.joblib');bundle=make_data(stage)
    indices=recipe['context_indices'];spec=recipe['spec']
    x,q,target,offset=frame_inputs(bundle,spec,indices)
    assert list(x)==recipe['columns']
    cats,nums=recipe['encoding'].transform(x.iloc[indices]);qc,qn=recipe['encoding'].transform(q)
    numbers=np.concatenate([nums,cats.astype(np.float32)],axis=1)
    query=np.concatenate([qn,qc.astype(np.float32)],axis=1)
    manifest=json.loads((OUT/'checkpoint_manifest.json').read_text())
    model=TabDPTRegressor(model_weight_path=str(ROOT/manifest['tabdpt13']['path']),device='cuda',
        compile=False,use_flash=False,context_reduction='subsample',verbose=False)
    model.fit(numbers,target[indices].astype(np.float32))
    model.model.load_state_dict(torch.load(directory/'adapted_state.pt',map_location='cpu',weights_only=True))
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=np.asarray(model.predict(query,n_ensembles=2,context_size=2048,batch_size=2048,seed=42),float)+offset
    expected=pd.read_csv(directory/'predictions.csv').sort_values('source_row')
    np.testing.assert_array_equal(expected.source_row,bundle['row_val'])
    difference=float(np.max(np.abs(prediction-expected.prediction.to_numpy(float))))
    np.testing.assert_allclose(prediction,expected.prediction,atol=1e-5,rtol=0)
    del model;gc.collect();torch.cuda.empty_cache()
    return {'stage':stage,'rows':len(prediction),'max_difference_sgd':difference}
