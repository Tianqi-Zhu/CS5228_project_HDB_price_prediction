import numpy as np
import pytest
from types import SimpleNamespace
from experiments.tabfm_finetune_v13.verify_raw_selection import metrics,check_identity


def test_metrics_and_exact_labels_with_reordered_ids():
    hold=SimpleNamespace(row_id=np.array([7,8]),actual_sgd=np.array([100.,200.]))
    idx=check_identity(np.array([8,7]),np.array([200.,100.]),hold)
    assert idx.tolist()==[1,0]
    assert metrics([201.,98.],[200.,100.])['rmse_sgd']==pytest.approx(np.sqrt(2.5))
    with pytest.raises(ValueError): check_identity(np.array([8,7]),np.array([200.,101.]),hold)
    with pytest.raises(ValueError): check_identity(np.array([8,8]),np.array([200.,200.]),hold)
    with pytest.raises(ValueError): metrics([np.nan],[1.])
