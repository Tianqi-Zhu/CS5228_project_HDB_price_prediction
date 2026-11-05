import unittest
import numpy as np
from analysis.experiments.mlp_temporal_calibration import train,apply


class CalibrationTests(unittest.TestCase):
    def test_bias_correction_uses_only_supplied_fit_labels(self):
        pred=np.array([1000.,2000.,3000.]);labels=pred+100
        model=train(pred,labels,'bias')
        actual=apply(np.array([1500.,2500.]),model,'bias')
        np.testing.assert_allclose(actual,[1600,2600])
    def test_identity_and_affine_are_applied_to_new_predictions(self):
        pred=np.arange(10.)*100+1000;y=pred*1.1+30
        model=train(pred,y,'affine')
        np.testing.assert_allclose(apply(np.array([2000.]),model,'affine'),[2230.])
        np.testing.assert_array_equal(apply(pred,None,'identity'),pred)


if __name__=='__main__':unittest.main()
