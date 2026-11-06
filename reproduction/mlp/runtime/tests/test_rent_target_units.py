import unittest
import numpy as np


class TargetUnitTests(unittest.TestCase):
    def test_weighted_transformed_mse_equals_sgd_mse_up_to_fixed_scale(self):
        y=np.array([2000.,3000.,4000.]);area=np.array([50.,80.,100.])
        target=y/area;mu=target.mean();sd=target.std()
        z=(target-mu)/sd;prediction=z+np.array([.1,-.2,.4])
        loss=np.mean((prediction-z)**2*(area/area.mean())**2)
        rent_prediction=(prediction*sd+mu)*area
        mse=np.mean((rent_prediction-y)**2)
        self.assertAlmostEqual(loss,mse/(sd*area.mean())**2)


if __name__=='__main__':unittest.main()
