"""Default submission and reload outputs stay in their documented categories."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from analysis.pipeline import workflow
from analysis.pipeline.training import HDBTrainer, TrainerConfig


class DefaultOutputTests(unittest.TestCase):
    def test_default_submit_and_saved_prediction_use_baseline_directories(self):
        train = pd.DataFrame({
            'RENT_APPROVAL_DATE': ['2024-01', '2024-02', '2024-03', '2024-04'],
            'FLAT_TYPE': ['3-room'] * 4,
            'FLAT_MODEL': ['Improved'] * 4,
            'TOWN': ['ANG MO KIO'] * 4,
            'FLOOR_AREA_SQM': [67., 68., 69., 70.],
            'LEASE_COMMENCE_DATE': [1980] * 4,
            'MONTHLY_RENT': [1800., 1900., 2000., 2100.],
        })
        test = train.iloc[:3].drop(columns='MONTHLY_RENT').copy()
        template = pd.DataFrame({'Id': [0, 1, 2], 'Predicted': [0., 0., 0.]})
        model = DummyRegressor(strategy='constant', constant=2000.).fit(np.zeros((4, 7)), train.MONTHLY_RENT)
        def load(path, **kwargs):
            return train.copy() if Path(path).name == 'train.csv' else test.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(workflow, 'DATA', root / 'data'), \
                 patch.object(workflow, 'MODELS', root / 'models'), \
                 patch.object(workflow, 'SUBMISSIONS', root / 'submissions'), \
                 patch.object(workflow, 'SOURCES', []), \
                 patch.object(workflow, 'load_main', side_effect=load), \
                 patch.object(workflow.pd, 'read_csv', return_value=template.copy()), \
                 patch.object(HDBTrainer, 'fit_frame', return_value=model):
                result = workflow.run_submission(TrainerConfig())
                saved = Path(result['model_path'])
                self.assertEqual(saved.parent, root / 'models/baseline_submissions')
                self.assertTrue(saved.is_file())
                replay = workflow.predict_saved(saved)
            expected = root / 'submissions/baselines/lightgbm_C_property_time_town.csv'
            reloaded = root / 'submissions/baselines/saved_model.csv'
            self.assertTrue(expected.is_file())
            self.assertTrue(reloaded.is_file())
            self.assertFalse((root / 'submissions/saved_model.csv').exists())
            pd.testing.assert_frame_equal(pd.read_csv(expected), pd.read_csv(reloaded))
            self.assertEqual(replay.Id.tolist(), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
