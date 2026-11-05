"""Check checkpoint identity and exact recovery of stochastic optimization."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.cross_attention_matched467_v12 import run, verify


class RecoveryTests(unittest.TestCase):
    def test_checkpoint_identity_rejects_other_arm_or_registration(self):
        validate = getattr(verify, "validate_checkpoint", None)
        self.assertIsNotNone(validate, "checkpoint provenance must be checked before replay")
        reg = {"fingerprint": "registered", "config": {"seed": 42}}
        ck = {"arm": "summary", **reg, "epoch": 23, "steps": 3979}
        validate(ck, reg, "summary")
        for key, value in (("arm", "concat"), ("fingerprint", "other"),
                           ("config", {"seed": 43}), ("epoch", 22), ("steps", 3806)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate({**ck, key: value}, reg, "summary")

    def test_saved_epoch_resumes_identical_dropout_adamw_trajectory(self):
        torch.set_num_threads(1)
        inputs = torch.linspace(-1, 1, 64).reshape(16, 4)
        target = inputs.square().mean(dim=1)

        def fresh():
            run.set_seed(42)
            model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(),
                                        torch.nn.Dropout(.2), torch.nn.Linear(8, 1))
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            return model, opt, run.scheduler(opt)

        def epoch(model, opt, sched, number):
            order = np.arange(16)
            np.random.default_rng(run.batch_seed(number)).shuffle(order)
            model.train()
            for start in (0, 8):
                idx = order[start:start + 8]
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(model(inputs[idx]).squeeze(-1), target[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
            sched.step()

        whole, opt, sched = fresh()
        for number in range(1, 4):
            epoch(whole, opt, sched, number)
        expected = copy.deepcopy(whole.state_dict())
        expected_rng = torch.get_rng_state().clone()
        expected_lr = sched.get_last_lr()

        model, opt, sched = fresh()
        epoch(model, opt, sched, 1)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "checkpoint.pt"
            run.save(file, {"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(), "rng": run.rng_get(), "epoch": 1})
            resumed, resumed_opt, resumed_sched = fresh()
            saved = torch.load(file, map_location="cpu", weights_only=False)
            resumed.load_state_dict(saved["state_dict"])
            resumed_opt.load_state_dict(saved["optimizer"])
            resumed_sched.load_state_dict(saved["scheduler"])
            run.rng_set(saved["rng"])
            for number in range(saved["epoch"] + 1, 4):
                epoch(resumed, resumed_opt, resumed_sched, number)
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        self.assertEqual(expected_lr, resumed_sched.get_last_lr())


if __name__ == "__main__":
    unittest.main()
