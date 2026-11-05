import unittest
import torch
from experiments.macro_encoders_v3.encoders import DBSEncoder

try:
    from experiments.cross_attention_v11.model import sequence_states
except ImportError:
    sequence_states = None

class SequenceTests(unittest.TestCase):
    def test_sequence_implementation_available(self):
        self.assertIsNotNone(sequence_states)

    def test_last_valid_matches_summary_and_padding_is_ignored(self):
        torch.manual_seed(42)
        encoder = DBSEncoder()
        x = torch.randn(2, 8, 7)
        mask = torch.tensor([[0, 1, 0, 1, 1, 0, 1, 1], [0]*8])
        tokens, valid = sequence_states(encoder, x, mask)
        summary, _ = encoder(x, mask)
        self.assertTrue(torch.allclose(tokens[0, 7], summary[0], atol=1e-6))
        self.assertTrue(torch.allclose(tokens[1, 0], summary[1], atol=1e-6))
        changed = x.clone(); changed[~mask.bool()] = 999
        other, _ = sequence_states(encoder, changed, mask)
        self.assertTrue(torch.allclose(tokens[valid], other[valid]))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertTrue(valid.any(dim=1).all())
        tokens[valid].sum().backward()
        self.assertGreater(encoder.lstm.weight_ih_l0.grad.abs().sum().item(), 0)


class OrderingTests(unittest.TestCase):
    def test_order_changes_hidden_and_lag_encodes_real_positions(self):
        from experiments.cross_attention_v11.model import lag_encoding
        torch.manual_seed(7)
        encoder = DBSEncoder()
        x = torch.randn(1, 8, 7)
        mask = torch.ones(1, 8)
        forward, _ = sequence_states(encoder, x, mask)
        reverse, _ = sequence_states(encoder, x.flip(1), mask)
        self.assertFalse(torch.allclose(forward[:, -1], reverse[:, -1]))
        lag = lag_encoding(8, 48, x.device, x.dtype)
        self.assertFalse(torch.equal(lag[0], lag[-1]))

    def test_masked_memory_does_not_change_attention(self):
        torch.manual_seed(3)
        layer = torch.nn.MultiheadAttention(48, 4, batch_first=True).eval()
        q = torch.randn(2, 3, 48)
        memory = torch.randn(2, 7, 48)
        invalid = torch.tensor([[False, True, False, True, False, True, False]] * 2)
        original = layer(q, memory, memory, key_padding_mask=invalid, need_weights=False)[0]
        memory[invalid] = 1000
        changed = layer(q, memory, memory, key_padding_mask=invalid, need_weights=False)[0]
        self.assertTrue(torch.allclose(original, changed))

class ModelTests(unittest.TestCase):
    def test_cross_model_backward_and_checkpoint_reload(self):
        import io
        from experiments.feature_transformer_v4.models import FeatureTransformer
        from experiments.feature_transformer_v4.macro_v4 import MacroOnFeatureTransformer
        from experiments.cross_attention_v11.model import CrossModel
        class Source(torch.nn.Module):
            n_tokens = 2
            d_model = 48
            def __init__(self):
                super().__init__(); self.tokens = torch.nn.Parameter(torch.randn(2, 48))
            def forward(self, batch):
                return self.tokens.unsqueeze(0).expand(len(batch['numeric']), -1, -1)
        torch.manual_seed(42)
        original = MacroOnFeatureTransformer(FeatureTransformer([f'x{i}' for i in range(56)], (3, 3, 3, 3), layers=3, pooling='cls'), Source(), 'both')
        model = CrossModel(original, 'summary').eval()
        batch = {'numeric': torch.randn(2, 56), 'categorical': torch.zeros(2, 4, dtype=torch.long)}
        output = model(batch)
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertGreater(model.original.source.tokens.grad.abs().sum().item(), 0)
        buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
        restored = CrossModel(MacroOnFeatureTransformer(FeatureTransformer([f'x{i}' for i in range(56)], (3, 3, 3, 3), layers=3, pooling='cls'), Source(), 'both'), 'summary').eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertTrue(torch.equal(output.detach(), restored(batch).detach()))

    def test_sequence_model_masks_padding_and_keeps_encoder_gradient(self):
        import numpy as np
        from experiments.feature_transformer_v4.models import FeatureTransformer
        from experiments.feature_transformer_v4.macro_v4 import MacroOnFeatureTransformer
        from experiments.macro_encoders_v3.encoders import build_encoders
        from experiments.macro_encoders_v3.api import WindowTensors
        from experiments.cross_attention_v11.model import CrossModel
        class Source(torch.nn.Module):
            n_tokens = 2
            d_model = 48
            def __init__(self):
                super().__init__(); self.encoders = build_encoders()
                self.register_buffer('row_window_index', torch.tensor([0, 0]), persistent=False)
                dbsmask = torch.zeros(1,63); dbsmask[:, -5:] = 1
                coemask = torch.zeros(1,12); coemask[:, -3:] = 1
                self.tensors = WindowTensors(torch.randn(1,63,7), dbsmask, torch.randn(1,12,24), coemask, torch.ones(1,12,24), np.array([1]), np.array([1]), np.zeros((1,12)), 'origin_asof', 'test')
            def _tensors_on(self, t, device): return t
        torch.manual_seed(43)
        source = Source()
        base = FeatureTransformer([f'x{i}' for i in range(56)], (3,3,3,3), layers=3, pooling='cls')
        model = CrossModel(MacroOnFeatureTransformer(base, source, 'both'), 'sequence').eval()
        batch = {'numeric':torch.randn(2,56), 'categorical':torch.zeros(2,4,dtype=torch.long), 'macro_row':torch.tensor([0,1])}
        baseline = model(batch)
        with torch.no_grad():
            source.tensors.dbs_seq[:, :-5] = 1000
            source.tensors.coe_seq[:, :-3] = 1000
        altered = model(batch)
        self.assertTrue(torch.allclose(baseline, altered))
        altered.sum().backward()
        self.assertGreater(source.encoders.dbs.lstm.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(source.encoders.coe.gru.weight_ih_l0.grad.abs().sum().item(), 0)
        with torch.no_grad():
            source.tensors.dbs_mask.zero_(); source.tensors.coe_mask.zero_()
        self.assertTrue(torch.isfinite(model(batch)).all())

if __name__ == '__main__': unittest.main()
