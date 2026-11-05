"""Cross-attention alternatives using the original macro encoders and FT weights."""
import math
import torch
from torch import nn
from experiments.macro_encoders_v3.encoders import compact_masked


def sequence_states(encoder, sequence, mask):
    compact, cmask, lengths = compact_masked(sequence, mask)
    rnn = encoder.lstm if hasattr(encoder, 'lstm') else encoder.gru
    packed = nn.utils.rnn.pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
    output, _ = rnn(packed)
    hidden, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=sequence.shape[1])
    positions = torch.arange(sequence.shape[1], device=sequence.device).expand_as(mask)
    order = torch.argsort(torch.where(mask.bool(), positions, positions + sequence.shape[1]), dim=1, stable=True)
    restored = torch.zeros_like(hidden).scatter(1, order.unsqueeze(-1).expand_as(hidden), hidden)
    valid = mask.bool().clone()
    missing = ~valid.any(dim=1)
    restored[missing, 0] = encoder.empty
    valid[missing, 0] = True
    return encoder.projection(restored), valid


def lag_encoding(length, width, device, dtype):
    lag = torch.arange(length - 1, -1, -1, device=device, dtype=dtype).unsqueeze(1)
    rate = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype) * (-math.log(10000.0) / width))
    out = torch.zeros(length, width, device=device, dtype=dtype)
    out[:, 0::2] = torch.sin(lag * rate)
    out[:, 1::2] = torch.cos(lag * rate)
    return out


class CrossModel(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.original = original
        self.mode = mode

    def memory(self, batch):
        original = self.original
        if self.mode == 'summary':
            values = original.source(batch)
            valid = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        else:
            from experiments.macro_encoders_v3.api import batch_indices, select_window_tensors
            source = original.source
            rows = batch['macro_row'].cpu().numpy()
            unique, inverse = batch_indices(source.row_window_index.cpu().numpy(), rows, source.tensors.n_windows)
            subset = source._tensors_on(select_window_tensors(source.tensors, unique), next(source.parameters()).device)
            parts, masks = [], []
            for encoder, x, mask in ((source.encoders.dbs, subset.dbs_seq, subset.dbs_mask), (source.encoders.coe, subset.coe_seq, subset.coe_mask)):
                states, valid = sequence_states(encoder, x, mask)
                states = states + lag_encoding(states.shape[1], states.shape[2], states.device, states.dtype)
                parts.append(states); masks.append(valid)
            idx = torch.as_tensor(inverse, device=parts[0].device)
            values = torch.cat([p.index_select(0, idx) for p in parts], dim=1)
            valid = torch.cat([p.index_select(0, idx) for p in masks], dim=1)
        types = original.base.type_embedding(torch.tensor(original.macro_type_ids, device=values.device))
        if self.mode == 'summary':
            values = values + types.unsqueeze(0)
        else:
            values = values + torch.cat([types[0].expand(63, -1), types[1].expand(12, -1)]).unsqueeze(0)
        return values, valid

    def forward(self, batch):
        b = self.original.base
        feature = b.feature_tokens(batch)
        query = b._query_token(batch, feature.dtype)
        x = torch.cat([query, feature], dim=1)
        for layer in b.encoder.layers[:2]:
            x = layer(x)
        memory, valid = self.memory(batch)
        layer = b.encoder.layers[2]
        normed = layer.norm1(x)
        memnorm = layer.norm1(memory)
        attention = layer.self_attn(normed, memnorm, memnorm, key_padding_mask=~valid, need_weights=False)[0]
        x = x + layer.dropout1(attention)
        x = x + layer._ff_block(layer.norm2(x))
        return b.head(b.final_norm(x)[:, 0]).squeeze(-1)
