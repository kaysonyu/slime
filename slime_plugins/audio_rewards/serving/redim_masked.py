"""Length-aware ReDimNet2 operations for full-utterance variable-length batches.

The original network has time-dependent GroupNorm, attention and ASTP. Masking
only the final pooling is insufficient. Convolution inputs must also be zero
outside each utterance so biased padded activations cannot leak across boundaries.
"""

import types

import torch
import torch.nn.functional as F


class LengthMasks:
    def __init__(self, thread_state):
        self.state = thread_state

    def mask(self, time):
        if not hasattr(self.state, "frame_lengths"):
            return None
        if time not in self.state.masks:
            stride = self.state.max_frames // time
            if stride not in {1, 2, 4} or self.state.max_frames % time:
                raise ValueError("unexpected time resolution in the pinned ReDimNet2 architecture")
            lengths = self.state.frame_lengths // stride
            self.state.masks[time] = torch.arange(time, device=lengths.device)[None] < lengths[:, None]
        return self.state.masks[time]

    def conv_input(self, module, args):
        x = args[0]
        mask = self.mask(x.shape[-1])
        if mask is None:
            return args
        shape = (len(mask),) + (1,) * (x.ndim - 2) + (x.shape[-1],)
        return (x * mask.reshape(shape), *args[1:])

    def install(self, model):
        masks = self

        def group_norm(layer, x):
            mask = masks.mask(x.shape[-1])
            if mask is None:
                return F.group_norm(x, layer.num_groups, layer.weight, layer.bias, layer.eps)
            dtype = x.dtype
            batch, channels, time = x.shape
            with torch.autocast(x.device.type, enabled=False):
                y = x.float().reshape(batch, layer.num_groups, channels // layer.num_groups, time)
                valid = mask[:, None, None, :]
                count = mask.sum(-1)[:, None, None, None] * (channels // layer.num_groups)
                mean = (y * valid).sum((2, 3), keepdim=True) / count
                variance = ((y - mean).square() * valid).sum((2, 3), keepdim=True) / count
                y = ((y - mean) * torch.rsqrt(variance + layer.eps)).reshape_as(x)
                if layer.affine:
                    y = y * layer.weight.float()[None, :, None] + layer.bias.float()[None, :, None]
                return (y * mask[:, None, :]).to(dtype)

        def pooling(layer, x):
            if x.ndim == 4:
                x = x.flatten(1, 2)
            mask = masks.mask(x.shape[-1])
            if mask is None:
                return layer._unmasked_forward(x)
            x = x.float()
            valid = mask[:, None, :]
            count = mask.sum(-1)[:, None, None]
            mean = (x * valid).sum(-1, keepdim=True) / count
            variance = ((x - mean).square() * valid).sum(-1, keepdim=True) / (count - 1)
            context = torch.cat((x, mean.expand_as(x), (variance + 1e-7).sqrt().expand_as(x)), dim=1)
            logits = layer.linear2(torch.tanh(layer.linear1(context)))
            alpha = torch.softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
            mean = (alpha * x).sum(-1)
            variance = (alpha * x.square()).sum(-1) - mean.square()
            return torch.cat((mean, variance.clamp(min=1e-7).sqrt()), dim=1)

        for module in model.backbone.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
                module.register_forward_pre_hook(self.conv_input)
            elif isinstance(module, torch.nn.GroupNorm):
                module.forward = types.MethodType(group_norm, module)
        model.pool._unmasked_forward = model.pool.forward
        model.pool.forward = types.MethodType(pooling, model.pool)
