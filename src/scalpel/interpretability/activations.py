"""Activation capture for the text tower (any supported architecture).

We record, at the pooled token position for every prompt:

- the residual stream after every encoder/decoder layer (``resid``),
- the input to every attention block's output projection (``attn_head_inputs``),
- the output of every MLP down-projection (``mlp_out``).

These feed bias-direction estimation and circuit attribution.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass

import torch

from scalpel.models.registry import LoadedModel


@dataclass
class ComponentActivations:
    """Pooled-position activations for a list of prompts.

    Shapes (N prompts, L layers, d model width):
      resid:            [L + 1, N, d]  (embedding output + each layer)
      attn_head_inputs: [L, N, d]      (concat head outputs, pre out_proj)
      mlp_out:          [L, N, d]
    """

    resid: torch.Tensor
    attn_head_inputs: torch.Tensor
    mlp_out: torch.Tensor

    def head_writes(self, lm: LoadedModel, layer: int) -> torch.Tensor:
        """Per-head residual-stream writes at ``layer``: [num_heads, N, d]."""
        view = lm.layers[layer]
        w_o = view.attn_weight
        z = self.attn_head_inputs[layer]  # [N, d]
        dh = lm.head_dim
        writes = []
        for h in range(lm.num_heads):
            cols = slice(h * dh, (h + 1) * dh)
            if view.layout == "conv1d":
                # y = z @ W  with W [d_in, d_out]; head h owns input rows.
                writes.append(z[:, cols] @ w_o[cols, :])
            else:
                # y = z @ W.T  with W [d_out, d_in]; head h owns weight columns.
                writes.append(z[:, cols] @ w_o[:, cols].T)
        return torch.stack(writes, dim=0)


@torch.no_grad()
def record_component_activations(
    lm: LoadedModel, texts: list[str], batch_size: int = 64
) -> ComponentActivations:
    """Run the text tower and capture pooled-position component activations."""
    num_layers = lm.num_layers
    resid_chunks: list[torch.Tensor] = []
    attn_chunks: list[torch.Tensor] = []
    mlp_chunks: list[torch.Tensor] = []

    for start in range(0, len(texts), batch_size):
        batch = lm.tokenize(texts[start : start + batch_size])
        pool = lm.pool_positions(batch["input_ids"])
        rows = torch.arange(pool.shape[0], device=lm.device)

        attn_inputs: list[torch.Tensor | None] = [None] * num_layers
        mlp_outputs: list[torch.Tensor | None] = [None] * num_layers

        def make_attn_hook(layer_idx: int, store=attn_inputs):
            def hook(_module, args):
                store[layer_idx] = args[0].detach()

            return hook

        def make_mlp_hook(layer_idx: int, store=mlp_outputs):
            def hook(_module, _args, output):
                store[layer_idx] = output.detach()

            return hook

        with ExitStack() as stack:
            for i, view in enumerate(lm.layers):
                stack.callback(
                    view.attn_out.register_forward_pre_hook(make_attn_hook(i)).remove
                )
                stack.callback(view.mlp_out.register_forward_hook(make_mlp_hook(i)).remove)
            outputs = lm.forward_hidden_states(batch)

        resid_chunks.append(
            torch.stack([hs[rows, pool] for hs in outputs.hidden_states], dim=0)
        )
        attn_chunks.append(torch.stack([a[rows, pool] for a in attn_inputs], dim=0))
        mlp_chunks.append(torch.stack([m[rows, pool] for m in mlp_outputs], dim=0))

    return ComponentActivations(
        resid=torch.cat(resid_chunks, dim=1),
        attn_head_inputs=torch.cat(attn_chunks, dim=1),
        mlp_out=torch.cat(mlp_chunks, dim=1),
    )
