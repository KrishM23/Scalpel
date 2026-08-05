"""Activation capture for the CLIP text tower.

We record, at the pooled (EOT) token position for every prompt:

- the residual stream after every encoder layer (``resid``),
- the input to every attention block's output projection, i.e. the
  concatenated per-head outputs before mixing (``attn_head_inputs``), and
- the output of every MLP block's down-projection (``mlp_out``).

These are exactly the quantities needed to (a) estimate a bias direction in
the residual stream and (b) attribute how much each attention head and MLP
writes onto that direction.
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
        """Per-head residual-stream writes at ``layer``: [num_heads, N, d].

        Head h of the attention block writes ``W_O[:, h*dh:(h+1)*dh] @ z_h``
        into the residual stream, where ``z`` is the concatenated head output
        we captured as the out_proj input.
        """
        w_o = lm.text_layers[layer].self_attn.out_proj.weight  # [d, d]
        z = self.attn_head_inputs[layer]  # [N, d]
        dh = lm.head_dim
        writes = []
        for h in range(lm.num_heads):
            cols = slice(h * dh, (h + 1) * dh)
            writes.append(z[:, cols] @ w_o[:, cols].T)  # [N, d]
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
        eot = lm.eot_positions(batch["input_ids"])
        rows = torch.arange(eot.shape[0], device=lm.device)

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
            for i, layer in enumerate(lm.text_layers):
                stack.callback(
                    layer.self_attn.out_proj.register_forward_pre_hook(make_attn_hook(i)).remove
                )
                stack.callback(layer.mlp.fc2.register_forward_hook(make_mlp_hook(i)).remove)
            outputs = lm.model.text_model(**batch, output_hidden_states=True)

        # hidden_states: tuple of L+1 tensors [B, T, d]
        resid_chunks.append(
            torch.stack([hs[rows, eot] for hs in outputs.hidden_states], dim=0)
        )
        attn_chunks.append(torch.stack([a[rows, eot] for a in attn_inputs], dim=0))
        mlp_chunks.append(torch.stack([m[rows, eot] for m in mlp_outputs], dim=0))

    return ComponentActivations(
        resid=torch.cat(resid_chunks, dim=1),
        attn_head_inputs=torch.cat(attn_chunks, dim=1),
        mlp_out=torch.cat(mlp_chunks, dim=1),
    )
