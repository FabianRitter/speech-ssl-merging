"""
G4.2: End-to-End Differentiable Sinkhorn Permutation Alignment

Applies learnable soft permutations inside model B's forward pass so that
gradients flow through the entire layer chain — a truly global search.

Key class: PermutedModelWrapper
Key function: optimize_sinkhorn_e2e()

Design doc: orchestration/agents/G4_2_e2e_research_report.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import re
from typing import Dict, List, Optional, Tuple
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sinkhorn utilities (shared with G4.1 in matching_functions.py)
# ---------------------------------------------------------------------------

def sinkhorn_normalization(log_alpha: torch.Tensor, num_iters: int = 20) -> torch.Tensor:
    """Sinkhorn normalization: log-space alternating row/column normalization."""
    for _ in range(num_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def get_temperature(step: int, total_steps: int,
                    tau_max: float = 1.0, tau_min: float = 0.01) -> float:
    """Exponential temperature annealing."""
    if total_steps <= 1:
        return tau_min
    progress = step / (total_steps - 1)
    return tau_max * (tau_min / tau_max) ** progress


def cross_correlation_loss(F_A: torch.Tensor, F_B: torch.Tensor,
                           use_simple_trace: bool = True,
                           lambda_off_diag: float = 0.01,
                           eps: float = 1e-7) -> torch.Tensor:
    """Cross-correlation loss between model A and (permuted) model B features."""
    D, N = F_A.shape
    # Ensure both tensors are float32 (model A may be float16 from autocast hooks)
    F_A = F_A.float()
    F_B = F_B.float()
    F_A_n = (F_A - F_A.mean(1, keepdim=True)) / (F_A.std(1, keepdim=True) + eps)
    F_B_n = (F_B - F_B.mean(1, keepdim=True)) / (F_B.std(1, keepdim=True) + eps)
    C = (F_A_n @ F_B_n.T) / N
    if use_simple_trace:
        return -C.diagonal().sum() / D
    diag_loss = -((C.diagonal() ** 2).sum()) / D
    off_mask = ~torch.eye(D, dtype=torch.bool, device=C.device)
    off_loss = (C[off_mask] ** 2).sum() / (D * (D - 1))
    return diag_loss + lambda_off_diag * off_loss


# ---------------------------------------------------------------------------
# PermutedModelWrapper
# ---------------------------------------------------------------------------

class PermutedModelWrapper(nn.Module):
    """
    Wraps model B and applies learnable soft permutations at each merge point
    during the forward pass.  Inner model parameters are FROZEN; only the
    logit matrices are learnable.

    Merge points (ff+attn mode):
      - CNN blocks 0-6: after each Conv1d block, dim=512
      - Transformer layers: after attention block (block-diag per head), after fc1 (before fc2)
    """

    def __init__(self, inner_model: nn.Module, num_layers: int = 3,
                 embed_dim: int = 768, ffn_dim: int = 3072,
                 num_heads: int = 12, merge_cnn: bool = True,
                 sinkhorn_iters: int = 20, device: str = 'cuda'):
        super().__init__()
        self.inner = inner_model
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.merge_cnn = merge_cnn
        self.sinkhorn_iters = sinkhorn_iters
        self._device = device
        self.cnn_dim = 512

        # Freeze inner model
        for p in self.inner.parameters():
            p.requires_grad = False
        self.inner.eval()

        # Learnable logit matrices — initialized near identity to avoid
        # degenerate uniform averaging (especially for large D like ffn_dim=3072).
        # A diagonal bias of ~1.0 ensures the initial Sinkhorn output is close
        # to identity rather than 1/D uniform, preserving gradient signal.
        identity_bias = 1.0

        if merge_cnn:
            self.cnn_logits = nn.ParameterList([
                nn.Parameter(identity_bias * torch.eye(self.cnn_dim, device=device))
                for _ in range(7)
            ])
        else:
            self.cnn_logits = None

        # Per-head attention logits
        self.attn_head_logits = nn.ParameterList()
        for _ in range(num_layers):
            heads = nn.ParameterList([
                nn.Parameter(identity_bias * torch.eye(self.head_dim, device=device))
                for _ in range(num_heads)
            ])
            self.attn_head_logits.append(heads)

        # FFN logits (before fc2)
        self.ffn_logits = nn.ParameterList([
            nn.Parameter(identity_bias * torch.eye(ffn_dim, device=device))
            for _ in range(num_layers)
        ])

    def get_all_logit_params(self) -> List[nn.Parameter]:
        params = []
        if self.cnn_logits is not None:
            params.extend(self.cnn_logits)
        for layer_heads in self.attn_head_logits:
            params.extend(layer_heads)
        params.extend(self.ffn_logits)
        return params

    def _soft_perm_attn(self, layer_idx: int, tau: float) -> torch.Tensor:
        """Block-diagonal soft permutation for attention output. [embed_dim, embed_dim]"""
        blocks = []
        for h in range(self.num_heads):
            S_h = sinkhorn_normalization(
                self.attn_head_logits[layer_idx][h] / tau,
                num_iters=self.sinkhorn_iters
            )
            blocks.append(S_h)
        return torch.block_diag(*blocks)

    def _soft_perm_ffn(self, layer_idx: int, tau: float) -> torch.Tensor:
        """Soft permutation for FFN intermediate. [ffn_dim, ffn_dim]"""
        return sinkhorn_normalization(
            self.ffn_logits[layer_idx] / tau, num_iters=self.sinkhorn_iters
        )

    def _soft_perm_cnn(self, idx: int, tau: float) -> torch.Tensor:
        """Soft permutation for CNN block. [512, 512]"""
        return sinkhorn_normalization(
            self.cnn_logits[idx] / tau, num_iters=self.sinkhorn_iters
        )

    def forward(self, wavs: list, tau: float = 1.0,
                collect_features: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with soft permutations at each merge point.

        Returns:
            features_dict: {merge_point_name: [D, N_tokens]}
        """
        features = {}
        mdm = self.inner.inner if hasattr(self.inner, 'inner') else self.inner

        # --- Pad waveforms ---
        max_len = max(w.shape[-1] for w in wavs)
        padded = torch.zeros(len(wavs), max_len, device=wavs[0].device)
        wav_lengths = []
        for i, wav in enumerate(wavs):
            length = wav.shape[-1]
            padded[i, :length] = wav
            wav_lengths.append(length)

        # --- CNN ---
        x = padded.unsqueeze(1)  # [B, 1, T]
        cnn_blocks = list(mdm.feature_extractor.conv_layers)
        for block_idx, conv_block in enumerate(cnn_blocks):
            x = conv_block(x)  # [B, 512, T_l]
            if self.merge_cnn and self.cnn_logits is not None:
                S = self._soft_perm_cnn(block_idx, tau)
                x = torch.einsum('ij,bjt->bit', S, x)
                if collect_features:
                    B_s, C_s, T_s = x.shape
                    features[f'cnn_{block_idx}'] = x.permute(1, 0, 2).reshape(C_s, -1)

        # [B, 512, T_cnn] -> [B, T_cnn, 512]
        feat_cnn = x.transpose(1, 2)

        # --- Post-extract projection ---
        if mdm.post_extract_proj is not None:
            feat_proj = mdm.post_extract_proj(feat_cnn)
        else:
            feat_proj = feat_cnn

        # --- Positional encoding ---
        x_enc = feat_proj
        pos_conv = mdm.encoder.pos_conv
        x_conv = pos_conv(x_enc.transpose(1, 2)).transpose(1, 2)
        x_enc = x_enc + x_conv

        if not mdm.encoder.layer_norm_first:
            x_enc = mdm.encoder.layer_norm(x_enc)

        x_enc = F.dropout(x_enc, p=mdm.encoder.dropout, training=False)
        x_enc = x_enc.transpose(0, 1)  # T x B x C

        # Encoder padding mask
        valid_lens = []
        conv_spec = eval(mdm.config.extractor_conv_feature_layers)
        for wl in wav_lengths:
            cl = wl
            for _, k, s in conv_spec:
                cl = (cl - k) // s + 1
            valid_lens.append(int(cl))
        encoder_pad_mask = torch.ones(len(wavs), feat_proj.shape[1],
                                      dtype=torch.bool, device=self._device)
        for i, vl in enumerate(valid_lens):
            encoder_pad_mask[i, vl:] = False
        encoder_pad_mask = ~encoder_pad_mask  # True = padded

        # --- Transformer layers with permutations ---
        for layer_idx, layer in enumerate(mdm.encoder.layers):
            if layer.layer_norm_first:
                # Pre-LN variant
                residual = x_enc
                x_normed = layer.self_attn_layer_norm(x_enc)
                attn_out, _ = layer.self_attn(
                    query=x_normed, key=x_normed, value=x_normed,
                    key_padding_mask=encoder_pad_mask, need_weights=False)

                # Attention output permutation
                S_attn = self._soft_perm_attn(layer_idx, tau)
                T_l, B_s, E_d = attn_out.shape
                attn_flat = attn_out.reshape(T_l * B_s, E_d)
                attn_out = (S_attn @ attn_flat.T).T.reshape(T_l, B_s, E_d)

                if collect_features:
                    features[f'attn_out_{layer_idx}'] = attn_out.permute(2, 0, 1).reshape(E_d, -1)

                attn_out = layer.dropout1(attn_out)
                x_enc = residual + attn_out

                residual = x_enc
                x_normed = layer.final_layer_norm(x_enc)
                ffn_out = layer.activation_fn(layer.fc1(x_normed))
                ffn_out = layer.dropout2(ffn_out)

                # FFN permutation
                S_ffn = self._soft_perm_ffn(layer_idx, tau)
                T_l, B_s, FFN_d = ffn_out.shape
                ffn_flat = ffn_out.reshape(T_l * B_s, FFN_d)
                ffn_out = (S_ffn @ ffn_flat.T).T.reshape(T_l, B_s, FFN_d)

                if collect_features:
                    features[f'ffn_{layer_idx}'] = ffn_out.permute(2, 0, 1).reshape(FFN_d, -1)

                ffn_out = layer.fc2(ffn_out)
                ffn_out = layer.dropout3(ffn_out)
                x_enc = residual + ffn_out
            else:
                # Post-LN variant (default for distilled models)
                residual = x_enc
                attn_out, _ = layer.self_attn(
                    query=x_enc, key=x_enc, value=x_enc,
                    key_padding_mask=encoder_pad_mask, need_weights=False)

                S_attn = self._soft_perm_attn(layer_idx, tau)
                T_l, B_s, E_d = attn_out.shape
                attn_flat = attn_out.reshape(T_l * B_s, E_d)
                attn_out = (S_attn @ attn_flat.T).T.reshape(T_l, B_s, E_d)

                if collect_features:
                    features[f'attn_out_{layer_idx}'] = attn_out.permute(2, 0, 1).reshape(E_d, -1)

                attn_out = layer.dropout1(attn_out)
                x_enc = residual + attn_out
                x_enc = layer.self_attn_layer_norm(x_enc)

                residual = x_enc
                ffn_out = layer.activation_fn(layer.fc1(x_enc))
                ffn_out = layer.dropout2(ffn_out)

                S_ffn = self._soft_perm_ffn(layer_idx, tau)
                T_l, B_s, FFN_d = ffn_out.shape
                ffn_flat = ffn_out.reshape(T_l * B_s, FFN_d)
                ffn_out = (S_ffn @ ffn_flat.T).T.reshape(T_l, B_s, FFN_d)

                if collect_features:
                    features[f'ffn_{layer_idx}'] = ffn_out.permute(2, 0, 1).reshape(FFN_d, -1)

                ffn_out = layer.fc2(ffn_out)
                ffn_out = layer.dropout3(ffn_out)
                x_enc = residual + ffn_out
                x_enc = layer.final_layer_norm(x_enc)

        if mdm.encoder.layer_norm_first:
            x_enc = mdm.encoder.layer_norm(x_enc)

        return features

    def extract_hard_permutations(self, tau_min: float = 0.01) -> Dict[str, torch.Tensor]:
        """Extract hard permutation matrices via Hungarian on final soft Sinkhorn matrices."""
        perms = {}

        if self.cnn_logits is not None:
            for i, logit in enumerate(self.cnn_logits):
                with torch.no_grad():
                    S = sinkhorn_normalization(logit / tau_min, self.sinkhorn_iters * 2)
                    _, col_ind = linear_sum_assignment(S.cpu().numpy(), maximize=True)
                    perms[f'cnn_{i}'] = torch.eye(self.cnn_dim, device=self._device)[
                        torch.tensor(col_ind).long().to(self._device)]

        for li in range(self.num_layers):
            head_Ps = []
            for h in range(self.num_heads):
                with torch.no_grad():
                    S_h = sinkhorn_normalization(
                        self.attn_head_logits[li][h] / tau_min, self.sinkhorn_iters * 2)
                    _, col_ind = linear_sum_assignment(S_h.cpu().numpy(), maximize=True)
                    P_h = torch.eye(self.head_dim, device=self._device)[
                        torch.tensor(col_ind).long().to(self._device)]
                    head_Ps.append(P_h)
            perms[f'attn_out_{li}'] = torch.block_diag(*head_Ps)

        for li in range(self.num_layers):
            with torch.no_grad():
                S = sinkhorn_normalization(
                    self.ffn_logits[li] / tau_min, self.sinkhorn_iters * 2)
                _, col_ind = linear_sum_assignment(S.cpu().numpy(), maximize=True)
                perms[f'ffn_{li}'] = torch.eye(self.ffn_dim, device=self._device)[
                    torch.tensor(col_ind).long().to(self._device)]

        return perms


# ---------------------------------------------------------------------------
# Feature collection for model A (hook-based, via existing graph infra)
# ---------------------------------------------------------------------------

def collect_model_a_features(graph_a, wavs_gpu: list,
                             device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Collect features from model A at PREFIX nodes using graph hooks."""
    from graphs.base_graph import NodeType

    graph_a.add_hooks(device=device)
    graph_a.intermediates = {}

    with torch.no_grad():
        graph_a.compute_intermediates(wavs_gpu, device=device)

    features = {}
    for node_id, feat in graph_a.intermediates.items():
        name = _node_id_to_name(graph_a, node_id)
        if name is not None:
            features[name] = feat.detach()

    graph_a.clear_hooks()
    return features


def _node_id_to_name(graph, node_id) -> Optional[str]:
    """Map graph PREFIX node ID to merge point name (cnn_X, attn_out_X, ffn_X)."""
    from graphs.base_graph import NodeType

    info = graph.get_node_info(node_id)
    if info.get('type') != NodeType.PREFIX:
        return None

    for succ in graph.G.successors(node_id):
        succ_info = graph.get_node_info(succ)
        if succ_info.get('type') == NodeType.MODULE:
            layer_name = succ_info.get('layer', '')
            m = re.search(r'conv_layers\.(\d+)', layer_name)
            if m:
                return f'cnn_{m.group(1)}'
            if 'self_attn.out_proj' in layer_name:
                m = re.search(r'encoder\.layers\.(\d+)', layer_name)
                if m:
                    return f'attn_out_{m.group(1)}'
            if '.fc2' in layer_name:
                m = re.search(r'encoder\.layers\.(\d+)', layer_name)
                if m:
                    return f'ffn_{m.group(1)}'
    return None


def build_node_id_to_name_mapping(graph) -> Dict[int, str]:
    """Build complete node_id → merge_point_name mapping."""
    mapping = {}
    for node_id in graph.G.nodes:
        name = _node_id_to_name(graph, node_id)
        if name is not None:
            mapping[node_id] = name
    return mapping


# ---------------------------------------------------------------------------
# Total loss across all merge points
# ---------------------------------------------------------------------------

def compute_total_loss(features_a: Dict[str, torch.Tensor],
                       features_b: Dict[str, torch.Tensor],
                       layer_weights: Optional[Dict[str, float]] = None,
                       use_simple_trace: bool = True,
                       lambda_off_diag: float = 0.01
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Sum of cross-correlation losses across all shared merge points."""
    device = next(iter(features_b.values())).device
    total = torch.tensor(0.0, device=device)
    per_layer = {}
    n = 0
    for name in features_b:
        if name not in features_a:
            continue
        fa, fb = features_a[name], features_b[name]
        min_t = min(fa.shape[1], fb.shape[1])
        loss_l = cross_correlation_loss(fa[:, :min_t], fb[:, :min_t],
                                        use_simple_trace=use_simple_trace,
                                        lambda_off_diag=lambda_off_diag)
        w = layer_weights.get(name, 1.0) if layer_weights else 1.0
        total = total + w * loss_l
        per_layer[name] = loss_l.item()
        n += 1
    if n > 0:
        total = total / n
    return total, per_layer


# ---------------------------------------------------------------------------
# Main optimization loop
# ---------------------------------------------------------------------------

def optimize_sinkhorn_e2e(
    model_a_graph,           # HuBERTGraph for model A
    permuted_model_b,        # PermutedModelWrapper
    dataloader,
    num_opt_steps: int = 200,
    lr: float = 0.005,
    tau_max: float = 1.0,
    tau_min: float = 0.05,
    grad_clip_norm: float = 5.0,
    use_simple_trace: bool = True,
    lambda_off_diag: float = 0.01,
    layer_weights: Optional[Dict[str, float]] = None,
    log_every: int = 10,
    device: str = 'cuda',
    max_batch_e2e: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    End-to-end Sinkhorn optimization.

    Returns: {merge_point_name: [D, D] hard permutation matrix}
    """
    # Build per-param-group optimizer with separate LRs for CNN/attention vs FFN
    import itertools
    param_groups = []
    cnn_attn_params = []
    ffn_params = list(permuted_model_b.ffn_logits.parameters())
    if permuted_model_b.cnn_logits is not None:
        cnn_attn_params.extend(permuted_model_b.cnn_logits.parameters())
    for layer_heads in permuted_model_b.attn_head_logits:
        cnn_attn_params.extend(layer_heads.parameters())
    lr_ffn = lr * 0.2  # FFN needs lower LR due to D=3072 gradient attenuation
    if cnn_attn_params:
        param_groups.append({'params': cnn_attn_params, 'lr': lr})
    if ffn_params:
        param_groups.append({'params': ffn_params, 'lr': lr_ffn})
    optimizer = torch.optim.Adam(param_groups)

    logit_params = permuted_model_b.get_all_logit_params()
    data_iter = iter(dataloader)
    node_to_name = build_node_id_to_name_mapping(model_a_graph)

    best_loss = float('inf')
    best_states = [p.data.clone() for p in logit_params]

    logger.info(f"E2E Sinkhorn: {num_opt_steps} steps, lr_cnn_attn={lr}, lr_ffn={lr_ffn}, "
                f"tau={tau_max}->{tau_min}, logit_params={len(logit_params)}")

    for step in range(num_opt_steps):
        tau = get_temperature(step, num_opt_steps, tau_max, tau_min)

        # Get batch
        try:
            wavs, *_ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            wavs, *_ = next(data_iter)
        wavs_gpu = [torch.FloatTensor(w).to(device) for w in wavs]

        # Limit batch size to avoid OOM during backward pass
        if len(wavs_gpu) > max_batch_e2e:
            wavs_gpu = wavs_gpu[:max_batch_e2e]

        # Model A features (no grad, hook-based)
        with torch.no_grad():
            model_a_graph.add_hooks(device=device)
            model_a_graph.intermediates = {}
            model_a_graph.compute_intermediates(wavs_gpu, device=device)
            features_a = {}
            for nid, feat in model_a_graph.intermediates.items():
                if nid in node_to_name:
                    features_a[node_to_name[nid]] = feat.detach()
            model_a_graph.clear_hooks()

        # Model B features (WITH grad through soft permutations)
        features_b = permuted_model_b(wavs_gpu, tau=tau, collect_features=True)

        # Debug: log feature name mismatches on first step
        if step == 0:
            logger.info(f"  features_a keys: {sorted(features_a.keys())}")
            logger.info(f"  features_b keys: {sorted(features_b.keys())}")
            for k in sorted(set(list(features_a.keys()) + list(features_b.keys()))):
                in_a = k in features_a
                in_b = k in features_b
                if in_a and in_b:
                    logger.info(f"    {k}: A={features_a[k].shape} B={features_b[k].shape} OK")
                elif in_a:
                    logger.info(f"    {k}: A={features_a[k].shape} B=MISSING")
                else:
                    logger.info(f"    {k}: A=MISSING B={features_b[k].shape}")

        # Loss
        loss, per_layer = compute_total_loss(
            features_a, features_b, layer_weights=layer_weights,
            use_simple_trace=use_simple_trace, lambda_off_diag=lambda_off_diag)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(logit_params, grad_clip_norm)
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_states = [p.data.clone() for p in logit_params]

        if step % log_every == 0 or step == num_opt_steps - 1:
            pl = ', '.join(f'{k}={v:.4f}' for k, v in sorted(per_layer.items()))
            logger.info(f"  Step {step:4d}/{num_opt_steps} | tau={tau:.4f} | "
                        f"loss={loss_val:.6f} | best={best_loss:.6f} | "
                        f"∇={grad_norm:.3f} | {pl}")

    # Restore best
    for p, s in zip(logit_params, best_states):
        p.data.copy_(s)

    logger.info(f"E2E optimization done. Best loss: {best_loss:.6f}")

    # Extract hard permutations
    hard_perms = permuted_model_b.extract_hard_permutations(tau_min=tau_min)
    for name, P in sorted(hard_perms.items()):
        D = P.shape[0]
        perm_idx = torch.argmax(P, dim=1)
        n_changed = (perm_idx != torch.arange(D, device=P.device)).sum().item()
        logger.info(f"  {name}: {n_changed}/{D} permuted ({100*n_changed/D:.1f}%)")

    return hard_perms
