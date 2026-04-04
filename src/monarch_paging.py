#!/usr/bin/env python3
"""Monarch v3 paged cache primitives for Transformers inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


@dataclass
class MonarchPagingConfig:
    """Configuration for Monarch v3 context paging."""

    window_size: int = 512
    max_hot_tokens: int = 768
    page_size: int = 16
    compression_mode: str = "turboquant"
    attention_promote_threshold: float = 0.12
    sticky_threshold: int = 2
    importance_decay: float = 0.92
    initial_sticky_tokens: int = 16
    int4_group_size: int = 64
    polar_bins: int = 256


@dataclass
class QuantizedTensor4Bit:
    packed: torch.Tensor
    shape: Tuple[int, ...]
    scale: torch.Tensor
    original_dtype: torch.dtype
    original_device: str
    group_size: int


@dataclass
class TurboQuantCompressedTensor:
    base: PolarCompressedTensor
    residual_signs: torch.Tensor
    residual_scale: torch.Tensor
    original_dtype: torch.dtype
    original_device: str


@dataclass
class PolarCompressedTensor:
    radius_q: torch.Tensor
    angle_q: torch.Tensor
    radius_scale: torch.Tensor
    shape: Tuple[int, ...]
    original_dtype: torch.dtype
    original_device: str
    bins: int
    padded: bool


@dataclass
class TokenState:
    position: int
    token_id: Optional[int] = None
    attention_score: float = 0.0
    importance_ema: float = 0.0
    promotion_count: int = 0
    sticky: bool = False
    desired_hot: bool = True
    resident_hot: bool = True


@dataclass
class HotPage:
    positions: List[int] = field(default_factory=list)
    slot_idx: Optional[int] = None
    length: int = 0


@dataclass
class ColdPage:
    positions: List[int]
    keys: object
    values: object


def quantize_tensor_int4(tensor: torch.Tensor, group_size: int = 64) -> QuantizedTensor4Bit:
    original = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
    flat = original.view(-1)
    if flat.numel() == 0:
        return QuantizedTensor4Bit(
            packed=torch.empty(0, dtype=torch.uint8),
            shape=tuple(original.shape),
            scale=torch.empty(0, dtype=torch.float32),
            original_dtype=tensor.dtype,
            original_device=str(tensor.device),
            group_size=group_size,
        )

    num_groups = math.ceil(flat.numel() / group_size)
    padded_length = num_groups * group_size
    if padded_length != flat.numel():
        flat = torch.cat([flat, torch.zeros(padded_length - flat.numel(), dtype=flat.dtype)])

    grouped = flat.view(num_groups, group_size)
    scale = grouped.abs().amax(dim=1).clamp_min(1e-6) / 7.0
    quantized = torch.round(grouped / scale.unsqueeze(1)).clamp(-8, 7).to(torch.int16)
    unsigned = (quantized + 8).to(torch.uint8).view(-1)
    if unsigned.numel() % 2:
        unsigned = torch.cat([unsigned, torch.zeros(1, dtype=torch.uint8)])

    packed = unsigned[0::2] | (unsigned[1::2] << 4)
    return QuantizedTensor4Bit(
        packed=packed,
        shape=tuple(original.shape),
        scale=scale.to(torch.float32),
        original_dtype=tensor.dtype,
        original_device=str(tensor.device),
        group_size=group_size,
    )


def dequantize_tensor_int4(quantized: QuantizedTensor4Bit, device: torch.device) -> torch.Tensor:
    if quantized.packed.numel() == 0:
        return torch.empty(quantized.shape, dtype=quantized.original_dtype, device=device)

    packed = quantized.packed.to(torch.uint8)
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    unpacked[0::2] = packed & 0x0F
    unpacked[1::2] = (packed >> 4) & 0x0F

    total_elements = math.prod(quantized.shape)
    unpacked = unpacked[:total_elements].to(torch.int16) - 8
    padded_length = quantized.scale.numel() * quantized.group_size
    if unpacked.numel() < padded_length:
        unpacked = torch.cat(
            [unpacked, torch.zeros(padded_length - unpacked.numel(), dtype=unpacked.dtype)]
        )

    grouped = unpacked.view(quantized.scale.numel(), quantized.group_size).to(torch.float32)
    restored = grouped * quantized.scale.unsqueeze(1)
    restored = restored.view(-1)[:total_elements].view(quantized.shape)
    return restored.to(device=device, dtype=quantized.original_dtype)


def polar_compress_tensor(tensor: torch.Tensor, bins: int = 256) -> PolarCompressedTensor:
    original = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
    flat = original.view(-1)
    padded = flat.numel() % 2 == 1
    if padded:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype)])

    pairs = flat.view(-1, 2)
    radius = torch.linalg.vector_norm(pairs, dim=1)
    angle = torch.atan2(pairs[:, 1], pairs[:, 0])
    radius_scale = radius.max().clamp_min(1e-6)
    radius_q = torch.round((radius / radius_scale) * (bins - 1)).clamp(0, bins - 1)
    angle_q = torch.round(((angle + math.pi) / (2.0 * math.pi)) * (bins - 1)).clamp(0, bins - 1)

    return PolarCompressedTensor(
        radius_q=radius_q.to(torch.uint8),
        angle_q=angle_q.to(torch.uint8),
        radius_scale=radius_scale.to(torch.float32),
        shape=tuple(original.shape),
        original_dtype=tensor.dtype,
        original_device=str(tensor.device),
        bins=bins,
        padded=padded,
    )


def polar_decompress_tensor(compressed: PolarCompressedTensor, device: torch.device) -> torch.Tensor:
    if compressed.radius_q.numel() == 0:
        return torch.empty(compressed.shape, dtype=compressed.original_dtype, device=device)

    scale = float(compressed.bins - 1) if compressed.bins > 1 else 1.0
    radius = compressed.radius_q.to(torch.float32) / scale * compressed.radius_scale
    angle = compressed.angle_q.to(torch.float32) / scale * (2.0 * math.pi) - math.pi
    flat = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1).view(-1)
    total_elements = math.prod(compressed.shape)
    if compressed.padded:
        flat = flat[:total_elements]
    return flat.view(compressed.shape).to(device=device, dtype=compressed.original_dtype)


def turboquant_compress_tensor(tensor: torch.Tensor, bins: int = 256) -> TurboQuantCompressedTensor:
    """TurboQuant-style compression: PolarQuant base plus 1-bit residual correction."""
    original = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
    base = polar_compress_tensor(tensor, bins=bins)
    base_reconstructed = polar_decompress_tensor(base, device=torch.device("cpu")).to(torch.float32)
    residual = (original - base_reconstructed).contiguous()
    residual_signs = (residual >= 0).to(torch.uint8)
    residual_scale = residual.abs().mean().clamp_min(1e-8).to(torch.float32)
    return TurboQuantCompressedTensor(
        base=base,
        residual_signs=residual_signs,
        residual_scale=residual_scale,
        original_dtype=tensor.dtype,
        original_device=str(tensor.device),
    )


def turboquant_decompress_tensor(
    compressed: TurboQuantCompressedTensor,
    device: torch.device,
) -> torch.Tensor:
    """Decompress the internal TurboQuant-style tensor."""
    base = polar_decompress_tensor(compressed.base, device=torch.device("cpu")).to(torch.float32)
    signed_residual = compressed.residual_signs.to(torch.float32) * 2.0 - 1.0
    restored = base + signed_residual * compressed.residual_scale
    return restored.to(device=device, dtype=compressed.original_dtype)


class MonarchPagedLayer(CacheLayerMixin):
    """One cache layer backed by page-sized hot and cold blocks."""

    is_sliding = False

    def __init__(self, owner: "MonarchTransformersCache", layer_idx: int):
        super().__init__()
        self.owner = owner
        self.layer_idx = layer_idx
        self.hot_pages: Dict[int, HotPage] = {}
        self.cold_pages: Dict[int, ColdPage] = {}
        self.slot_keys: Optional[torch.Tensor] = None
        self.slot_values: Optional[torch.Tensor] = None
        self.free_slots: List[int] = []

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        batch_size, num_heads = key_states.shape[:2]
        head_dim = key_states.shape[-1]
        slot_count = self.owner.slot_count
        page_size = self.owner.config.page_size
        self.slot_keys = torch.zeros(
            (slot_count, batch_size, num_heads, page_size, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.slot_values = torch.zeros(
            (slot_count, batch_size, num_heads, page_size, value_states.shape[-1]),
            dtype=self.dtype,
            device=self.device,
        )
        self.free_slots = list(range(slot_count - 1, -1, -1))
        self.keys = torch.empty(0, dtype=self.dtype, device=self.device)
        self.values = torch.empty(0, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, object] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        positions = self.owner.begin_forward(key_states.shape[-2])
        for offset, position in enumerate(positions):
            self._append_token(
                position=position,
                key_tensor=key_states[:, :, offset : offset + 1, :].detach().clone(),
                value_tensor=value_states[:, :, offset : offset + 1, :].detach().clone(),
            )

        self.owner.finish_layer_update()
        self._refresh_materialized()
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.owner.hot_length + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        return self.owner.hot_length

    def get_max_cache_shape(self) -> int:
        return self.owner.config.max_hot_tokens + self.owner.config.page_size

    def ensure_page_hot(self, page_id: int) -> None:
        if page_id in self.hot_pages:
            return
        cold_page = self.cold_pages.pop(page_id, None)
        if cold_page is None:
            return
        slot_idx = self._acquire_slot()
        if self.owner.config.compression_mode == "turboquant":
            key_tensor = turboquant_decompress_tensor(cold_page.keys, device=self.device)
            value_tensor = turboquant_decompress_tensor(cold_page.values, device=self.device)
        else:
            key_tensor = polar_decompress_tensor(cold_page.keys, device=self.device)
            value_tensor = dequantize_tensor_int4(cold_page.values, device=self.device)
        length = key_tensor.shape[2]
        self.slot_keys[slot_idx].zero_()
        self.slot_values[slot_idx].zero_()
        self.slot_keys[slot_idx, :, :, :length, :] = key_tensor
        self.slot_values[slot_idx, :, :, :length, :] = value_tensor
        self.hot_pages[page_id] = HotPage(
            positions=list(cold_page.positions),
            slot_idx=slot_idx,
            length=length,
        )

    def ensure_page_cold(self, page_id: int) -> None:
        if page_id not in self.hot_pages:
            return
        hot_page = self.hot_pages.pop(page_id)
        if hot_page.slot_idx is None or hot_page.length == 0:
            if hot_page.slot_idx is not None:
                self._release_slot(hot_page.slot_idx)
            return
        key_tensor = self.slot_keys[hot_page.slot_idx, :, :, : hot_page.length, :].detach().clone()
        value_tensor = self.slot_values[hot_page.slot_idx, :, :, : hot_page.length, :].detach().clone()
        if self.owner.config.compression_mode == "turboquant":
            cold_keys = turboquant_compress_tensor(key_tensor, bins=self.owner.config.polar_bins)
            cold_values = turboquant_compress_tensor(value_tensor, bins=self.owner.config.polar_bins)
        else:
            cold_keys = polar_compress_tensor(key_tensor, bins=self.owner.config.polar_bins)
            cold_values = quantize_tensor_int4(
                value_tensor,
                group_size=self.owner.config.int4_group_size,
            )
        self.cold_pages[page_id] = ColdPage(
            positions=list(hot_page.positions),
            keys=cold_keys,
            values=cold_values,
        )
        self.slot_keys[hot_page.slot_idx].zero_()
        self.slot_values[hot_page.slot_idx].zero_()
        self._release_slot(hot_page.slot_idx)

    def _append_token(self, position: int, key_tensor: torch.Tensor, value_tensor: torch.Tensor) -> None:
        page_id = self.owner.page_id(position)
        self.ensure_page_hot(page_id)
        page = self.hot_pages.get(page_id)
        if page is None:
            page = HotPage(positions=[], slot_idx=self._acquire_slot(), length=0)
            self.hot_pages[page_id] = page

        if page.slot_idx is None:
            page.slot_idx = self._acquire_slot()
        page.positions.append(position)
        offset = page.length
        self.slot_keys[page.slot_idx, :, :, offset : offset + 1, :] = key_tensor
        self.slot_values[page.slot_idx, :, :, offset : offset + 1, :] = value_tensor
        page.length += 1
        self.cold_pages.pop(page_id, None)

    def _refresh_materialized(self) -> None:
        page_ids = self.owner.hot_page_ids_in_order()
        if not page_ids:
            page_ids = sorted(self.hot_pages.keys())

        if not page_ids:
            self.keys = torch.empty(0, dtype=self.dtype, device=self.device)
            self.values = torch.empty(0, dtype=self.dtype, device=self.device)
            return

        key_chunks: List[torch.Tensor] = []
        value_chunks: List[torch.Tensor] = []
        for page_id in page_ids:
            page = self.hot_pages.get(page_id)
            if page is None or page.slot_idx is None or page.length == 0:
                continue
            key_chunks.append(self.slot_keys[page.slot_idx, :, :, : page.length, :])
            value_chunks.append(self.slot_values[page.slot_idx, :, :, : page.length, :])

        self.keys = torch.cat(key_chunks, dim=2) if key_chunks else torch.empty(0, dtype=self.dtype, device=self.device)
        self.values = (
            torch.cat(value_chunks, dim=2) if value_chunks else torch.empty(0, dtype=self.dtype, device=self.device)
        )

    def _acquire_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("No free hot-page slots available in MonarchPagedLayer")
        return self.free_slots.pop()

    def _release_slot(self, slot_idx: int) -> None:
        self.free_slots.append(slot_idx)


class MonarchTransformersCache(Cache):
    """A Transformers cache that applies Monarch paging using page-sized residency."""

    def __init__(self, config: MonarchPagingConfig, num_hidden_layers: int):
        self.config = config
        # One extra slot covers a partially filled newest page; the second covers
        # the transient "append before eviction" case during decode.
        self.slot_count = max(1, math.ceil(config.max_hot_tokens / config.page_size) + 2)
        self.tokens: Dict[int, TokenState] = {}
        self.sequence: List[int] = []
        self.hot_positions: List[int] = []
        self.hot_page_ids: List[int] = []
        self.total_seen_tokens = 0
        self.page_ins = 0
        self.page_outs = 0
        self._current_forward_positions: Optional[List[int]] = None
        self._updated_layers_in_forward = 0
        layers = [MonarchPagedLayer(self, idx) for idx in range(num_hidden_layers)]
        super().__init__(layers=layers)

    @property
    def hot_length(self) -> int:
        return len(self.hot_positions)

    def page_id(self, position: int) -> int:
        return position // self.config.page_size

    def hot_page_ids_in_order(self) -> List[int]:
        return list(self.hot_page_ids)

    def active_positions(self) -> List[int]:
        return list(self.hot_positions)

    def begin_forward(self, seq_len: int) -> List[int]:
        if self._current_forward_positions is None:
            positions = list(range(self.total_seen_tokens, self.total_seen_tokens + seq_len))
            self._current_forward_positions = positions
            for position in positions:
                self.tokens[position] = TokenState(position=position, desired_hot=True, resident_hot=True)
                self.sequence.append(position)
        return self._current_forward_positions

    def finish_layer_update(self) -> None:
        self._updated_layers_in_forward += 1
        if self._updated_layers_in_forward == len(self.layers):
            self.total_seen_tokens += len(self._current_forward_positions or [])
            self._current_forward_positions = None
            self._updated_layers_in_forward = 0
            self._refresh_hot_positions()

    def finalize_prefill(self, input_ids: torch.Tensor, attentions: Optional[Sequence[torch.Tensor]]) -> None:
        token_ids = input_ids[0].detach().tolist()
        positions = list(range(len(token_ids)))
        attention_scores = self._aggregate_attention_scores(attentions, positions)
        initial_sticky = set(self._select_initial_sticky(attention_scores))
        for position, token_id in enumerate(token_ids):
            token = self.tokens[position]
            token.token_id = int(token_id)
            token.attention_score = attention_scores[position]
            token.importance_ema = attention_scores[position]
            token.sticky = position in initial_sticky
        self._apply_policy()

    def complete_decode_step(self, token_id: int, attentions: Optional[Sequence[torch.Tensor]]) -> None:
        latest_position = self.sequence[-1]
        self.tokens[latest_position].token_id = int(token_id)
        self._update_attention_scores(attentions)
        self._apply_policy()

    def _aggregate_attention_scores(
        self,
        attentions: Optional[Sequence[torch.Tensor]],
        positions: Optional[List[int]] = None,
    ) -> List[float]:
        """Aggregate attention scores across layers for specified positions.

        Args:
            attentions: Per-layer attention tensors
            positions: If provided, extract scores for these specific positions.
                      Otherwise returns empty list.
        Returns:
            List of aggregated attention scores, one per position.
        """
        if not positions:
            return [0.0 for _ in range(len(positions) if positions else 0)]
        if not attentions:
            return [0.0 for _ in range(len(positions))]

        scores = torch.zeros(len(positions), dtype=torch.float32)
        total_layers = 0
        for layer_attention in attentions:
            attn = layer_attention.detach().to("cpu", dtype=torch.float32)
            # attn shape: (batch, num_heads, 1, seq_len) during decode
            # We need to extract attention scores for specific positions
            attn_seq = attn.mean(dim=(0, 1, 2)).view(-1)  # shape: (seq_len,)
            for idx, position in enumerate(positions):
                if position < len(attn_seq):
                    scores[idx] += attn_seq[position]
            total_layers += 1
        if total_layers:
            scores /= float(total_layers)
        return scores.tolist()

    def _select_initial_sticky(self, attention_scores: Sequence[float]) -> List[int]:
        reserved_for_window = min(self.config.window_size, len(attention_scores))
        available = max(0, self.config.max_hot_tokens - reserved_for_window)
        sticky_count = min(self.config.initial_sticky_tokens, len(attention_scores), available)
        ranked = sorted(range(len(attention_scores)), key=lambda idx: attention_scores[idx], reverse=True)
        return ranked[:sticky_count]

    def _update_attention_scores(self, attentions: Optional[Sequence[torch.Tensor]]) -> None:
        for token in self.tokens.values():
            token.importance_ema *= self.config.importance_decay

        if not attentions or not self.hot_positions:
            return

        scores = self._aggregate_attention_scores(attentions, self.hot_positions)
        for position, score in zip(self.hot_positions, scores):
            token = self.tokens[position]
            token.attention_score = score
            token.importance_ema = max(token.importance_ema, score)
            if score >= self.config.attention_promote_threshold:
                token.promotion_count += 1
                if token.promotion_count >= self.config.sticky_threshold:
                    token.sticky = True

    def _apply_policy(self) -> None:
        recent_positions = set(self.sequence[-self.config.window_size :])
        desired_hot = {
            position
            for position in self.sequence
            if position in recent_positions or self.tokens[position].sticky
        }

        promotable = sorted(
            (
                self.tokens[position]
                for position in self.sequence
                if position not in desired_hot
                and self.tokens[position].importance_ema >= self.config.attention_promote_threshold
            ),
            key=lambda token: token.importance_ema,
            reverse=True,
        )
        for token in promotable:
            if len(desired_hot) >= self.config.max_hot_tokens:
                break
            desired_hot.add(token.position)
            token.promotion_count += 1
            if token.promotion_count >= self.config.sticky_threshold:
                token.sticky = True

        for token in self.tokens.values():
            token.desired_hot = token.position in desired_hot

        desired_pages = {
            self.page_id(position)
            for position in desired_hot
        }
        ordered_pages = [page_id for page_id in sorted(desired_pages)]
        previous_pages = set(self.hot_page_ids)
        page_out_ids = previous_pages - desired_pages
        page_in_ids = desired_pages - previous_pages

        for layer in self.layers:
            for page_id in page_out_ids:
                layer.ensure_page_cold(page_id)
            for page_id in page_in_ids:
                layer.ensure_page_hot(page_id)

        self.page_outs += len(page_out_ids)
        self.page_ins += len(page_in_ids)

        self.hot_page_ids = ordered_pages
        self._refresh_hot_positions()
        for token in self.tokens.values():
            token.resident_hot = self.page_id(token.position) in desired_pages
        for layer in self.layers:
            layer._refresh_materialized()

    def _refresh_hot_positions(self) -> None:
        hot_positions: List[int] = []
        hot_pages = set(self.hot_page_ids)
        for position in self.sequence:
            if self.page_id(position) in hot_pages:
                hot_positions.append(position)
        self.hot_positions = hot_positions

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        return tuple((layer.keys, layer.values) for layer in self.layers)

    def summary(self) -> Dict[str, int]:
        sticky = sum(1 for token in self.tokens.values() if token.sticky)
        desired_hot = sum(1 for token in self.tokens.values() if token.desired_hot)
        resident_hot = sum(1 for token in self.tokens.values() if token.resident_hot)
        promotions = sum(token.promotion_count for token in self.tokens.values())
        hot_attention_scores = [
            token.attention_score for token in self.tokens.values() if token.resident_hot
        ]
        hot_importance_scores = [
            token.importance_ema for token in self.tokens.values() if token.resident_hot
        ]
        return {
            "total_tokens": len(self.tokens),
            "desired_hot_tokens": desired_hot,
            "resident_hot_tokens": resident_hot,
            "cold_tokens": len(self.tokens) - resident_hot,
            "hot_pages": len(self.hot_page_ids),
            "slot_count": self.slot_count,
            "sticky_tokens": sticky,
            "promotions": promotions,
            "page_ins": self.page_ins,
            "page_outs": self.page_outs,
            "avg_attention_score": float(sum(hot_attention_scores) / len(hot_attention_scores)) if hot_attention_scores else 0.0,
            "avg_importance_ema": float(sum(hot_importance_scores) / len(hot_importance_scores)) if hot_importance_scores else 0.0,
        }
