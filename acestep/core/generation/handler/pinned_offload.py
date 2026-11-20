"""Fast CPU<->GPU weight offload backed by persistent pinned host copies.

Inference-time offload moves read-only weights.  The legacy path copies every
tensor GPU->CPU on offload into freshly allocated pageable memory (slow D2H,
plus GC / malloc_trim to reclaim the old storage) and copies CPU->GPU from
pageable memory on load (staged, ~2-3 GB/s).

This module keeps one pinned CPU "master" per parameter/buffer:

* load:    ``p.data = master.to(device, non_blocking=True)`` (full PCIe speed)
* offload: ``p.data = master`` -- no copy at all, unless the GPU tensor was
  modified in place since it was loaded (tracked via ``Tensor._version``), in
  which case it is copied back first so the master never goes stale.

Masters are stored in the dtype the module is loaded with (e.g. the VAE is
fp32 on CPU in the legacy path but bf16 on GPU; its bf16 checkpoint values
round-trip exactly, so numerics are unchanged).  Modules containing tensor
subclasses (e.g. torchao quantized weights) are reported as unsupported so the
caller can fall back to the legacy path.  Set ``ACESTEP_PINNED_OFFLOAD=0`` to
disable globally.
"""

from __future__ import annotations

import os
import weakref
from typing import Dict, Iterator, Optional, Tuple

import torch
from torch import nn

_STATE: "weakref.WeakKeyDictionary[nn.Module, Dict[str, object]]" = weakref.WeakKeyDictionary()


def pinned_offload_enabled() -> bool:
    """Return True unless disabled via ``ACESTEP_PINNED_OFFLOAD=0`` or CUDA is unavailable."""
    return os.environ.get("ACESTEP_PINNED_OFFLOAD", "1") != "0" and torch.cuda.is_available()


def _iter_slots(module: nn.Module) -> Iterator[Tuple[nn.Module, str, str, torch.Tensor]]:
    """Yield (owner, kind, name, tensor) for every unique parameter/buffer."""
    seen = set()
    for owner in module.modules():
        for kind, table in (("p", owner._parameters), ("b", owner._buffers)):
            for name, t in table.items():
                if t is None:
                    continue
                key = id(t)
                if key in seen:  # tied / shared tensors are handled once
                    continue
                seen.add(key)
                yield owner, kind, name, t


def _plain(t: torch.Tensor) -> bool:
    data = t.data if isinstance(t, nn.Parameter) else t
    return type(data) is torch.Tensor


def supports(module: nn.Module, device=None) -> bool:
    """Whether the fast path can handle *module* (plain tensors, CUDA target only)."""
    if not pinned_offload_enabled():
        return False
    if device is not None and torch.device(device).type != "cuda":
        return False
    return all(_plain(t) for _, _, _, t in _iter_slots(module))


def _set(owner: nn.Module, kind: str, name: str, t: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Install *value* into the slot and return the tensor whose version counter tracks it.

    ``param.data`` returns a fresh alias with its *own* version counter, so for
    parameters the Parameter object itself must be tracked.
    """
    if kind == "p":
        t.data = value
        return t
    owner._buffers[name] = value
    return value


def load_to_device(module: nn.Module, device, dtype: Optional[torch.dtype] = None) -> None:
    """Move *module* to *device* (casting floating tensors to *dtype*) from pinned masters."""
    # Callers may be inside ``torch.inference_mode()``; inference tensors carry no
    # version counter, so create ordinary (no-grad) tensors here.
    with torch.inference_mode(False), torch.no_grad():
        _load_to_device(module, device, dtype)


def _load_to_device(module: nn.Module, device, dtype: Optional[torch.dtype]) -> None:
    st = _STATE.setdefault(module, {"masters": {}, "versions": {}})
    masters: Dict[int, torch.Tensor] = st["masters"]  # keyed by slot index
    versions: Dict[int, int] = st["versions"]
    versions.clear()
    for i, (owner, kind, name, t) in enumerate(_iter_slots(module)):
        data = t.data if kind == "p" else t
        if data.device.type != "cpu":
            # already resident (e.g. partially moved elsewhere): adopt as-is
            continue
        m = masters.get(i)
        if m is None or m.data_ptr() != data.data_ptr():
            if dtype is not None and data.is_floating_point() and data.dtype != dtype:
                data = data.to(dtype)
            m = data if data.is_pinned() else data.pin_memory()
            masters[i] = m
        elif dtype is not None and m.is_floating_point() and m.dtype != dtype:
            m = m.to(dtype).pin_memory()
            masters[i] = m
        gpu = m.to(device, non_blocking=True)
        tracked = _set(owner, kind, name, t, gpu)
        versions[i] = _version_of(tracked)


def offload_to_cpu(module: nn.Module) -> None:
    """Point *module* back at its pinned masters; copy back only modified tensors."""
    with torch.inference_mode(False), torch.no_grad():
        _offload_to_cpu(module)


def _version_of(t: torch.Tensor) -> Optional[int]:
    try:
        return t._version
    except RuntimeError:  # inference tensor (created elsewhere under inference_mode)
        return None


def _offload_to_cpu(module: nn.Module) -> None:
    st = _STATE.get(module)
    if st is None:
        raise RuntimeError("offload_to_cpu called before load_to_device")
    masters: Dict[int, torch.Tensor] = st["masters"]
    versions: Dict[int, int] = st["versions"]
    for i, (owner, kind, name, t) in enumerate(_iter_slots(module)):
        data = t.data if kind == "p" else t
        if data.device.type == "cpu":
            continue
        current_version = _version_of(t)  # Parameter / buffer object, not the .data alias
        m = masters.get(i)
        if m is None or m.shape != data.shape or m.dtype != data.dtype:
            m = torch.empty(data.shape, dtype=data.dtype, pin_memory=True)
            m.copy_(data)
            masters[i] = m
        elif versions.get(i) is None or versions.get(i) != current_version:
            m.copy_(data)  # weights changed on device (e.g. training): refresh master
        _set(owner, kind, name, t, m)
    versions.clear()
