"""Fast CPU<->GPU weight offload backed by persistent pinned host copies.

Inference-time offload moves read-only weights.  The legacy path copies every
tensor GPU->CPU on offload into freshly allocated pageable memory (slow D2H,
plus GC / malloc_trim to reclaim the old storage) and copies CPU->GPU from
pageable memory on load (staged, ~2-3 GB/s).

This module keeps one pinned CPU "master" per parameter/buffer:

* load:    ``p.data = master.to(device, non_blocking=True)`` (full PCIe speed)
* offload: ``p.data = master`` -- no copy at all, unless the GPU tensor was
  modified in place since it was loaded (tracked via the Parameter's
  ``_version``), in which case it is copied back first so the master never
  goes stale.

Masters are attached to the tensors' owners (the Parameter object itself, or a
dict on the owning module for buffers), so loading a submodule and later the
whole model share the same masters.  ``prefetch_to_device`` runs the H2D copy
on a side stream; ``wait_prefetch`` must be called before the weights are used
(``load_to_device`` / ``offload_to_cpu`` do it automatically).

Masters are stored in the dtype the module is loaded with (e.g. the VAE is
fp32 on CPU in the legacy path but bf16 on GPU; its bf16 checkpoint values
round-trip exactly, so numerics are unchanged).  Modules containing tensor
subclasses (e.g. torchao quantized weights) are reported as unsupported so the
caller can fall back to the legacy path.  Set ``ACESTEP_PINNED_OFFLOAD=0`` to
disable globally.
"""

from __future__ import annotations

import os
from typing import Iterator, List, Optional, Tuple

import torch
from torch import nn

_MASTER = "_acestep_pinned_master"
_VERSION = "_acestep_loaded_version"
_BUF_MASTERS = "_acestep_pinned_buffer_masters"
_BUF_VERSIONS = "_acestep_pinned_buffer_versions"
_PREFETCH = "_acestep_prefetch"


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


def module_bytes(module: nn.Module, only_on_cpu: bool = False) -> int:
    """Total bytes of the module's tensors (optionally only those still on CPU)."""
    total = 0
    for _, kind, _, t in _iter_slots(module):
        data = t.data if kind == "p" else t
        if only_on_cpu and data.device.type != "cpu":
            continue
        total += data.numel() * data.element_size()
    return total


def _version_of(t: torch.Tensor) -> Optional[int]:
    try:
        return t._version
    except RuntimeError:  # inference tensor (created elsewhere under inference_mode)
        return None


def _get_master(owner: nn.Module, kind: str, name: str, t: torch.Tensor) -> Optional[torch.Tensor]:
    if kind == "p":
        return getattr(t, _MASTER, None)
    return getattr(owner, _BUF_MASTERS, {}).get(name)


def _set_master(owner: nn.Module, kind: str, name: str, t: torch.Tensor, m: torch.Tensor) -> None:
    if kind == "p":
        setattr(t, _MASTER, m)
    else:
        if _BUF_MASTERS not in owner.__dict__:
            owner.__dict__[_BUF_MASTERS] = {}
        owner.__dict__[_BUF_MASTERS][name] = m


def _record_loaded_version(owner: nn.Module, kind: str, name: str, tracked: torch.Tensor) -> None:
    if kind == "p":
        setattr(tracked, _VERSION, _version_of(tracked))
    else:
        owner.__dict__.setdefault(_BUF_VERSIONS, {})[name] = _version_of(tracked)


def _loaded_version(owner: nn.Module, kind: str, name: str, t: torch.Tensor) -> Optional[int]:
    if kind == "p":
        return getattr(t, _VERSION, None)
    return owner.__dict__.get(_BUF_VERSIONS, {}).get(name)


def _install(owner: nn.Module, kind: str, name: str, t: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Install *value* into the slot; return the tensor whose version counter tracks it.

    ``param.data`` returns a fresh alias with its *own* version counter, so for
    parameters the Parameter object itself must be tracked.
    """
    if kind == "p":
        t.data = value
        return t
    owner._buffers[name] = value
    return value


def _load_slots(module: nn.Module, device, dtype: Optional[torch.dtype]) -> List[torch.Tensor]:
    loaded = []
    for owner, kind, name, t in _iter_slots(module):
        data = t.data if kind == "p" else t
        if data.device.type != "cpu":
            continue  # already resident (e.g. prefetched): adopt as-is
        m = _get_master(owner, kind, name, t)
        if m is None or m.data_ptr() != data.data_ptr():
            if dtype is not None and data.is_floating_point() and data.dtype != dtype:
                data = data.to(dtype)
            m = data if data.is_pinned() else data.pin_memory()
            _set_master(owner, kind, name, t, m)
        elif dtype is not None and m.is_floating_point() and m.dtype != dtype:
            m = m.to(dtype).pin_memory()
            _set_master(owner, kind, name, t, m)
        gpu = m.to(device, non_blocking=True)
        tracked = _install(owner, kind, name, t, gpu)
        _record_loaded_version(owner, kind, name, tracked)
        loaded.append(gpu)
    return loaded


def load_to_device(module: nn.Module, device, dtype: Optional[torch.dtype] = None) -> None:
    """Move *module* to *device* (casting floating tensors to *dtype*) from pinned masters."""
    # Callers may be inside ``torch.inference_mode()``; inference tensors carry no
    # version counter, so create ordinary (no-grad) tensors here.
    with torch.inference_mode(False), torch.no_grad():
        wait_prefetch(module)
        _load_slots(module, device, dtype)


def prefetch_to_device(module: nn.Module, device, dtype: Optional[torch.dtype] = None) -> None:
    """Start an asynchronous H2D load of *module* on a side stream."""
    device = torch.device(device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.inference_mode(False), torch.no_grad(), torch.cuda.stream(stream):
        loaded = _load_slots(module, device, dtype)
        event = torch.cuda.Event()
        event.record(stream)
    module.__dict__[_PREFETCH] = (device, event, loaded)


def is_prefetched(module: nn.Module) -> bool:
    return any(m.__dict__.get(_PREFETCH) is not None for m in module.modules())


def wait_prefetch(module: nn.Module) -> None:
    """Make the current stream wait for any pending prefetch of *module* or its submodules."""
    for m in module.modules():
        pending = m.__dict__.get(_PREFETCH)
        if pending is None:
            continue
        device, event, loaded = pending
        cur = torch.cuda.current_stream(device)
        cur.wait_event(event)
        for t in loaded:  # allocated on the side stream, used and freed on the current one
            t.record_stream(cur)
        m.__dict__[_PREFETCH] = None


def offload_to_cpu(module: nn.Module) -> None:
    """Point *module* back at its pinned masters; copy back only modified tensors."""
    with torch.inference_mode(False), torch.no_grad():
        wait_prefetch(module)
        for owner, kind, name, t in _iter_slots(module):
            data = t.data if kind == "p" else t
            if data.device.type == "cpu":
                continue
            m = _get_master(owner, kind, name, t)
            loaded_version = _loaded_version(owner, kind, name, t)
            if m is None or m.shape != data.shape or m.dtype != data.dtype:
                m = torch.empty(data.shape, dtype=data.dtype, pin_memory=True)
                m.copy_(data)
                _set_master(owner, kind, name, t, m)
            elif loaded_version is None or loaded_version != _version_of(t):
                m.copy_(data)  # weights changed on device (e.g. training): refresh master
            _install(owner, kind, name, t, m)
