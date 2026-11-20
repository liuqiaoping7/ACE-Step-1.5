"""Task-type helpers for Gradio generation events."""


def resolve_no_fsq_task_type(task_type: str, no_fsq: bool) -> str:
    """Resolve the Remix ``no_fsq`` checkbox into the backend task type.

    Args:
        task_type: Current hidden Gradio task type.
        no_fsq: Whether Remix should skip FSQ-quantized source conditioning.

    Returns:
        ``cover-nofsq`` only for Remix/cover with the checkbox enabled;
        otherwise the original task type.
    """
    if task_type == "cover" and no_fsq:
        return "cover-nofsq"
    return task_type


def resolve_chunk_mask_mode(task_type: str) -> str:
    """Resolve model chunk-mask control for a Gradio task.

    Args:
        task_type: Resolved backend task type from the Gradio mode.

    Returns:
        Always ``explicit``: Repaint/Lego intervals reach model conditioning and
        full-song tasks get an all-ones mask (identical to the historical
        output, because ``auto``'s 2.0 used to be clamped to 1.0 by a bool
        tensor).  ``auto`` (2.0) remains available through the API.
    """
    return "explicit"
