"""Native MLX helpers for Differential Correction in Wavelet domain (DCW).

ACE-Step's MLX diffusion loop (``acestep.models.mlx.dit_generate``) runs
entirely on ``mx.array`` objects and we want to keep DCW in-graph on Apple
Silicon — no per-step ``mx ↔ torch`` conversions, no extra dependency on
``pytorch_wavelets``.

We implement a single-level 1-D Haar DWT/IDWT by hand (Haar is the paper's
default basis and is an exact, 2-tap, orthogonal wavelet).  This covers
``dcw_wavelet="haar"`` on MLX.  For richer bases the MLX path logs a
warning and falls back to Haar — anything fancier (db4 / sym8 / ...) would
need longer convolutional filters which we intentionally keep out of
scope for the initial integration.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

logger = logging.getLogger(__name__)

# Cache the warning-once flag so we don't spam the log per-step.
_warned_fallback = False


def _haar_dwt_1d(x):
    """Single-level Haar DWT along the T axis of a ``[B, T, C]`` ``mx.array``.

    Returns ``(low, high)``, each of shape ``[B, T//2, C]``.  If ``T`` is
    odd we zero-pad one sample on the right to mirror ``pytorch_wavelets``'
    ``mode='zero'`` behaviour at the boundary.
    """
    import mlx.core as mx

    T = x.shape[1]
    if T % 2 == 1:
        pad = mx.zeros((x.shape[0], 1, x.shape[2]), dtype=x.dtype)
        x = mx.concatenate([x, pad], axis=1)
    even = x[:, 0::2, :]
    odd = x[:, 1::2, :]
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    low = (even + odd) * inv_sqrt2
    high = (even - odd) * inv_sqrt2
    return low, high


def _haar_idwt_1d(low, high, out_T: int):
    """Inverse of :func:`_haar_dwt_1d`; returns an array of length ``out_T``."""
    import mlx.core as mx

    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    even = (low + high) * inv_sqrt2
    odd = (low - high) * inv_sqrt2
    # Interleave even/odd back along axis 1 -> shape [B, 2*T//2, C].
    stacked = mx.stack([even, odd], axis=2)   # [B, T//2, 2, C]
    reconstructed = stacked.reshape(even.shape[0], -1, even.shape[2])
    return reconstructed[:, :out_T, :]


def apply_mlx_dcw(
    x_next,
    denoised,
    t_curr: float,
    enabled: bool,
    mode: str,
    scaler: float,
    high_scaler: float,
    wavelet: str,
):
    """Apply DCW correction to an MLX latent.

    Args:
        x_next: Post-step latent, shape ``[B, T, C]``.
        denoised: Predicted clean sample ``x_before - v * t``.
        t_curr: Current timestep in ``[0, 1]`` (flow-matching convention).
        enabled: Master switch.
        mode: ``"low"``, ``"high"``, ``"double"`` or ``"pix"``.
        scaler: Low-band strength (also used for ``high``/``pix``).
        high_scaler: High-band strength (used only for ``double``).
        wavelet: PyWavelets basis name.  Only ``"haar"`` is supported on
            MLX right now; anything else falls back to Haar with a
            one-time warning.

    Returns:
        Corrected ``mx.array`` with the same shape and dtype as ``x_next``.
    """
    if not enabled:
        return x_next

    s = float(t_curr) * float(scaler)
    hs = float(t_curr) * float(high_scaler)

    if mode == "double":
        if s == 0.0 and hs == 0.0:
            return x_next
    elif s == 0.0:
        return x_next

    if mode == "pix":
        return x_next + s * (x_next - denoised)

    global _warned_fallback
    if wavelet != "haar" and not _warned_fallback:
        logger.warning(
            "[MLX-DiT] DCW wavelet '%s' is not implemented on the MLX path "
            "(only 'haar' is). Falling back to Haar.",
            wavelet,
        )
        _warned_fallback = True

    T_out = x_next.shape[1]
    xL, xH = _haar_dwt_1d(x_next)
    yL, yH = _haar_dwt_1d(denoised)

    if mode == "low":
        xL = xL + s * (xL - yL)
    elif mode == "high":
        xH = xH + s * (xH - yH)
    elif mode == "double":
        if s != 0.0:
            xL = xL + s * (xL - yL)
        if hs != 0.0:
            xH = xH + hs * (xH - yH)
    else:
        raise ValueError(
            f"Invalid dcw_mode='{mode}' on MLX path. "
            "Expected one of 'low', 'high', 'double', 'pix'."
        )

    return _haar_idwt_1d(xL, xH, T_out)
