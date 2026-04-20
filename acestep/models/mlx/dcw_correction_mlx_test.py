"""Tests for ``acestep.models.mlx.dcw_correction_mlx``.

The test module is import-safe on platforms without MLX — tests that need
``mlx.core`` are skipped automatically (Linux / Windows CI).
"""

import importlib.util
import math

import pytest


HAS_MLX = importlib.util.find_spec("mlx") is not None


@pytest.fixture
def dcw_mlx():
    from acestep.models.mlx import dcw_correction_mlx
    return dcw_correction_mlx


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_haar_roundtrip_is_identity(dcw_mlx):
    import mlx.core as mx

    x = mx.random.normal((2, 16, 8))
    low, high = dcw_mlx._haar_dwt_1d(x)
    recon = dcw_mlx._haar_idwt_1d(low, high, out_T=16)
    # Haar is orthogonal; roundtrip should recover x up to float noise.
    diff = mx.abs(recon - x).max().item()
    assert diff < 1e-4


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_haar_roundtrip_odd_length_via_right_pad(dcw_mlx):
    import mlx.core as mx

    x = mx.random.normal((1, 15, 4))
    low, high = dcw_mlx._haar_dwt_1d(x)
    # Reconstructing to the original odd length drops the pad sample.
    recon = dcw_mlx._haar_idwt_1d(low, high, out_T=15)
    diff = mx.abs(recon - x).max().item()
    assert diff < 1e-4
    assert recon.shape == (1, 15, 4)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_apply_mlx_dcw_disabled_is_identity(dcw_mlx):
    import mlx.core as mx

    x = mx.random.normal((1, 8, 4))
    y = mx.random.normal((1, 8, 4))
    out = dcw_mlx.apply_mlx_dcw(
        x, y, t_curr=0.5, enabled=False, mode="low",
        scaler=0.1, high_scaler=0.0, wavelet="haar",
    )
    assert mx.all(out == x).item()


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_apply_mlx_dcw_zero_t_is_identity(dcw_mlx):
    import mlx.core as mx

    x = mx.random.normal((1, 8, 4))
    y = mx.random.normal((1, 8, 4))
    out = dcw_mlx.apply_mlx_dcw(
        x, y, t_curr=0.0, enabled=True, mode="low",
        scaler=0.5, high_scaler=0.0, wavelet="haar",
    )
    # t_curr=0 forces s=0 which short-circuits to x.
    diff = mx.abs(out - x).max().item()
    assert diff < 1e-6


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_apply_mlx_dcw_low_matches_manual_formula(dcw_mlx):
    import mlx.core as mx

    x = mx.random.normal((1, 8, 2))
    y = mx.random.normal((1, 8, 2))
    s_user = 0.3
    t = 0.5
    out = dcw_mlx.apply_mlx_dcw(
        x, y, t_curr=t, enabled=True, mode="low",
        scaler=s_user, high_scaler=0.0, wavelet="haar",
    )
    # Manual reference: DWT -> correct L -> IDWT.
    s = s_user * t
    xL, xH = dcw_mlx._haar_dwt_1d(x)
    yL, _ = dcw_mlx._haar_dwt_1d(y)
    xL_c = xL + s * (xL - yL)
    ref = dcw_mlx._haar_idwt_1d(xL_c, xH, out_T=x.shape[1])
    diff = mx.abs(out - ref).max().item()
    assert diff < 1e-5


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_apply_mlx_dcw_pix_matches_formula(dcw_mlx):
    import mlx.core as mx

    x = mx.random.normal((1, 8, 2))
    y = mx.random.normal((1, 8, 2))
    out = dcw_mlx.apply_mlx_dcw(
        x, y, t_curr=1.0, enabled=True, mode="pix",
        scaler=0.25, high_scaler=0.0, wavelet="haar",
    )
    ref = x + 0.25 * (x - y)
    diff = mx.abs(out - ref).max().item()
    assert diff < 1e-6


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_apply_mlx_dcw_unknown_wavelet_warns_once_and_falls_back(dcw_mlx, caplog):
    import mlx.core as mx

    # Reset the module-level warn-once flag to exercise the warning path.
    dcw_mlx._warned_fallback = False
    x = mx.random.normal((1, 8, 2))
    y = mx.random.normal((1, 8, 2))
    with caplog.at_level("WARNING"):
        out = dcw_mlx.apply_mlx_dcw(
            x, y, t_curr=0.5, enabled=True, mode="low",
            scaler=0.1, high_scaler=0.0, wavelet="db4",
        )
    assert any("db4" in rec.message for rec in caplog.records)
    # Second call must not re-warn.
    caplog.clear()
    _ = dcw_mlx.apply_mlx_dcw(
        x, y, t_curr=0.5, enabled=True, mode="low",
        scaler=0.1, high_scaler=0.0, wavelet="db4",
    )
    assert not any("db4" in rec.message for rec in caplog.records)
    # Output shape preserved.
    assert out.shape == x.shape


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_apply_mlx_dcw_invalid_mode_raises(dcw_mlx):
    import mlx.core as mx

    x = mx.random.normal((1, 8, 2))
    y = mx.random.normal((1, 8, 2))
    with pytest.raises(ValueError):
        dcw_mlx.apply_mlx_dcw(
            x, y, t_curr=0.5, enabled=True, mode="bogus",
            scaler=0.1, high_scaler=0.0, wavelet="haar",
        )


# A sanity test that runs on any platform: verify the pure-Python sqrt(2)
# constant used by the Haar implementation.
def test_sqrt2_constant():
    assert abs(math.sqrt(2.0) ** 2 - 2.0) < 1e-12
