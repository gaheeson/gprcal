"""
gprcal_backends.py  —  GPRCal: Backend implementations for gamma index computation.

Three backends:
  "Pure Python"  — scipy-based, no extra deps (slow for large 3D volumes)
  "Numba"        — JIT-compiled inner loop, ~50-200x faster (requires: numba)
  "pymedphys"    — pymedphys library (requires: pymedphys)

Use detect_backends() to see what is available on the current system.
"""

from __future__ import annotations

import numpy as np


# =============================================================================
# Backend availability
# =============================================================================

def detect_backends() -> dict[str, bool]:
    """Return a dict of {backend_name: is_available}."""
    result = {"Pure Python": True}

    try:
        import numba  # noqa: F401
        result["Numba"] = True
    except ImportError:
        result["Numba"] = False

    try:
        import pymedphys  # noqa: F401
        result["pymedphys"] = True
    except ImportError:
        result["pymedphys"] = False

    return result


# =============================================================================
# Unified entry point
# =============================================================================

def compute_gamma(
    ref,
    evl,
    dd_percent:        float = 3.0,
    dta_mm:            float = 3.0,
    threshold_percent: float = 30.0,
    local_gamma:       bool  = False,
    normalize:         bool  = True,
    interp_fraction:   int   = 10,
    max_gamma:         float = 2.0,
    backend:           str   = "Pure Python",
    progress_cb=None,
    cancel_event=None,
) -> tuple[np.ndarray, float]:
    """Dispatch to the selected backend.

    Parameters
    ----------
    ref, evl     : DoseVolume (from gprcal.load_rtdose)
    progress_cb  : optional callable(message: str) for status updates
    backend      : "Pure Python" | "Numba" | "pymedphys"

    Returns
    -------
    gamma_map  : ndarray (nz, ny, nx), NaN below threshold
    pass_rate  : float (%)
    """
    if backend == "Numba":
        return _gamma_numba(ref, evl, dd_percent, dta_mm, threshold_percent,
                            local_gamma, normalize, interp_fraction, max_gamma,
                            progress_cb, cancel_event)
    elif backend == "pymedphys":
        return _gamma_pymedphys(ref, evl, dd_percent, dta_mm, threshold_percent,
                                local_gamma, normalize, interp_fraction, max_gamma,
                                progress_cb)
    else:
        return _gamma_pure_python(ref, evl, dd_percent, dta_mm, threshold_percent,
                                  local_gamma, normalize, interp_fraction, max_gamma,
                                  progress_cb, cancel_event)


# =============================================================================
# Backend 1: Pure Python  (wraps gprcal.compute_gamma_3d)
# =============================================================================

def _gamma_pure_python(ref, evl, dd_percent, dta_mm, threshold_percent,
                        local_gamma, normalize, interp_fraction, max_gamma,
                        progress_cb, cancel_event=None):
    from gprcal import compute_gamma_3d

    def _prog(msg):
        if progress_cb:
            progress_cb(msg)

    _prog("Backend: Pure Python\n")
    return compute_gamma_3d(
        ref, evl,
        dd_percent=dd_percent, dta_mm=dta_mm,
        threshold_percent=threshold_percent,
        local_gamma=local_gamma, normalize=normalize,
        interp_fraction=interp_fraction, max_gamma=max_gamma,
        progress=False,
        cancel_event=cancel_event,
    )


# =============================================================================
# Backend 2: Numba  (JIT-compiled, parallel inner loop)
# =============================================================================

_NUMBA_BATCH_SIZE = 50_000


def _gamma_numba(ref, evl, dd_percent, dta_mm, threshold_percent,
                  local_gamma, normalize, interp_fraction, max_gamma,
                  progress_cb, cancel_event=None):
    import numba
    from gprcal import _grids_match, resample_to_ref, _build_offsets_3d

    def _prog(msg):
        if progress_cb:
            progress_cb(msg)

    # ── Resample if needed ────────────────────────────────────────────────────
    if not _grids_match(ref, evl):
        _prog("  Resampling evaluation to reference grid…\n")
        evl = resample_to_ref(ref, evl)

    # ── Working arrays ────────────────────────────────────────────────────────
    r_dose = ref.dose.copy()
    e_dose = evl.dose.copy()

    if normalize:
        r_dose = r_dose / r_dose.max()
        e_dose = e_dose / e_dose.max()

    # ── Threshold ─────────────────────────────────────────────────────────────
    threshold = threshold_percent / 100.0 * r_dose.max()
    iz_v, iy_v, ix_v = np.where(r_dose >= threshold)
    n_valid = len(iz_v)
    if n_valid == 0:
        return np.full(r_dose.shape, np.nan, dtype=np.float32), 0.0

    z_v  = ref.z[iz_v].astype(np.float64)
    y_v  = ref.y[iy_v].astype(np.float64)
    x_v  = ref.x[ix_v].astype(np.float64)
    dr_v = r_dose[iz_v, iy_v, ix_v].astype(np.float64)

    offsets, r2s = _build_offsets_3d(dta_mm, interp_fraction, max_gamma)
    DD = dd_percent / 100.0

    _prog(f"  Backend: Numba  (parallel={numba.config.NUMBA_NUM_THREADS} threads)\n")
    _prog(f"  Valid pts: {n_valid:,}   Offsets: {len(r2s):,}\n")
    _prog("  Compiling JIT (first run ~10-20s, cached after)…\n")

    # ── Batched loop so cancel_event is checked between batches ───────────────
    e_grid = e_dose.astype(np.float64)
    z_ax   = ref.z.astype(np.float64)
    y_ax   = ref.y.astype(np.float64)
    x_ax   = ref.x.astype(np.float64)
    off64  = offsets.astype(np.float64)
    r2s64  = r2s.astype(np.float64)

    gamma_vals = np.full(n_valid, np.nan, dtype=np.float64)

    for b_start in range(0, n_valid, _NUMBA_BATCH_SIZE):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Gamma calculation cancelled by user.")
        b_end = min(b_start + _NUMBA_BATCH_SIZE, n_valid)
        batch = _numba_worker(
            z_v[b_start:b_end], y_v[b_start:b_end],
            x_v[b_start:b_end], dr_v[b_start:b_end],
            e_grid, z_ax, y_ax, x_ax,
            off64, r2s64, DD, dta_mm, local_gamma,
        )
        gamma_vals[b_start:b_end] = batch
        pct = b_end / n_valid * 100
        _prog(f"  Progress: {pct:5.1f}%\r")

    _prog("\n")

    gamma_map = np.full(r_dose.shape, np.nan, dtype=np.float32)
    gamma_map[iz_v, iy_v, ix_v] = gamma_vals.astype(np.float32)

    valid_g   = gamma_vals[~np.isnan(gamma_vals)]
    pass_rate = float(np.sum(valid_g <= 1.0) / len(valid_g) * 100) if len(valid_g) else 0.0
    return gamma_map, pass_rate


def _make_numba_worker():
    """Build and return the numba-JIT-compiled worker (lazy, cached)."""
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True)
    def _trilinear(grid, z_ax, y_ax, x_ax, z, y, x):
        """Trilinear interpolation assuming axes may be non-uniform."""
        nz, ny, nx = grid.shape

        # Clamp to grid bounds — return NaN if outside
        if z < z_ax[0] or z > z_ax[nz - 1]:
            return np.nan
        if y < y_ax[0] or y > y_ax[ny - 1]:
            return np.nan
        if x < x_ax[0] or x > x_ax[nx - 1]:
            return np.nan

        # Binary search for bracketing index
        iz = min(np.searchsorted(z_ax, z) - 1, nz - 2)
        iy = min(np.searchsorted(y_ax, y) - 1, ny - 2)
        ix = min(np.searchsorted(x_ax, x) - 1, nx - 2)
        if iz < 0: iz = 0
        if iy < 0: iy = 0
        if ix < 0: ix = 0

        fz = (z - z_ax[iz]) / (z_ax[iz + 1] - z_ax[iz])
        fy = (y - y_ax[iy]) / (y_ax[iy + 1] - y_ax[iy])
        fx = (x - x_ax[ix]) / (x_ax[ix + 1] - x_ax[ix])

        return (grid[iz,   iy,   ix  ] * (1-fz) * (1-fy) * (1-fx) +
                grid[iz,   iy,   ix+1] * (1-fz) * (1-fy) * fx     +
                grid[iz,   iy+1, ix  ] * (1-fz) * fy     * (1-fx) +
                grid[iz,   iy+1, ix+1] * (1-fz) * fy     * fx     +
                grid[iz+1, iy,   ix  ] * fz     * (1-fy) * (1-fx) +
                grid[iz+1, iy,   ix+1] * fz     * (1-fy) * fx     +
                grid[iz+1, iy+1, ix  ] * fz     * fy     * (1-fx) +
                grid[iz+1, iy+1, ix+1] * fz     * fy     * fx)

    @numba.njit(parallel=True, cache=True)
    def worker(z_v, y_v, x_v, dr_v,
               e_grid, z_ax, y_ax, x_ax,
               offsets, r2s,
               DD, dta_mm, local_gamma):
        """Parallel gamma computation over all valid reference points."""
        n_valid   = len(z_v)
        n_offsets = len(r2s)
        gamma_out = np.empty(n_valid, dtype=np.float64)

        for i in numba.prange(n_valid):
            z0   = z_v[i]
            y0   = y_v[i]
            x0   = x_v[i]
            d_r  = dr_v[i]
            min_g2 = np.inf

            for j in range(n_offsets):
                dz_o = offsets[j, 0]
                dy_o = offsets[j, 1]
                dx_o = offsets[j, 2]

                d_e = _trilinear(e_grid, z_ax, y_ax, x_ax,
                                  z0 + dz_o, y0 + dy_o, x0 + dx_o)
                if np.isnan(d_e):
                    continue

                if local_gamma:
                    denom = DD * max(d_r, 1e-12)
                else:
                    denom = DD

                g2 = ((d_e - d_r) / denom) ** 2 + r2s[j] / dta_mm ** 2
                if g2 < min_g2:
                    min_g2 = g2

            if min_g2 == np.inf:
                gamma_out[i] = np.nan
            else:
                gamma_out[i] = np.sqrt(min_g2)

        return gamma_out

    return worker


# Lazy-initialized worker (compiled on first call)
_numba_worker_cache = None

def _numba_worker(*args, **kwargs):
    global _numba_worker_cache
    if _numba_worker_cache is None:
        _numba_worker_cache = _make_numba_worker()
        if _numba_worker_cache is None:
            raise ImportError("numba is not installed")
    return _numba_worker_cache(*args, **kwargs)


# =============================================================================
# Backend 3: pymedphys
# =============================================================================

def _gamma_pymedphys(ref, evl, dd_percent, dta_mm, threshold_percent,
                      local_gamma, normalize, interp_fraction, max_gamma,
                      progress_cb):
    import pymedphys

    def _prog(msg):
        if progress_cb:
            progress_cb(msg)

    _prog("  Backend: pymedphys\n")

    r_dose = ref.dose.copy().astype(np.float64)
    e_dose = evl.dose.copy().astype(np.float64)

    if normalize:
        r_dose = r_dose / r_dose.max()
        e_dose = e_dose / e_dose.max()

    axes_ref = (ref.z, ref.y, ref.x)
    axes_evl = (evl.z, evl.y, evl.x)

    gamma = pymedphys.gamma(
        axes_ref, r_dose,
        axes_evl, e_dose,
        dose_percent_threshold=dd_percent,
        distance_mm_threshold=dta_mm,
        lower_percent_dose_cutoff=threshold_percent,
        local_gamma=local_gamma,
        interp_fraction=interp_fraction,
        max_gamma=max_gamma,
        quiet=True,
    )

    gamma_map = gamma.astype(np.float32)
    valid     = gamma_map[~np.isnan(gamma_map)]
    pass_rate = float(np.sum(valid <= 1.0) / len(valid) * 100) if len(valid) else 0.0
    return gamma_map, pass_rate
