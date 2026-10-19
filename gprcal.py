#!/usr/bin/env python3
"""
gprcal.py  —  GPRCal: Gamma Pass Rate Calculator for DICOM RT Dose files

Algorithm: Low et al. (1998) Med. Phys. 25:656-661
  gamma(r_ref) = min over r_eval of:
      sqrt[ (delta_dose / DD)^2  +  (|delta_r| / DTA)^2 ]

Dependencies
------------
    pip install pydicom numpy scipy matplotlib

No pymedphys, ITK, or OpenCV needed.

CLI usage
---------
    # Single criterion
    python gprcal.py plan.dcm log.dcm

    # Custom criterion + threshold
    python gprcal.py plan.dcm log.dcm --dd 2 --dta 2 --threshold 10

    # Run 3%/3mm, 2%/2mm, 1%/1mm, 3%/2mm in one go
    python gprcal.py plan.dcm log.dcm --multi

    # Visualize and save results
    python gprcal.py plan.dcm log.dcm --plot --save-plot gamma.png --save-map gamma.npy

    # Fast 2D mode (middle slice only)
    python gprcal.py plan.dcm log.dcm --2d

Python API
----------
    from gprcal import load_rtdose, compute_gamma_3d, compute_gamma_2d

    ref = load_rtdose("plan.dcm")
    evl = load_rtdose("log.dcm")

    gamma_map, pass_rate = compute_gamma_3d(ref, evl, dd_percent=3, dta_mm=3)
    print(f"Pass rate: {pass_rate:.2f}%")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pydicom
from scipy.interpolate import RegularGridInterpolator


# =============================================================================
# Data container
# =============================================================================

class DoseVolume:
    """3D dose array with physical coordinate axes.

    Attributes
    ----------
    dose : ndarray (nz, ny, nx)  in Gy
    x    : 1D array, length nx  (mm, patient X)
    y    : 1D array, length ny  (mm, patient Y)
    z    : 1D array, length nz  (mm, patient Z / slice positions)
    path : source file path (optional)
    """

    def __init__(self, dose, x, y, z, path=None):
        self.dose = np.asarray(dose, dtype=np.float64)
        self.x    = np.asarray(x,    dtype=np.float64)
        self.y    = np.asarray(y,    dtype=np.float64)
        self.z    = np.asarray(z,    dtype=np.float64)
        self.path = Path(path) if path else None

    @property
    def shape(self):
        return self.dose.shape

    @property
    def max_dose(self):
        return float(self.dose.max())

    def __repr__(self):
        nz, ny, nx = self.shape
        name = self.path.name if self.path else "?"
        return (
            f"DoseVolume('{name}', shape=({nz},{ny},{nx}), "
            f"x=[{self.x[0]:.1f}..{self.x[-1]:.1f}] mm, "
            f"y=[{self.y[0]:.1f}..{self.y[-1]:.1f}] mm, "
            f"z=[{self.z[0]:.1f}..{self.z[-1]:.1f}] mm, "
            f"max={self.max_dose:.4f} Gy)"
        )


# =============================================================================
# DICOM loader
# =============================================================================

def load_rtdose(path: str | Path) -> DoseVolume:
    """Load an RT Dose DICOM file.

    Reads pixel_array, DoseGridScaling, PixelSpacing,
    GridFrameOffsetVector, and ImagePositionPatient.

    Returns
    -------
    DoseVolume  with dose in Gy and coordinates in mm.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ds = pydicom.dcmread(str(path), force=True)

    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    dose    = ds.pixel_array.astype(np.float64) * scaling   # (nz, ny, nx)

    # Pixel spacing: [row spacing = dy, col spacing = dx]
    ps = ds.PixelSpacing
    dy, dx = float(ps[0]), float(ps[1])

    # Slice positions from GridFrameOffsetVector + ImagePositionPatient
    offsets = [float(v) for v in ds.GridFrameOffsetVector]
    origin  = [float(v) for v in ds.ImagePositionPatient]
    ox, oy, oz = origin[0], origin[1], origin[2]

    nz, ny, nx = dose.shape
    x = ox + np.arange(nx) * dx
    y = oy + np.arange(ny) * dy
    z = oz + np.array(offsets)

    return DoseVolume(dose, x, y, z, path=path)


# =============================================================================
# Grid utilities
# =============================================================================

def _grids_match(a: DoseVolume, b: DoseVolume, tol: float = 1e-3) -> bool:
    """Return True if a and b share the same voxel grid."""
    if a.shape != b.shape:
        return False
    return (np.allclose(a.x, b.x, atol=tol) and
            np.allclose(a.y, b.y, atol=tol) and
            np.allclose(a.z, b.z, atol=tol))


def resample_to_ref(ref: DoseVolume, evl: DoseVolume) -> DoseVolume:
    """Resample *evl* onto the coordinate grid of *ref* (trilinear interpolation).

    Points outside the evaluation volume are filled with 0.
    """
    interp = RegularGridInterpolator(
        (evl.z, evl.y, evl.x), evl.dose,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    zz, yy, xx = np.meshgrid(ref.z, ref.y, ref.x, indexing="ij")
    pts = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])
    resampled = interp(pts).reshape(ref.shape)
    return DoseVolume(resampled, ref.x.copy(), ref.y.copy(), ref.z.copy())


# =============================================================================
# Offset table for DTA sphere sampling
# =============================================================================

def _build_offsets_3d(
    dta_mm: float,
    interp_fraction: int,
    max_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return offset vectors and squared distances for a 3D DTA search sphere.

    Offsets are on a regular grid with step = dta_mm / interp_fraction,
    within radius = max_gamma * dta_mm.

    Returns
    -------
    offsets : ndarray (N, 3)  — [dz, dy, dx] in mm
    r2      : ndarray (N,)    — |offset|^2 in mm^2
    """
    step     = dta_mm / interp_fraction
    search_r = max_gamma * dta_mm

    r    = np.arange(-search_r, search_r + step * 0.5, step)
    oz, oy, ox = np.meshgrid(r, r, r, indexing="ij")
    r2   = ox**2 + oy**2 + oz**2
    mask = r2 <= search_r**2

    offsets = np.column_stack([oz[mask].ravel(),
                                oy[mask].ravel(),
                                ox[mask].ravel()])
    # Sort by distance so the inner loop can early-exit (optional future opt)
    order   = np.argsort(r2[mask].ravel())
    return offsets[order], r2[mask].ravel()[order]


def _build_offsets_2d(
    dta_mm: float,
    interp_fraction: int,
    max_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """2D version of _build_offsets_3d."""
    step     = dta_mm / interp_fraction
    search_r = max_gamma * dta_mm

    r      = np.arange(-search_r, search_r + step * 0.5, step)
    oy, ox = np.meshgrid(r, r, indexing="ij")
    r2     = ox**2 + oy**2
    mask   = r2 <= search_r**2

    offsets = np.column_stack([oy[mask].ravel(), ox[mask].ravel()])
    order   = np.argsort(r2[mask].ravel())
    return offsets[order], r2[mask].ravel()[order]


# =============================================================================
# 3D Gamma Index
# =============================================================================

def compute_gamma_3d(
    ref: DoseVolume,
    evl: DoseVolume,
    dd_percent:         float = 3.0,
    dta_mm:             float = 3.0,
    threshold_percent:  float = 30.0,
    local_gamma:        bool  = False,
    normalize:          bool  = True,
    interp_fraction:    int   = 10,
    max_gamma:          float = 2.0,
    batch_size:         int   = 50_000,
    progress:           bool  = True,
    cancel_event=None,
) -> tuple[np.ndarray, float]:
    """Compute 3D gamma index map using the Low 1998 algorithm.

    For each reference point r above the dose threshold:

        gamma(r) = min over candidate positions c of:
            sqrt[ (dose_r - dose_c)^2 / DD^2  +  |r - c|^2 / DTA^2 ]

    Candidate positions are sampled on a sub-voxel grid with step
    = dta_mm / interp_fraction within a sphere of radius max_gamma * dta_mm.
    The evaluation dose at each candidate is obtained by trilinear
    interpolation.  Processing in batches keeps memory bounded.

    Parameters
    ----------
    ref, evl          : DoseVolume — evaluation is resampled to ref grid if needed
    dd_percent        : dose-difference criterion in % of max-ref (global) or
                        local reference dose (local_gamma=True)
    dta_mm            : distance-to-agreement criterion in mm
    threshold_percent : skip reference points below this % of max ref dose
    local_gamma       : True → local DD, False → global DD
    normalize         : True → scale each volume to its own max before comparison
    interp_fraction   : sub-voxel sampling density within DTA sphere (>=10 recommended)
    max_gamma         : maximum gamma to search (caps DTA sphere radius)
    batch_size        : reference points processed per batch (tune for available RAM)
    progress          : print progress to stderr

    Returns
    -------
    gamma_map : ndarray (nz, ny, nx), dtype float32
                NaN at points below threshold or outside evaluation coverage
    pass_rate : float — percentage of valid points with gamma <= 1
    """
    # ── Resample if grids differ ──────────────────────────────────────────────
    if not _grids_match(ref, evl):
        if progress:
            print("  Resampling evaluation volume to reference grid…", flush=True)
        evl = resample_to_ref(ref, evl)

    # ── Working dose arrays ───────────────────────────────────────────────────
    r_dose = ref.dose.copy()
    e_dose = evl.dose.copy()

    if normalize:
        r_dose = r_dose / r_dose.max()
        e_dose = e_dose / e_dose.max()

    # ── Valid reference points above dose threshold ───────────────────────────
    threshold = threshold_percent / 100.0 * r_dose.max()
    iz_v, iy_v, ix_v = np.where(r_dose >= threshold)
    n_valid = len(iz_v)
    if n_valid == 0:
        return np.full(r_dose.shape, np.nan, dtype=np.float32), 0.0

    z_v  = ref.z[iz_v]
    y_v  = ref.y[iy_v]
    x_v  = ref.x[ix_v]
    dr_v = r_dose[iz_v, iy_v, ix_v]

    # ── Interpolator on evaluation grid ──────────────────────────────────────
    eval_interp = RegularGridInterpolator(
        (ref.z, ref.y, ref.x), e_dose,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    # ── DTA search offsets ────────────────────────────────────────────────────
    offsets, r2s = _build_offsets_3d(dta_mm, interp_fraction, max_gamma)
    n_offsets = len(r2s)
    DD = dd_percent / 100.0   # fraction of max dose in normalized space

    if progress:
        print(f"  Valid points: {n_valid:,}   Offsets: {n_offsets:,}   "
              f"Batches: {(n_valid + batch_size - 1) // batch_size}",
              flush=True)

    # ── Batch loop ────────────────────────────────────────────────────────────
    gamma_vals = np.full(n_valid, np.inf, dtype=np.float64)

    for b_start in range(0, n_valid, batch_size):
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Gamma calculation cancelled by user.")
        b_end = min(b_start + batch_size, n_valid)
        bz = z_v[b_start:b_end]
        by = y_v[b_start:b_end]
        bx = x_v[b_start:b_end]
        bd = dr_v[b_start:b_end]

        min_g2 = np.full(len(bz), np.inf, dtype=np.float64)

        for (dz_o, dy_o, dx_o), r2 in zip(offsets, r2s):
            pts  = np.column_stack([bz + dz_o, by + dy_o, bx + dx_o])
            de   = eval_interp(pts)
            ok   = ~np.isnan(de)

            if local_gamma:
                dd_v = DD * np.maximum(bd, 1e-12)
                g2   = ((de - bd) / dd_v) ** 2 + r2 / dta_mm**2
            else:
                g2   = ((de - bd) / DD) ** 2 + r2 / dta_mm**2

            g2[~ok] = np.inf
            np.minimum(min_g2, g2, out=min_g2)

        gamma_vals[b_start:b_end] = np.where(np.isinf(min_g2), np.nan,
                                              np.sqrt(min_g2))

        if progress:
            pct = min(b_end, n_valid) / n_valid * 100
            print(f"  Progress: {pct:5.1f}%", end="\r", flush=True)

    if progress:
        print(flush=True)

    # ── Assemble result map ───────────────────────────────────────────────────
    gamma_map = np.full(r_dose.shape, np.nan, dtype=np.float32)
    gamma_map[iz_v, iy_v, ix_v] = gamma_vals.astype(np.float32)

    valid_g   = gamma_vals[~np.isnan(gamma_vals)]
    pass_rate = (float(np.sum(valid_g <= 1.0) / len(valid_g) * 100)
                 if len(valid_g) else 0.0)
    return gamma_map, pass_rate


# =============================================================================
# 2D Gamma Index  (faster, for single-plane comparison)
# =============================================================================

def compute_gamma_2d(
    ref_dose:          np.ndarray,
    ref_x:             np.ndarray,
    ref_y:             np.ndarray,
    eval_dose:         np.ndarray,
    eval_x:            np.ndarray,
    eval_y:            np.ndarray,
    dd_percent:        float = 3.0,
    dta_mm:            float = 3.0,
    threshold_percent: float = 30.0,
    local_gamma:       bool  = False,
    normalize:         bool  = True,
    interp_fraction:   int   = 10,
    max_gamma:         float = 2.0,
) -> tuple[np.ndarray, float]:
    """Compute 2D gamma index for a single dose plane.

    Parameters
    ----------
    ref_dose  : 2D ndarray (ny, nx)
    ref_x/y   : 1D coordinate arrays in mm
    eval_dose : 2D ndarray (may be on a different grid)
    eval_x/y  : corresponding coordinate arrays
    (other params same as compute_gamma_3d)

    Returns
    -------
    gamma_map : 2D ndarray (ny, nx), NaN below threshold
    pass_rate : float (%)
    """
    r = np.asarray(ref_dose,  dtype=np.float64)
    e = np.asarray(eval_dose, dtype=np.float64)

    if normalize:
        r = r / r.max()
        e = e / e.max()

    # Interpolator on evaluation grid
    eval_interp = RegularGridInterpolator(
        (eval_y, eval_x), e,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    # Valid reference points
    threshold = threshold_percent / 100.0 * r.max()
    iy_v, ix_v = np.where(r >= threshold)
    n_valid = len(iy_v)
    if n_valid == 0:
        return np.full(r.shape, np.nan, dtype=np.float32), 0.0

    y_v  = ref_y[iy_v]
    x_v  = ref_x[ix_v]
    dr_v = r[iy_v, ix_v]

    # 2D DTA offsets
    offsets, r2s = _build_offsets_2d(dta_mm, interp_fraction, max_gamma)
    DD = dd_percent / 100.0

    min_g2 = np.full(n_valid, np.inf, dtype=np.float64)

    for (dy_o, dx_o), r2 in zip(offsets, r2s):
        pts = np.column_stack([y_v + dy_o, x_v + dx_o])
        de  = eval_interp(pts)
        ok  = ~np.isnan(de)

        if local_gamma:
            dd_v = DD * np.maximum(dr_v, 1e-12)
            g2   = ((de - dr_v) / dd_v) ** 2 + r2 / dta_mm**2
        else:
            g2   = ((de - dr_v) / DD) ** 2 + r2 / dta_mm**2

        g2[~ok] = np.inf
        np.minimum(min_g2, g2, out=min_g2)

    gamma_vals = np.where(np.isinf(min_g2), np.nan, np.sqrt(min_g2))

    gamma_map = np.full(r.shape, np.nan, dtype=np.float32)
    gamma_map[iy_v, ix_v] = gamma_vals.astype(np.float32)

    valid_g   = gamma_vals[~np.isnan(gamma_vals)]
    pass_rate = (float(np.sum(valid_g <= 1.0) / len(valid_g) * 100)
                 if len(valid_g) else 0.0)
    return gamma_map, pass_rate


# =============================================================================
# Visualization
# =============================================================================

def plot_results(
    ref:        DoseVolume,
    evl:        DoseVolume,
    gamma_map:  np.ndarray,
    pass_rate:  float,
    dd:         float,
    dta:        float,
    threshold:  float,
    local:      bool = False,
    slice_axis: str  = "z",
    save_path:  str | None = None,
):
    """4-panel plot: reference dose, evaluation dose, difference, gamma map.

    slice_axis : 'z' (axial), 'y' (coronal), 'x' (sagittal)
    """
    import matplotlib.pyplot as plt

    nz, ny, nx = gamma_map.shape
    if slice_axis == "z":
        sl  = nz // 2
        r_s = ref.dose[sl]
        e_s = evl.dose[sl]
        g_s = gamma_map[sl]
        xa, ya   = ref.x, ref.y
        xlbl, ylbl = "X (mm)", "Y (mm)"
    elif slice_axis == "y":
        sl  = ny // 2
        r_s = ref.dose[:, sl, :]
        e_s = evl.dose[:, sl, :]
        g_s = gamma_map[:, sl, :]
        xa, ya   = ref.x, ref.z
        xlbl, ylbl = "X (mm)", "Z (mm)"
    else:  # 'x' / sagittal
        sl  = nx // 2
        r_s = ref.dose[:, :, sl]
        e_s = evl.dose[:, :, sl]
        g_s = gamma_map[:, :, sl]
        xa, ya   = ref.y, ref.z
        xlbl, ylbl = "Y (mm)", "Z (mm)"

    # Normalize for display
    r_max = ref.dose.max()
    e_max = evl.dose.max()
    r_n   = r_s / r_max
    e_n   = e_s / e_max
    diff  = e_n - r_n

    norm_label = "Local" if local else "Global"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                             constrained_layout=True)
    fig.suptitle(
        f"{norm_label} Gamma  {dd}%/{dta}mm/{threshold:.0f}%   "
        f"Pass Rate: {pass_rate:.2f}%",
        fontsize=13, fontweight="bold",
    )

    ext = [xa.min(), xa.max(), ya.max(), ya.min()]
    kw  = dict(extent=ext, aspect="auto")

    im0 = axes[0, 0].imshow(r_n, vmin=0, vmax=1, cmap="jet", **kw)
    axes[0, 0].set_title(f"Reference  (max {r_max:.3f} Gy)")
    fig.colorbar(im0, ax=axes[0, 0]).set_label("Norm. Dose")

    im1 = axes[0, 1].imshow(e_n, vmin=0, vmax=1, cmap="jet", **kw)
    axes[0, 1].set_title(f"Evaluation  (max {e_max:.3f} Gy)")
    fig.colorbar(im1, ax=axes[0, 1]).set_label("Norm. Dose")

    vd  = max(np.abs(diff).max(), 1e-9)
    im2 = axes[1, 0].imshow(diff, vmin=-vd, vmax=vd, cmap="seismic", **kw)
    axes[1, 0].set_title("Dose Difference  (eval − ref, normalized)")
    fig.colorbar(im2, ax=axes[1, 0]).set_label("Delta Dose")

    im3 = axes[1, 1].imshow(
        np.where(np.isnan(g_s), np.nan, g_s),
        vmin=0, vmax=2, cmap="RdYlGn_r", **kw,
    )
    axes[1, 1].set_title("Gamma Map  (green = pass, gamma <= 1)")
    fig.colorbar(im3, ax=axes[1, 1]).set_label("gamma")

    for ax in axes.ravel():
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    plt.show()


def plot_histogram(
    gamma_map: np.ndarray,
    pass_rate: float,
    dd: float,
    dta: float,
    threshold: float,
    max_gamma: float = 2.0,
    local: bool = False,
    save_path: str | None = None,
):
    """Gamma index histogram."""
    import matplotlib.pyplot as plt

    valid = gamma_map[~np.isnan(gamma_map)]
    bins  = np.linspace(0, max_gamma, int(max_gamma * 20) + 1)

    norm_label = "Local" if local else "Global"
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(valid, bins=bins, density=True, color="steelblue", edgecolor="white",
            linewidth=0.4, label="gamma distribution")
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1.5, label="gamma = 1")
    ax.set_xlim(0, max_gamma)
    ax.set_xlabel("Gamma index")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"{norm_label} Gamma  {dd}%/{dta}mm/{threshold:.0f}%  |  "
        f"Pass Rate: {pass_rate:.2f}%  (n={len(valid):,})"
    )
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved histogram: {save_path}")

    plt.show()


# =============================================================================
# CLI
# =============================================================================

_MULTI_CRITERIA = [
    (3.0, 3.0),
    (2.0, 2.0),
    (1.0, 1.0),
    (3.0, 2.0),
]


def _cli():
    ap = argparse.ArgumentParser(
        prog="gprcal",
        description="Gamma Index Calculator for DICOM RT Dose files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("ref",  help="Reference RT Dose DICOM (e.g. plan.dcm)")
    ap.add_argument("eval", help="Evaluation RT Dose DICOM (e.g. log.dcm)")

    ap.add_argument("--dd",        type=float, default=3.0,
                    help="Dose difference criterion (%%)")
    ap.add_argument("--dta",       type=float, default=3.0,
                    help="Distance-to-agreement criterion (mm)")
    ap.add_argument("--threshold", type=float, default=30.0,
                    help="Dose cutoff — skip ref points below this %% of max dose")
    ap.add_argument("--local",     action="store_true",
                    help="Local gamma (DD relative to local ref dose)")
    ap.add_argument("--no-norm",   action="store_true",
                    help="Skip normalization (use raw Gy values)")
    ap.add_argument("--interp",    type=int,   default=10,
                    help="Sub-voxel sampling density within DTA sphere")
    ap.add_argument("--multi",     action="store_true",
                    help="Run 3%%/3mm, 2%%/2mm, 1%%/1mm, 3%%/2mm criteria")
    ap.add_argument("--2d",        dest="mode2d", action="store_true",
                    help="2D gamma on middle Z-slice only (much faster)")
    ap.add_argument("--save-map",  metavar="PATH",
                    help="Save gamma map as .npy  (single-criterion only)")
    ap.add_argument("--plot",      action="store_true",
                    help="Show 4-panel dose/gamma figure")
    ap.add_argument("--hist",      action="store_true",
                    help="Show gamma histogram")
    ap.add_argument("--save-plot", metavar="PATH",
                    help="Save figure as PNG (single-criterion only)")
    args = ap.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading reference : {args.ref}")
    ref = load_rtdose(args.ref)
    print(f"  {ref}")

    print(f"Loading evaluation: {args.eval}")
    evl = load_rtdose(args.eval)
    print(f"  {evl}\n")

    criteria = _MULTI_CRITERIA if args.multi else [(args.dd, args.dta)]
    normalize = not args.no_norm
    last_map  = None
    last_pr   = None

    for dd, dta in criteria:
        label = "Local" if args.local else "Global"
        print(f"{'─'*60}")
        print(f"{label} Gamma  {dd}%/{dta}mm/{args.threshold:.0f}%"
              f"  [interp={args.interp}, {'2D' if args.mode2d else '3D'}]")
        print(f"{'─'*60}")

        if args.mode2d:
            sl = ref.shape[0] // 2
            gmap, pr = compute_gamma_2d(
                ref.dose[sl], ref.x, ref.y,
                evl.dose[sl], evl.x, evl.y,
                dd_percent=dd, dta_mm=dta,
                threshold_percent=args.threshold,
                local_gamma=args.local,
                normalize=normalize,
                interp_fraction=args.interp,
            )
        else:
            gmap, pr = compute_gamma_3d(
                ref, evl,
                dd_percent=dd, dta_mm=dta,
                threshold_percent=args.threshold,
                local_gamma=args.local,
                normalize=normalize,
                interp_fraction=args.interp,
                progress=True,
            )

        valid   = gmap[~np.isnan(gmap)]
        n_pass  = int(np.sum(valid <= 1.0))
        n_total = len(valid)

        print(f"  Pass rate (gamma <= 1) : {pr:.2f}%  ({n_pass}/{n_total} points)")
        print(f"  Mean gamma             : {np.nanmean(gmap):.4f}")
        print(f"  Max gamma              : {np.nanmax(gmap):.4f}")

        last_map = gmap
        last_pr  = pr

    print(f"{'═'*60}")

    # ── Output ────────────────────────────────────────────────────────────────
    if args.save_map and last_map is not None:
        np.save(args.save_map, last_map)
        print(f"Saved gamma map: {args.save_map}")

    if (args.plot or args.save_plot) and last_map is not None:
        dd_l, dta_l = criteria[-1]
        plot_results(ref, evl, last_map, last_pr,
                     dd_l, dta_l, args.threshold, args.local,
                     save_path=args.save_plot)

    if args.hist and last_map is not None:
        dd_l, dta_l = criteria[-1]
        plot_histogram(last_map, last_pr, dd_l, dta_l, args.threshold,
                       local=args.local)


if __name__ == "__main__":
    _cli()
