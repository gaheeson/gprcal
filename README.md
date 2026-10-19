# GPRCal

**Gamma Pass Rate Calculator** — standalone gamma index calculator for DICOM RT Dose files.

GitHub: https://github.com/gaheeson/gprcal

**Algorithm**: Low et al. (1998) Med. Phys. 25:656-661

```
gamma(r_ref) = min over r_eval of:
    sqrt[ (delta_dose / DD)^2 + (|delta_r| / DTA)^2 ]
```

**Dependencies**: `pydicom`, `numpy`, `scipy`, `matplotlib` — no pymedphys, ITK, or OpenCV needed.

한국어 설명: [README_KO.md](README_KO.md)

---

## Installation

```bash
pip install pydicom numpy scipy matplotlib
```

Python 3.8 or later required. tkinter is included in standard Python builds.

> **macOS (Homebrew Python)**: tkinter is not bundled. Install separately:
> ```bash
> brew install python-tk
> ```

---

## GUI (`gprcal_gui.py`)

```bash
python gprcal_gui.py
```

### Layout

```
┌──────────────────────────────────────────────────────┐
│  Input Files                                         │
│    Reference (.dcm):  [path]           [Browse…]     │
│    Evaluation (.dcm): [path]           [Browse…]     │
├──────────────────────────────────────────────────────┤
│  Gamma Criteria                                      │
│    DD (%)  [3.0]   DTA (mm)  [3.0]   Threshold [30] │
│    Sub-voxel interp [10]  □ Local gamma  □ Normalize │
├──────────────────────────────────────────────────────┤
│  Mode                                                │
│    □ 2D mode (fast)    □ Multi-criteria (3/3·2/2…)   │
├──────────────────────────────────────────────────────┤
│  [Run Gamma] [Show Map] [Show Histogram]             │
│  [Save Map (.npy)…]   [Save Figure…]                 │
├──────────────────────────────────────────────────────┤
│  Results                                             │
│    Pass rate (gamma <= 1) : 98.52%  (12304/12494 pts)│
│    Mean gamma             : 0.4123                   │
│    Max gamma              : 1.8847                   │
└──────────────────────────────────────────────────────┘
```

### Workflow

1. **Reference** — Browse and select the reference dose file (e.g. `plan.dcm`)
2. **Evaluation** — Browse and select the evaluation dose file (e.g. `log.dcm`)
3. Set **Gamma Criteria**:
   - `DD (%)`: dose difference criterion (default 3%)
   - `DTA (mm)`: distance-to-agreement (default 3 mm)
   - `Threshold (%)`: reference points below this % of max dose are excluded (default 30%)
   - `Local gamma`: DD relative to local reference dose (unchecked = global max)
   - `Normalize to max`: scale each volume to its own maximum before comparison
4. Select **Mode**:
   - `2D mode`: evaluate middle Z-slice only — fast preview
   - `Multi-criteria`: run 3%/3mm, 2%/2mm, 1%/1mm, 3%/2mm in one go
5. Click **Run Gamma** — progress bar shown; UI stays responsive
6. Use **Show Map** / **Show Histogram** / **Save** buttons after completion

### Buttons

| Button | Action |
|--------|--------|
| **Run Gamma** | Start gamma computation (background thread) |
| **Show Map** | 4-panel figure: reference / evaluation / difference / gamma map |
| **Show Histogram** | Gamma index histogram |
| **Save Map (.npy)** | Save gamma map array as NumPy file |
| **Save Figure** | Save 4-panel figure as PNG / PDF / SVG |

---

## CLI (`gprcal.py`)

```bash
# Default: 3%/3mm/30%, global, 3D
python gprcal.py plan.dcm log.dcm

# Custom criterion
python gprcal.py plan.dcm log.dcm --dd 2 --dta 2 --threshold 10

# Local gamma
python gprcal.py plan.dcm log.dcm --local

# Run all four criteria (3/3, 2/2, 1/1, 3/2)
python gprcal.py plan.dcm log.dcm --multi

# Fast 2D mode (middle Z-slice only)
python gprcal.py plan.dcm log.dcm --2d

# Visualize and save
python gprcal.py plan.dcm log.dcm --plot --hist --save-plot gamma.png
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dd` | 3.0 | Dose difference criterion (%) |
| `--dta` | 3.0 | Distance-to-agreement (mm) |
| `--threshold` | 30.0 | Skip ref points below this % of max dose |
| `--local` | off | Local gamma (DD relative to local ref dose) |
| `--no-norm` | off | Skip normalization (use raw Gy values) |
| `--interp` | 10 | Sub-voxel sampling density within DTA sphere |
| `--multi` | off | Run 3/3, 2/2, 1/1, 3/2 criteria in one run |
| `--2d` | off | 2D gamma on middle Z-slice only (faster) |
| `--save-map` | — | Save gamma map as `.npy` |
| `--plot` | off | Show 4-panel figure |
| `--hist` | off | Show gamma histogram |
| `--save-plot` | — | Save figure as PNG |

---

## Python API

```python
from gprcal import load_rtdose, compute_gamma_3d, compute_gamma_2d

ref = load_rtdose("plan.dcm")
evl = load_rtdose("log.dcm")
print(ref)
# DoseVolume('plan.dcm', shape=(80,256,256), x=[-128.0..126.0] mm, ...)

# 3D gamma — full volume
gamma_map, pass_rate = compute_gamma_3d(
    ref, evl,
    dd_percent=3.0,
    dta_mm=3.0,
    threshold_percent=30.0,
    local_gamma=False,
    normalize=True,
    interp_fraction=10,
)
print(f"Pass rate: {pass_rate:.2f}%")

# 2D gamma — single slice
import numpy as np
sl = ref.shape[0] // 2
gamma_2d, pass_2d = compute_gamma_2d(
    ref.dose[sl], ref.x, ref.y,
    evl.dose[sl], evl.x, evl.y,
    dd_percent=2.0, dta_mm=2.0,
)

# Load a saved gamma map
gmap = np.load("gamma.npy")
valid = gmap[~np.isnan(gmap)]
print(f"Mean gamma: {valid.mean():.3f}, Max: {valid.max():.3f}")
```

---

## Algorithm

### Gamma index (Low 1998)

For each reference point **r** with dose **d_r**:

1. Sample candidate positions **c = r + offset** on a sub-voxel grid within sphere of radius `max_gamma × DTA`
2. Interpolate evaluation dose **d_c** at each candidate (trilinear)
3. Compute: `gamma² = (d_r − d_c)² / DD² + |r − c|² / DTA²`
4. `gamma(r) = sqrt(min gamma²)` → **pass if gamma ≤ 1**

### Normalization (default on)

Both volumes are divided by their own maximum before comparison.
DD is applied as a fraction of that normalized maximum (= 1.0).

### Grid resampling

If the two files have different spacing, size, or origin, the evaluation volume is automatically resampled onto the reference grid (trilinear interpolation).

### Sub-voxel interpolation

`--interp N` sets the DTA sphere sampling step to `DTA / N` mm.
Higher N → more accurate, slower. Recommended ≥ 10 for publication-quality results.

### Performance

| Volume | Criterion | Interp | Approx. time |
|--------|-----------|--------|--------------|
| 100×100×100 | 3%/3mm | 10 | 2–5 min |
| 100×100×100 | 3%/3mm | 5  | 30–90 s |
| 256×256 (2D) | any | 10 | 5–15 s |

Use `--2d` or `--interp 5` for quick checks.

---

## File structure

```
gprcal/
  gprcal.py          # core algorithm + CLI
  gprcal_gui.py      # tkinter GUI
  gprcal_backends.py # computation backends (Pure Python / Numba / pymedphys)
  requirements.txt
  README.md          # this file (English)
  README_KO.md       # Korean version
  LICENSE
  .gitignore
```

---

## References

1. **Low DA, Harms WB, Mutic S, Purdy JA** (1998).
   A technique for the quantitative evaluation of dose distributions.
   *Med Phys* **25**(5):656–661.
   https://doi.org/10.1118/1.598248
   *(Original gamma index algorithm)*

2. **Low DA, Dempsey JF** (2003).
   Evaluation of the gamma dose distribution comparison method.
   *Med Phys* **30**(9):2455–2464.
   https://doi.org/10.1118/1.1598711
   *(Methodological refinement and clinical validation)*

3. **Wendling M, Zijp LJ, McDermott LN, et al.** (2007).
   A fast algorithm for gamma evaluation in 3D.
   *Med Phys* **34**(5):1647–1654.
   https://doi.org/10.1118/1.2721657
   *(Efficient 3D DTA search strategy)*

4. **Miften M, Olch A, Mihailidis D, et al.** (2018).
   Tolerance limits and methodologies for IMRT measurement-based verification QA:
   Recommendations of AAPM Task Group No. 218.
   *Med Phys* **45**(4):e53–e83.
   https://doi.org/10.1002/mp.12810
   *(Clinical acceptance criteria: 3%/2mm ≥ 95%, 3%/3mm ≥ 95%)*

---

## License

MIT
