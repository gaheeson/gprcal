# GPRCal

**Gamma Pass Rate Calculator** — DICOM RT Dose 파일을 위한 감마 인덱스 계산기입니다.

GitHub: https://github.com/gaheeson/gprcal

**알고리즘**: Low et al. (1998) Med. Phys. 25:656-661

```
gamma(r_ref) = min over r_eval:
    sqrt[ (delta_dose / DD)^2 + (|delta_r| / DTA)^2 ]
```

**의존성**: `pydicom`, `numpy`, `scipy`, `matplotlib` — pymedphys, ITK, OpenCV 불필요

English version: [README.md](README.md)

---

## 설치

```bash
pip install pydicom numpy scipy matplotlib
```

Python 3.8 이상 필요. tkinter는 Python 표준 라이브러리에 포함됩니다.

> **macOS (Homebrew Python)**: tkinter가 기본 포함되지 않으므로 별도 설치 필요:
> ```bash
> brew install python-tk
> ```

---

## GUI 사용법 (`gprcal_gui.py`)

```bash
python gprcal_gui.py
```

### 화면 구성

```
┌──────────────────────────────────────────────────────┐
│  Input Files                                         │
│    Reference (.dcm):  [경로]         [Browse…]       │
│    Evaluation (.dcm): [경로]         [Browse…]       │
├──────────────────────────────────────────────────────┤
│  Gamma Criteria                                      │
│    DD (%)  [3.0]   DTA (mm)  [3.0]   Threshold [30] │
│    Sub-voxel interp [10]  □ Local gamma  □ Normalize │
├──────────────────────────────────────────────────────┤
│  Mode                                                │
│    □ 2D mode (빠름)    □ Multi-criteria (3/3·2/2…)   │
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

### 사용 순서

1. **Reference** — Browse로 기준 파일 선택 (계획 선량 `plan.dcm`)
2. **Evaluation** — Browse로 평가 파일 선택 (측정/로그 선량 `log.dcm`)
3. **Gamma Criteria** 설정:
   - `DD (%)`: 선량 차이 기준 (기본값 3%)
   - `DTA (mm)`: 위치 허용 오차 (기본값 3 mm)
   - `Threshold (%)`: 기준 선량의 이 % 미만 포인트는 평가 제외 (기본값 30%)
   - `Local gamma`: 체크 시 로컬 선량 기준 / 미체크 시 글로벌 최대값 기준
   - `Normalize to max`: 각 파일을 자신의 최대값으로 정규화 후 비교
4. **Mode** 선택:
   - `2D mode`: 중간 Z-슬라이스만 평가 — 빠른 미리 확인용
   - `Multi-criteria`: 3%/3mm, 2%/2mm, 1%/1mm, 3%/2mm 일괄 실행
5. **Run Gamma** 클릭 — 백그라운드 스레드로 실행, UI 블로킹 없음
6. 결과 확인 후 **Show Map** / **Show Histogram** / **Save** 사용

### 버튼 설명

| 버튼 | 동작 |
|------|------|
| **Run Gamma** | 감마 계산 실행 |
| **Show Map** | 4-panel 그림 (기준 선량 / 평가 선량 / 차이 / 감마 맵) |
| **Show Histogram** | 감마 인덱스 히스토그램 |
| **Save Map (.npy)** | 감마 맵 배열을 NumPy 파일로 저장 |
| **Save Figure** | 4-panel 그림을 PNG / PDF / SVG로 저장 |

---

## CLI 사용법 (`gprcal.py`)

```bash
# 기본: 3%/3mm/30%, Global, 3D
python gprcal.py plan.dcm log.dcm

# 조건 변경
python gprcal.py plan.dcm log.dcm --dd 2 --dta 2 --threshold 10

# Local gamma
python gprcal.py plan.dcm log.dcm --local

# 네 가지 기준 한 번에 (3/3, 2/2, 1/1, 3/2)
python gprcal.py plan.dcm log.dcm --multi

# 빠른 2D 모드 (중간 Z-슬라이스만)
python gprcal.py plan.dcm log.dcm --2d

# 시각화 및 저장
python gprcal.py plan.dcm log.dcm --plot --hist --save-plot gamma.png
```

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dd` | 3.0 | 선량 차이 기준 (%) |
| `--dta` | 3.0 | 위치 허용 오차 (mm) |
| `--threshold` | 30.0 | 기준 선량의 이 % 미만 포인트 제외 |
| `--local` | off | 로컬 감마 사용 |
| `--no-norm` | off | 정규화 안 함 (원본 Gy 값 사용) |
| `--interp` | 10 | DTA 구 내 서브복셀 샘플링 밀도 |
| `--multi` | off | 3/3, 2/2, 1/1, 3/2 일괄 실행 |
| `--2d` | off | 중간 Z-슬라이스만 평가 (빠름) |
| `--save-map` | — | 감마 맵을 `.npy`로 저장 |
| `--plot` | off | 4-panel 그림 표시 |
| `--hist` | off | 감마 히스토그램 표시 |
| `--save-plot` | — | 그림을 PNG로 저장 |

---

## Python API

```python
from gprcal import load_rtdose, compute_gamma_3d, compute_gamma_2d

ref = load_rtdose("plan.dcm")
evl = load_rtdose("log.dcm")
print(ref)
# DoseVolume('plan.dcm', shape=(80,256,256), x=[-128.0..126.0] mm, ...)

# 3D 감마 (전체 볼륨)
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

# 2D 감마 (단일 슬라이스)
import numpy as np
sl = ref.shape[0] // 2
gamma_2d, pass_2d = compute_gamma_2d(
    ref.dose[sl], ref.x, ref.y,
    evl.dose[sl], evl.x, evl.y,
    dd_percent=2.0, dta_mm=2.0,
)
```

---

## 알고리즘 설명

### 감마 인덱스 (Low 1998)

각 기준 포인트 **r** (선량 d_r)에 대해:

1. DTA 구 안에서 서브복셀 격자로 후보 위치 **c = r + offset** 샘플링
2. 각 후보 위치에서 평가 선량 **d_c** 를 3선형 보간으로 계산
3. `gamma² = (d_r − d_c)² / DD² + |r − c|² / DTA²` 계산
4. `gamma(r) = sqrt(min gamma²)` → **gamma ≤ 1 이면 pass**

### 정규화 (기본 on)

각 볼륨을 자신의 최대값으로 나눠 [0, 1] 범위로 변환 후 비교.
DD 기준은 정규화된 공간에서 적용 (global: DD = dd_percent/100).

### 그리드 리샘플링

두 파일의 복셀 크기나 원점이 다른 경우, 평가 볼륨을 기준 그리드에 자동 리샘플링 (3선형 보간).

### 서브복셀 보간 (`--interp N`)

DTA 구 탐색 스텝 = `DTA / N` mm. N이 클수록 정확하지만 연산 시간 증가.
논문 품질: N ≥ 10 권장. 빠른 확인: N = 5.

### 속도 참고

| 볼륨 크기 | 기준 | interp | 소요 시간 |
|-----------|------|--------|-----------|
| 100×100×100 | 3%/3mm | 10 | 2–5 분 |
| 100×100×100 | 3%/3mm | 5  | 30–90 초 |
| 256×256 (2D) | 임의 | 10 | 5–15 초 |

---

## 파일 구조

```
gprcal/
  gprcal.py          # 핵심 알고리즘 + CLI
  gprcal_gui.py      # tkinter GUI
  gprcal_backends.py # 계산 백엔드 (Pure Python / Numba / pymedphys)
  requirements.txt
  README.md          # 영문 설명서
  README_KO.md       # 한국어 설명서 (이 파일)
  LICENSE
  .gitignore
```

---

## 참고 문헌

1. **Low DA, Harms WB, Mutic S, Purdy JA** (1998).
   A technique for the quantitative evaluation of dose distributions.
   *Med Phys* **25**(5):656–661.
   https://doi.org/10.1118/1.598248
   *(감마 인덱스 원본 알고리즘 제안)*

2. **Low DA, Dempsey JF** (2003).
   Evaluation of the gamma dose distribution comparison method.
   *Med Phys* **30**(9):2455–2464.
   https://doi.org/10.1118/1.1598711
   *(방법론 확장 및 임상 검증)*

3. **Wendling M, Zijp LJ, McDermott LN, et al.** (2007).
   A fast algorithm for gamma evaluation in 3D.
   *Med Phys* **34**(5):1647–1654.
   https://doi.org/10.1118/1.2721657
   *(3D DTA 탐색 고속 알고리즘)*

4. **Miften M, Olch A, Mihailidis D, et al.** (2018).
   Tolerance limits and methodologies for IMRT measurement-based verification QA:
   Recommendations of AAPM Task Group No. 218.
   *Med Phys* **45**(4):e53–e83.
   https://doi.org/10.1002/mp.12810
   *(임상 허용 기준 권고: 3%/2mm ≥ 95%, 3%/3mm ≥ 95%)*

---

## 라이선스

MIT
