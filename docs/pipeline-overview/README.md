# Pipeline overview figure

`pipeline-overview.png` (300 dpi, shown in the repository README) summarises the method of the thesis in four stages, numbered in data-flow order: micrographs and annotation, texton learning, superpixel–texton segmentation, and the post-processing that turns a segmentation into phase fractions, pearlite spacing and strength. An unnumbered side panel covers model selection and evaluation.

Only the PNG and the sources in this folder are committed. The build regenerates `assets/` and the vector `pipeline-overview.pdf`, and git ignores both.

## Build

One micromamba environment from conda-forge holds the Python packages and [Tectonic](https://tectonic-typesetting.github.io/), which downloads the LaTeX packages it needs on its first run.

```bash
micromamba create -n smseg-figure -c conda-forge python=3.12 numpy scipy scikit-image scikit-learn opencv pandas matplotlib-base joblib openpyxl prettytable pillow pymupdf tectonic
```

The raw micrograph in stage 1 comes from the thesis PDF, which is not in git either: download the [thesis](https://repositorio.uniandes.edu.co/entities/publication/60ccf854-b041-441b-b7f7-8a79662c4bc1) to `docs/bsc-thesis-spa.pdf` first. Then, from this folder:

```bash
micromamba run -n smseg-figure python make_assets.py
```

```bash
micromamba run -n smseg-figure tectonic pipeline-overview.tex
```

```bash
micromamba run -n smseg-figure python -c "import pymupdf; pymupdf.open('pipeline-overview.pdf')[0].get_pixmap(dpi=300).save('pipeline-overview.png')"
```

Tectonic runs XeTeX, so the figure sets its fonts with `fontspec` and `unicode-math` (TeX Gyre Pagella). `xelatex` or `lualatex` from TeX Live also build it; pdfLaTeX does not.

## Contents

- `make_assets.py` runs the repository's own code (`model.py`, `utils_classes.py`, `utils_functions.py`) on the micrographs stored in `saved_variables/` and writes the thumbnails, the plot data and `assets/numbers.tex`, the values the figure prints.
  - It uses the final hyperparameters reported in the thesis: K = 9 textons per class, and Felzenszwalb superpixels with scale 125, σ 0.8 and min_size 115.
  - It refits the textons with scikit-learn's `KMeans` on the train + dev feature vectors; the thesis used cuML on the GPU.
  - It replaces the GPU-only `cudf`/`cuml` imports and `numba` with empty stand-ins; none of them is used on these code paths.
  - Its asserts check that the annotation windows are located exactly, that the drawn sub-segmentation matches `model.segment`, and that the plotted spectral peak gives the spacing `calculate_spacing` returns.
- `pipeline-overview.tex` draws the figure with TikZ and pgfplots. The annotated example is training micrograph cs0301, and the segmentation example is test micrograph cs0328.
- `assets/raw.jpg` is Fig. 31 of the thesis (micrograph as0014), extracted from `docs/bsc-thesis-spa.pdf`. Without that PDF, the script still writes the other assets but leaves `raw.jpg` as it is, so the figure only builds if an earlier copy exists.

All micrographs come from the ASM Micrograph Database, © ASM International.
