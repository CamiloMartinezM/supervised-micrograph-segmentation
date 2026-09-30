"""Generate the images and data drawn by pipeline-overview.tex.

Every thumbnail comes from the repository's own code (model.py, utils_classes.py,
utils_functions.py) run on the micrographs stored in saved_variables/, with the final
hyperparameters reported in the thesis: K = 9 textons per class and Felzenszwalb
superpixels with scale = 125, sigma = 0.8, min_size = 115. The thesis fitted the textons
with cuML's GPU K-means; here scikit-learn's KMeans fits them on the same train + dev
feature vectors. The raw micrograph with its scale bar is Fig. 31 of the thesis.

Run from any folder:  python make_assets.py
"""
import os
import sys
import types

import cv2
import numpy as np
import pymupdf
from PIL import Image
from scipy.ndimage import find_objects, gaussian_filter
from skimage.segmentation import find_boundaries, mark_boundaries
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ASSETS = os.path.join(HERE, "assets")

TRAIN_IMG = "cs0301.png"  # annotated training micrograph (stage 1)
TEST_IMG = "cs0328.png"  # test micrograph followed through inference (stages 3-4)
K = 9
FELZENSZWALB = (125, 0.8, 115)  # scale, sigma, min_size
PF, PEARLITE, FERRITE = 0, 1, 2  # class indices in model.segment; cementite keeps PEARLITE
COLOURS = {PF: (52, 101, 214), PEARLITE: (214, 69, 65), FERRITE: (242, 201, 46)}
THUMB = 360  # px, side of the square thumbnails


# The repo's modules import the GPU-only RAPIDS packages (cudf, cuml), numba and pycm at
# module level, but none of them is used on the code paths below. Empty stand-ins let
# the modules import on a CPU-only machine; numba.jit becomes a no-op because the code
# relied on numba's (since removed) object-mode fallback.
def _stub(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module


def _dummy(name):
    return type(name, (), {})


_stub("cudf", DataFrame=_dummy("DataFrame"), Series=_dummy("Series"))
_stub("cuml", KMeans=_dummy("KMeans"))
_stub("cuml.metrics")
_stub("cuml.metrics.cluster")
_stub("cuml.metrics.cluster.entropy", cython_entropy=None)
_stub("numba", jit=lambda f: f)
_stub("numba.core")
_stub("numba.core.errors", NumbaWarning=Warning)
_stub("pycm", ConfusionMatrix=_dummy("ConfusionMatrix"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, REPO)

import model  # noqa: E402
import utils_functions as uf  # noqa: E402
from utils_classes import FilterBank, SuperpixelSegmentation  # noqa: E402

# numpy >= 2 refuses to write sub_segment's fill_value=-1 into the uint8 binary image
# (numpy 1.x wrapped it to 255). Both values fall outside {0, 1}, which is all
# sub_segment maps, so a signed copy of the binary image gives the same result.
_img_to_binary = uf.img_to_binary
model.img_to_binary = lambda img: _img_to_binary(img).astype(np.int16)


def save(name, pixels, size=THUMB):
    """Save a [0, 1] grey image or a uint8 RGB image as a square thumbnail."""
    if pixels.dtype != np.uint8:
        pixels = (255 * np.clip(pixels, 0, 1)).astype(np.uint8)
    Image.fromarray(pixels).resize((size, size), Image.LANCZOS).save(os.path.join(ASSETS, name))


def overlay(gray, classes, alpha=0.55):
    """Colour a class map over a grey micrograph, as model.visualize_segmentation does."""
    rgb = np.repeat(255 * gray[..., None], 3, axis=-1)
    for k, colour in COLOURS.items():
        rgb[classes == k] = (1 - alpha) * rgb[classes == k] + alpha * np.array(colour)
    return rgb.astype(np.uint8)


def outline(rgb, mask, colour=(255, 255, 255)):
    """Draw a 2 px outline of mask onto an RGB image."""
    edge = cv2.dilate(find_boundaries(mask, mode="outer").astype(np.uint8), np.ones((2, 2)))
    rgb = rgb.copy()
    rgb[edge > 0] = colour
    return rgb


def box_area(box):
    return (box[0].stop - box[0].start) * (box[1].stop - box[1].start)


def normalise(x, low=1, high=99):
    lo, hi = np.percentile(x, [low, high])
    return np.clip((x - lo) / (hi - lo), 0, 1)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    load = lambda name: uf.load_variable_from_file(name, os.path.join(REPO, "saved_variables"))
    images = {}
    for split in ("training", "development", "testing"):
        images.update(load(f"{split}_set_imgs_as_numpy_arrays"))
    windows = load("full_windows_data")  # {label: [(filename, window)]}, all 51 images
    features = load("full_feature_vectors")  # {label: (N, 8)} MR8 vectors, train + dev
    classes = np.array(list(features))
    assert classes.tolist() == ["proeutectoid ferrite", "pearlite"], classes  # app.py order
    tex = {}

    # Stage 1: raw ASM micrograph (thesis Fig. 31). The thesis stretched it horizontally;
    # its red 500 x 500 crop box is square in the original, which restores the aspect.
    # The thesis PDF is not in git; without it, raw.jpg is left as it is.
    thesis_path = os.path.join(REPO, "docs", "bsc-thesis-spa.pdf")
    if os.path.exists(thesis_path):
        thesis = pymupdf.open(thesis_path)
        pix = pymupdf.Pixmap(thesis, thesis[61].get_images()[0][0])
        raw = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3]
        assert raw.shape[:2] == (363, 775), raw.shape
        red = (raw[..., 0] > 180) & (raw[..., 1] < 80) & (raw[..., 2] < 80)
        rows, cols = np.nonzero(red)
        aspect = (np.ptp(cols) + 1) / (np.ptp(rows) + 1)
        height = round(480 / raw.shape[1] * raw.shape[0] * aspect)
        Image.fromarray(raw).resize((480, height)).save(os.path.join(ASSETS, "raw.jpg"), quality=92)
    else:
        print(f"{thesis_path} not found; assets/raw.jpg (thesis Fig. 31) is not regenerated")
    save("crop.png", images["as0014.png"])  # the same micrograph after preprocessing

    # Stage 1: annotated windows. Each window is a slice of its micrograph; template
    # matching recovers where it was drawn, and the match must be exact and unique.
    train = images[TRAIN_IMG]
    save("train.png", train)
    H, W = train.shape
    boxes, crops = [], {PF: [], PEARLITE: []}
    for k, label in enumerate(classes):
        for filename, window in windows[label]:
            if filename != TRAIN_IMG:
                continue
            ssd = cv2.matchTemplate(
                train.astype(np.float32), window.astype(np.float32), cv2.TM_SQDIFF
            )
            assert ssd.min() < 1e-2 and (ssd < 1e-2).sum() == 1, (label, ssd.min())
            y, x = np.unravel_index(ssd.argmin(), ssd.shape)
            h, w = window.shape
            # x0/y0/x1/y1 in [0, 1] with y up, then the class index
            boxes.append(f"{x/W:.4f}/{1-(y+h)/H:.4f}/{(x+w)/W:.4f}/{1-y/H:.4f}/{k}")
            crops[k].append(window)
    tex["smTrainWindows"] = ",".join(boxes)
    for k, name in ((PF, "pf"), (PEARLITE, "p")):
        for i, window in enumerate(sorted(crops[k], key=lambda a: -a.size)[:3]):
            side = min(window.shape)
            y0, x0 = (window.shape[0] - side) // 2, (window.shape[1] - side) // 2
            save(f"win_{name}{i + 1}.png", window[y0 : y0 + side, x0 : x0 + side], size=120)

    # Stage 2: the 38 RFS filters behind MR8 (edge and bar at 3 scales x 6 orientations,
    # then the Gaussian and Laplacian of Gaussian), each scaled so that 0 is mid-grey.
    bank = FilterBank("MR8")
    rows_of_filters = list(bank.edge) + list(bank.bar) + [bank.rot[:, 0]]
    tile, gap = bank.edge.shape[-1], 6
    grid = np.full((7 * (tile + gap) - gap, 6 * (tile + gap) - gap), 255, np.uint8)
    for r, filters in enumerate(rows_of_filters):
        for c, f in enumerate(filters):
            y, x = r * (tile + gap), c * (tile + gap)
            grid[y : y + tile, x : x + tile] = 127.5 * (1 + f / np.abs(f).max())
    Image.fromarray(grid).save(os.path.join(ASSETS, "filters.png"))

    # Stage 2: per-class K-means textons, T of shape (C, K, 8), shown in a 2-D PCA view.
    kmeans = lambda X: KMeans(n_clusters=K, n_init=10, random_state=0).fit(X)
    T = np.stack([kmeans(features[c]).cluster_centers_ for c in classes])
    X = np.concatenate([features[c] for c in classes])
    mean = X.mean(axis=0)
    axes = np.linalg.eigh(np.cov((X - mean).T))[1][:, ::-1][:, :2]
    rng = np.random.default_rng(0)
    with open(os.path.join(ASSETS, "textons.dat"), "w") as points, open(
        os.path.join(ASSETS, "centres.dat"), "w"
    ) as centres:
        points.write("x y c\n")
        centres.write("x y c\n")
        for k, label in enumerate(classes):
            sample = features[label][rng.choice(len(features[label]), 350, replace=False)]
            for x, y in (sample - mean) @ axes:
                points.write(f"{x:.3f} {y:.3f} {k}\n")
            for x, y in (T[k] - mean) @ axes:
                centres.write(f"{x:.3f} {y:.3f} {k}\n")

    # Stage 3: superpixels, per-pixel texton votes, superpixel decision, sub-segmentation.
    img = images[TEST_IMG]
    save("test.png", img)
    segments = SuperpixelSegmentation("felzenszwalb", FELZENSZWALB).segment(img)
    tex["smSuperpixels"] = str(len(np.unique(segments)))
    responses = model.get_response_vector(img)  # (H, W, 8)
    for i in (1, 4, 8):  # the three drawn in the stack: edge and bar at the finest scale, LoG
        save(f"resp{i}.png", normalise(responses[..., i - 1]), size=240)
    x = responses.reshape(-1, 8)
    t = T.reshape(-1, 8)
    distances = (x**2).sum(1)[:, None] - 2 * x @ t.T + (t**2).sum(1)[None]
    votes = (distances.argmin(axis=1) // K).reshape(img.shape)  # class of the nearest texton

    _, two_class, _, _ = model.segment(img, classes, T, "felzenszwalb", FELZENSZWALB)
    _, final, _, _ = model.segment(
        img, classes, T, "felzenszwalb", FELZENSZWALB, subsegment_class=("pearlite", "ferrite")
    )
    img_255 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_64F).astype(np.uint8)
    binary = _img_to_binary(img_255)  # Laplacian sharpening + Otsu: 0 dark, 1 bright
    assert np.array_equal(final, np.where((two_class == PEARLITE) & (binary == 1), FERRITE, two_class))

    # One compact, mixed superpixel, outlined in every panel, illustrates the collective
    # decision: the one nearest the image centre among the compact, mixed candidates.
    labels = segments.ravel()
    sizes = np.bincount(labels)
    share = np.bincount(labels, weights=(votes == PEARLITE).ravel()) / np.maximum(sizes, 1)
    fill = np.array([sizes[i] / box_area(b) for i, b in enumerate(find_objects(segments + 1))])
    rows, cols = np.indices(img.shape)
    radius = np.hypot(rows - img.shape[0] / 2, cols - img.shape[1] / 2).ravel()
    distance = np.bincount(labels, weights=radius) / np.maximum(sizes, 1)
    candidates = np.nonzero((sizes > 1000) & (share > 0.6) & (share < 0.9) & (fill > 0.45))[0]
    chosen = candidates[np.argmin(distance[candidates])]
    mask = segments == chosen
    assert two_class[mask].min() == two_class[mask].max() == PEARLITE
    tex["smVotePearlite"] = f"{100 * share[chosen]:.0f}"
    tex["smVotePixels"] = str(sizes[chosen])

    # Magnified votes around that superpixel, for the inset; the box is given in [0, 1]
    # thumbnail coordinates with y up.
    half = 40
    rows, cols = np.nonzero(mask)
    cy = int(np.clip(rows.mean(), half, img.shape[0] - half))
    cx = int(np.clip(cols.mean(), half, img.shape[1] - half))
    zoom = outline(overlay(img, votes), mask)[cy - half : cy + half, cx - half : cx + half]
    zoom = Image.fromarray(zoom).resize((320, 320), Image.NEAREST)
    zoom.save(os.path.join(ASSETS, "votes_zoom.png"))
    tex["smZoomX"], tex["smZoomY"] = f"{cx / img.shape[1]:.4f}", f"{1 - cy / img.shape[0]:.4f}"
    tex["smZoomSize"] = f"{2 * half / img.shape[1]:.4f}"

    gray_rgb = np.repeat(255 * img[..., None], 3, axis=-1).astype(np.uint8)
    boundaries = (255 * mark_boundaries(gray_rgb, segments, color=(0, 0.82, 1))).astype(np.uint8)
    save("superpixels.png", outline(boundaries, mask))
    save("votes.png", outline(overlay(img, votes), mask))
    save("twoclass.png", outline(overlay(img, two_class), mask))
    save("binary.png", np.where(two_class == PEARLITE, binary, 0.55))
    save("final.png", overlay(img, final))
    fractions = np.bincount(final.ravel(), minlength=3) / final.size * 100
    tex["smFracPF"], tex["smFracCementite"], tex["smFracFerrite"] = (f"{v:.1f}" for v in fractions)
    tex["smFracPearlite"] = f"{fractions[PEARLITE] + fractions[FERRITE]:.1f}"

    # Stage 4: spectral spacing (Matteis 2020) on the cementite pixels, as
    # model.calculate_interlamellar_spacing does. calculate_spacing only returns the two
    # spacings, so peakpos is wrapped to read the curve it fits.
    cementite = uf.highlight_class_in_img(img, final, class_=PEARLITE)
    save("cementite.png", cementite)
    # Display only: a light blur shows the ring at the lamellar wavenumber through the speckle.
    spectrum = gaussian_filter(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(cementite)))), 2)
    c = img.shape[0] // 2
    save("dft.png", normalise(spectrum[c - 160 : c + 160, c - 160 : c + 160], 2, 99.9), size=240)
    fits = []
    peakpos = uf.peakpos
    uf.peakpos = lambda X, Y: fits.append((X, Y, peakpos(X, Y))) or fits[-1][2]
    spacing1, spacing2 = uf.calculate_spacing(cementite)
    uf.peakpos = peakpos
    wavenumber, magnitude, (peak, (amplitude, _, width), fwhm) = fits[-1]
    assert np.isclose(spacing1, 1 / peak)
    with open(os.path.join(ASSETS, "spectrum.dat"), "w") as f:  # scaled so the fit peaks at 1
        f.write("w F\n")
        f.writelines(
            f"{w:.5f} {m / amplitude:.5f}\n" for w, m in zip(wavenumber, magnitude) if w <= 0.7
        )
    tex["smPeak"], tex["smFitWidth"], tex["smFWHM"] = (f"{v:.4f}" for v in (peak, width, fwhm))
    tex["smSpacingOne"], tex["smSpacingTwo"] = f"{spacing1:.2f}", f"{spacing2:.2f}"

    with open(os.path.join(ASSETS, "numbers.tex"), "w") as f:
        f.write("% Written by make_assets.py; do not edit.\n")
        f.writelines(f"\\def\\{key}{{{value}}}\n" for key, value in tex.items())
    print("\n".join(f"{key} = {value}" for key, value in tex.items() if key != "smTrainWindows"))


if __name__ == "__main__":
    main()
