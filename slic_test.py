import matplotlib.pyplot as plt

from utils_classes import SLICSegmentation
from utils_functions import find_path_of_micrograph, load_img

SRC = "C:\\Users\\Camilo Martínez\\Google Drive"
LABELED = "Anotadas"
PATH_LABELED = f"{SRC}\\{LABELED}"

n_segments = 500
sigma = 5
compactness = 0.1

n_segments_test_size = 5
n_segments_increase_percentage = 0.1
compactness_test_size = 6
compactness_increase = 0.025

img_name = "cs0327.png"

print("Loading test image... ", end="")
test_img = load_img(
    find_path_of_micrograph(img_name, PATH_LABELED), as_255=False, with_io=True
)
print("Done")

fig, ax = plt.subplots(n_segments_test_size, compactness_test_size)
dpi = 600
fig.set_size_inches(18, 12)
fig.set_dpi(dpi)
for x, N in enumerate(
    [
        n_segments + n_segments * i * (1 + n_segments_increase_percentage)
        for i in range(n_segments_test_size)
    ]
):
    for y, C in enumerate(
        [compactness + j * compactness_increase for j in range(compactness_test_size)]
    ):
        print(f"Segmenting {x, y}... ", end="")
        slic_model = SLICSegmentation(n_segments=N, sigma=sigma, compactness=C)
        segments = slic_model.segment(test_img)
        ax[x, y] = slic_model.plot_output(test_img, segments, dpi=dpi, ax=ax[x, y])
        ax[x, y].set_title(f"N = {round(N)}, C = {round(C, 2)}")
        print("Done")

plt.tight_layout()
plt.savefig("SLIC Test.png", dpi=dpi)
