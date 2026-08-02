import pandas as pd
from tqdm import tqdm

from scripts.dataset_class import SimpleDataset
from scripts.preprocessing import Preprocessor
from scripts.metrics import calc_ncc

# --------------------------
# Конфигурация
# --------------------------

config = {
    "normalization": True,
    "channel_mode": "hed",
    "flip_intensity": False,
    "match_histogram": False,
    "clahe": False,
    "smoothing": True,
    "metric": "ncc",
}

csv_path = "train_classification_renamed_columns_2.csv"

dataset = SimpleDataset(
    csv_path,
    dist_col="Similarity_Score",
    class_col="class3",
    return_meta=True,
)

preprocessor = Preprocessor(device="cpu")

scores = []

for item in tqdm(dataset):
    f_patch, w_patch, dist, target, meta = dataset.unpack_item(item)

    inp = preprocessor.process(f_patch, w_patch, config)

    score = calc_lpips(
        inp.src_np,
        inp.trg_np,
        mask_np=None,
    )

    scores.append(score)

# --------------------------
# Сохраняем
# --------------------------

df = dataset.annotations.copy()
df["ncc_hed"] = scores

df.to_csv("train_classification_with_ncc.csv", index=False)

print(df[["patch_id", "ncc_hed"]].head())