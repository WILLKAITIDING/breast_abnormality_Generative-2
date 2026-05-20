#!/usr/bin/env python
"""Build ControlNet datalist pairing ODELIA embeddings with preprocessed images.

Reads the existing diffusion model dataset.json and pairs each embedding
with the corresponding preprocessed (intensity-normed + resized) NIfTI volume.
The train/val split is inherited from the diffusion model's dataset.json
(training -> fold 1, testing -> fold 0) to ensure consistent splits.

Output format matches MAISI ControlNet expectations:
    {
        "training": [
            {
                "image": "embeddings_unilateral/UID_SEQ_emb.nii.gz",
                "label": "preprocessed_unilateral/UID_SEQ.nii.gz",
                "fold": 1,
                "dim": [256, 256, 128],
                "spacing": [0.7, 0.7, 0.75],
                "modality": "mri_breast_post"
            }, ...
        ]
    }

Paths are relative to data_base_dir (dataset/ODELIA/).

Usage:
    python prepare_odelia_controlnet_datalist.py \
        --embedding_dir /path/to/embeddings_unilateral \
        --preprocessed_dir /path/to/preprocessed_unilateral \
        --output datalists/controlnet_odelia_datalist.json
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--preprocessed_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_json = os.path.join(args.embedding_dir, "dataset.json")
    with open(dataset_json) as f:
        ds = json.load(f)

    emb_rel = os.path.basename(args.embedding_dir)
    prep_rel = os.path.basename(args.preprocessed_dir)

    entries = []
    skipped = 0

    for split_name in ["training", "testing"]:
        fold = 1 if split_name == "training" else 0
        for item in ds.get(split_name, []):
            stem = item["image"].replace(".nii.gz", "")
            emb_file = f"{stem}_emb.nii.gz"
            prep_file = f"{stem}.nii.gz"

            emb_path = os.path.join(args.embedding_dir, emb_file)
            prep_path = os.path.join(args.preprocessed_dir, prep_file)
            json_path = emb_path + ".json"

            if not os.path.isfile(emb_path) or not os.path.isfile(prep_path):
                skipped += 1
                continue

            with open(json_path) as f:
                meta = json.load(f)

            target_shape = meta.get("target_shape", [256, 256, 128])
            spacing = meta.get("spacing", [0.7, 0.7, 0.75])
            modality = item.get("modality", meta.get("modality", "mri_breast_post"))

            entries.append({
                "image": f"{emb_rel}/{emb_file}",
                "label": f"{prep_rel}/{prep_file}",
                "fold": fold,
                "dim": target_shape,
                "spacing": spacing,
                "modality": modality,
            })

    datalist = {"training": entries}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(datalist, f, indent=4)

    n_train = sum(1 for e in entries if e["fold"] != 0)
    n_val_actual = sum(1 for e in entries if e["fold"] == 0)
    print(f"Created {args.output}")
    print(f"  Total: {len(entries)} (train fold: {n_train}, val fold: {n_val_actual})")
    print(f"  Skipped (missing files): {skipped}")


if __name__ == "__main__":
    main()
