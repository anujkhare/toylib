# ── H5 dataset inspector & sampler ────────────────────────────────────────────
# Deps: h5py, hdf5plugin (for LZ4/zstd compression), numpy
# !pip install h5py hdf5plugin numpy

import h5py
import hdf5plugin  # registers compression filters — must import before h5py reads
import numpy as np
import random

INPUT_PATH = "/path/to/your/input.h5"
OUTPUT_PATH = "/path/to/output_sample.h5"
N_SAMPLE = 100  # number of records to sample

# ── 1. Inspect ─────────────────────────────────────────────────────────────────
with h5py.File(INPUT_PATH, "r") as f:
    n_frames = int(f.attrs["n_frames"])
    height = int(f.attrs["height"])
    width = int(f.attrs["width"])
    print(f"Records : {n_frames}")
    print(f"Shape   : ({height}, {width}, 3)  uint8")

# ── 2. Sample ──────────────────────────────────────────────────────────────────
assert N_SAMPLE <= n_frames, f"Asked for {N_SAMPLE} but file only has {n_frames}"
indices = sorted(random.sample(range(n_frames), N_SAMPLE))

with h5py.File(INPUT_PATH, "r") as src, h5py.File(OUTPUT_PATH, "w") as dst:
    # Copy the sampled frames one-by-one to avoid loading everything into RAM
    frames_dst = dst.create_dataset(
        "frames",
        shape=(N_SAMPLE, height, width, 3),
        dtype=np.uint8,
        chunks=(1, height, width, 3),
        **hdf5plugin.LZ4(),  # same compression as the source file
    )
    for out_idx, src_idx in enumerate(indices):
        frames_dst[out_idx] = src["frames"][src_idx]

    # Mirror the top-level attrs
    dst.attrs["n_frames"] = N_SAMPLE
    dst.attrs["height"] = height
    dst.attrs["width"] = width

print(f"Wrote {N_SAMPLE} sampled frames → {OUTPUT_PATH}")
