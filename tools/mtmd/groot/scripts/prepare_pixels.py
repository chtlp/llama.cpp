#!/usr/bin/env python3
"""Reference ViT input for the case, produced with OpenCV's INTER_AREA (the
baseline's resize) and the same patch layout. Used to feed the baseline runtime
on the board, where OpenCV is not installed."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def bf16_as_half(x):
    f = np.asarray(x, np.float32).copy()
    u = f.view(np.uint32)
    u += np.uint32(0x7FFF) + ((u >> 16) & 1)
    u &= np.uint32(0xFFFF0000)
    return f.astype(np.float16)


def patch_view(image):
    crop = int(256 * 0.95)
    first = (256 - crop) // 2
    resized = cv2.resize(image[first:first + crop, first:first + crop], (256, 256),
                         interpolation=cv2.INTER_AREA)
    lut = bf16_as_half((np.arange(256, dtype=np.float32) * np.float32(1 / 255) - np.float32(0.5))
                       / np.float32(0.5))
    out = np.empty((256, 1536), np.float16)
    for gh in range(8):
        for gw in range(8):
            for mh in range(2):
                for mw in range(2):
                    patch = (gh * 8 + gw) * 4 + mh * 2 + mw
                    base = resized[(gh * 2 + mh) * 16:(gh * 2 + mh + 1) * 16,
                                   (gw * 2 + mw) * 16:(gw * 2 + mw + 1) * 16]
                    flat = base.reshape(256, 3)  # row-major 16x16 pixels
                    for channel in range(3):
                        values = lut[flat[:, channel]]
                        out[patch, channel * 512:channel * 512 + 256] = values
                        out[patch, channel * 512 + 256:channel * 512 + 512] = values
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    front = np.fromfile(a.case / "front.u8", np.uint8).reshape(256, 256, 3)
    wrist = np.fromfile(a.case / "wrist.u8", np.uint8).reshape(256, 256, 3)
    out = np.concatenate([patch_view(front), patch_view(wrist)], axis=0)
    a.out.write_bytes(out.tobytes())
    print(f"wrote {a.out} {out.shape}")


if __name__ == "__main__":
    main()
