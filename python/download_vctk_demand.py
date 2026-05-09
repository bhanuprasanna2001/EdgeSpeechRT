#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path


FILES = {
    "clean_testset_wav.zip": 1,
    "clean_trainset_28spk_wav.zip": 2,
    "clean_trainset_56spk_wav.zip": 3,
    "logfiles.zip": 4,
    "noisy_testset_wav.zip": 5,
    "noisy_trainset_28spk_wav.zip": 6,
    "noisy_trainset_56spk_wav.zip": 7,
    "testset_txt.zip": 8,
    "trainset_28spk_txt.zip": 9,
    "trainset_56spk_txt.zip": 10,
    "license_text": 11,
}


def download_file(name: str, sequence: int, zip_dir: Path) -> Path:
    output = zip_dir / name
    if output.exists() and output.stat().st_size > 0:
        print(f"exists {output}")
        return output
    url = (
        "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/"
        f"{name}?sequence={sequence}&isAllowed=y"
    )
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--continue-at",
            "-",
            "--retry",
            "3",
            "--retry-delay",
            "5",
            "-o",
            str(output),
            url,
        ],
        check=True,
    )
    return output


def unzip(path: Path, raw_dir: Path) -> None:
    if path.suffix != ".zip":
        return
    marker = raw_dir / f".{path.stem}.extracted"
    if marker.exists():
        print(f"already extracted {path.name}")
        return
    with zipfile.ZipFile(path) as archive:
        archive.extractall(raw_dir)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"extracted {path.name}")


def selected_files(mode: str) -> list[str]:
    if mode == "test":
        return ["clean_testset_wav.zip", "noisy_testset_wav.zip", "testset_txt.zip", "license_text"]
    if mode == "train28":
        return [
            "clean_trainset_28spk_wav.zip",
            "noisy_trainset_28spk_wav.zip",
            "trainset_28spk_txt.zip",
            "license_text",
        ]
    if mode == "all28":
        return [
            "clean_testset_wav.zip",
            "noisy_testset_wav.zip",
            "testset_txt.zip",
            "clean_trainset_28spk_wav.zip",
            "noisy_trainset_28spk_wav.zip",
            "trainset_28spk_txt.zip",
            "license_text",
        ]
    raise ValueError(f"unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoiceBank-DEMAND from Edinburgh DataShare.")
    parser.add_argument("--root", default="datasets/vctk-demand")
    parser.add_argument("--mode", choices=["test", "train28", "all28"], default="test")
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    zip_dir = root / "zips"
    raw_dir = root / "raw"
    zip_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in selected_files(args.mode):
        path = download_file(name, FILES[name], zip_dir)
        if not args.no_extract:
            unzip(path, raw_dir)


if __name__ == "__main__":
    main()
