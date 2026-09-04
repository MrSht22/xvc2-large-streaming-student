from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import re
import sys
from typing import Any


REQUIREMENTS = {
    "PyYAML": ((6, 0), None),
    "torch": ((2, 4), None),
    "torchaudio": ((2, 4), None),
    "transformers": ((4, 44), (5, 0)),
}


def version_tuple(value: str) -> tuple[int, ...]:
    numbers = re.match(r"^(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in numbers.group(1).split(".")) if numbers else ()


def check_environment(require_cuda: bool = False) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    failures: list[str] = []
    for distribution, (minimum, maximum) in REQUIREMENTS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
            failures.append(f"missing:{distribution}")
            continue
        packages[distribution] = version
        parsed = version_tuple(version)
        if parsed < minimum or (maximum is not None and parsed >= maximum):
            failures.append(f"version:{distribution}={version}")

    cuda = {"available": False, "device_count": 0}
    if all(packages.get(name) is not None for name in ("torch", "torchaudio", "transformers")):
        try:
            torch = importlib.import_module("torch")
            torchaudio = importlib.import_module("torchaudio")
            from torchaudio.models import Emformer
            from transformers import Wav2Vec2Config
            from transformers.models.wav2vec2.modeling_wav2vec2 import (
                Wav2Vec2FeatureEncoder,
            )

            if version_tuple(torch.__version__)[:2] != version_tuple(torchaudio.__version__)[:2]:
                failures.append("torch_torchaudio_minor_version_mismatch")
            Emformer(
                input_dim=8,
                num_heads=2,
                ffn_dim=16,
                num_layers=1,
                segment_length=2,
                left_context_length=2,
                right_context_length=1,
            )
            Wav2Vec2FeatureEncoder(
                Wav2Vec2Config(
                    conv_dim=[8],
                    conv_kernel=[4],
                    conv_stride=[2],
                    conv_bias=True,
                    feat_extract_norm="layer",
                )
            )
            cuda = {
                "available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
            }
        except Exception as error:  # pragma: no cover - depends on installed binaries
            failures.append(f"api:{type(error).__name__}:{error}")
    if require_cuda and not cuda["available"]:
        failures.append("cuda_unavailable")
    return {
        "python": sys.version.split()[0],
        "packages": packages,
        "cuda": cuda,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Student training environment")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    report = check_environment(args.require_cuda)
    print(json.dumps(report, sort_keys=True))
    print(f"ctc_gop_student_environment={report['status']}")
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
