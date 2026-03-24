#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEST_BACKEND_DIR = SCRIPT_DIR
SKIP_DIR_NAMES = {"__pycache__"}
SKIP_SUFFIXES = {".pyc"}
SKIP_FILE_NAMES = {"BUCK", "TARGETS"}

COMMON_REPLACEMENTS = (
    ("cortex_m", "nuclei"),
    ("CORTEX_M", "NUCLEI"),
    ("Cortex_M", "Nuclei"),
    ("CortexM", "Nuclei"),
    ("CORTEXM", "NUCLEI"),
    ("cmsis", "nmsis"),
    ("CMSIS", "NMSIS"),
    ("Cmsis", "Nmsis"),
)

OPS_REPLACEMENTS = COMMON_REPLACEMENTS + (
    ("arm", "riscv"),
    ("ARM", "RISCV"),
    ("Arm", "Riscv"),
)

PYTHON_BACKEND_REPLACEMENTS = COMMON_REPLACEMENTS + (
    (
        "executorch.backends.arm.quantizer.arm_quantizer_utils",
        "executorch.backends.nuclei.quantizer.nuclei_quantizer_utils",
    ),
    (
        "executorch.backends.arm.quantizer.quantization_config",
        "executorch.backends.nuclei.quantizer.quantization_config",
    ),
    (
        "executorch.backends.arm.quantizer.quantization_annotator",
        "executorch.backends.nuclei.quantizer.quantization_annotator",
    ),
)

ARM_QUANTIZER_HELPER_REPLACEMENTS = (
    (
        "executorch.backends.arm.quantizer.quantization_config",
        "executorch.backends.nuclei.quantizer.quantization_config",
    ),
    (
        "executorch.backends.arm.quantizer.arm_quantizer_utils",
        "executorch.backends.nuclei.quantizer.nuclei_quantizer_utils",
    ),
    (
        "executorch.backends.arm.quantizer.quantization_annotator",
        "executorch.backends.nuclei.quantizer.quantization_annotator",
    ),
    (
        "from executorch.backends.arm.quantizer import QuantizationConfig",
        "from executorch.backends.nuclei.quantizer.quantization_config import QuantizationConfig",
    ),
    ("from .arm_quantizer_utils import", "from .nuclei_quantizer_utils import"),
)


@dataclass(frozen=True)
class TranslationTarget:
    source_relative_dir: Path
    dest_relative_dir: Path
    replacements: tuple[tuple[str, str], ...]
    allowed_suffixes: frozenset[str] | None = None
    include_files: frozenset[str] | None = None
    rename_map: tuple[tuple[str, str], ...] = ()

    def should_translate(self, path: Path) -> bool:
        if not path.is_file():
            return False
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            return False
        if path.name in SKIP_FILE_NAMES:
            return False
        if path.suffix in SKIP_SUFFIXES:
            return False
        if self.include_files is not None:
            return path.name in self.include_files
        if self.allowed_suffixes is None:
            return True
        return path.suffix in self.allowed_suffixes

    def translate_name(self, name: str) -> str:
        translated = name
        for old, new in self.rename_map:
            translated = translated.replace(old, new)
        for old, new in self.replacements:
            translated = translated.replace(old, new)
        return translated

    def translate_text(self, text: str) -> str:
        for old, new in self.replacements:
            text = text.replace(old, new)
        return text


TRANSLATION_TARGETS = (
    TranslationTarget(
        source_relative_dir=Path("backends/cortex_m/ops"),
        dest_relative_dir=Path("ops"),
        replacements=OPS_REPLACEMENTS,
        allowed_suffixes=frozenset({".cpp", ".h", ".py", ".yaml"}),
    ),
    TranslationTarget(
        source_relative_dir=Path("backends/cortex_m/passes"),
        dest_relative_dir=Path("passes"),
        replacements=PYTHON_BACKEND_REPLACEMENTS,
        allowed_suffixes=frozenset({".py"}),
    ),
    TranslationTarget(
        source_relative_dir=Path("backends/cortex_m/quantizer"),
        dest_relative_dir=Path("quantizer"),
        replacements=PYTHON_BACKEND_REPLACEMENTS,
        allowed_suffixes=frozenset({".py"}),
    ),
    TranslationTarget(
        source_relative_dir=Path("backends/arm/quantizer"),
        dest_relative_dir=Path("quantizer"),
        replacements=ARM_QUANTIZER_HELPER_REPLACEMENTS,
        include_files=frozenset(
            {
                "arm_quantizer_utils.py",
                "quantization_annotator.py",
                "quantization_config.py",
            }
        ),
        rename_map=(("arm_quantizer_utils.py", "nuclei_quantizer_utils.py"),),
    ),
)


def iter_source_files(repo_root: Path) -> Iterable[tuple[TranslationTarget, Path]]:
    for target in TRANSLATION_TARGETS:
        source_dir = repo_root / target.source_relative_dir
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.rglob("*")):
            if target.should_translate(path):
                yield target, path



def translated_relative_path(source_path: Path, repo_root: Path, target: TranslationTarget) -> Path:
    source_dir = repo_root / target.source_relative_dir
    relative_path = source_path.relative_to(source_dir)
    return target.dest_relative_dir / Path(
        *(target.translate_name(part) for part in relative_path.parts)
    )



def translate_file(
    source_path: Path,
    repo_root: Path,
    dest_backend_dir: Path,
    target: TranslationTarget,
    dry_run: bool,
) -> tuple[Path, Path]:
    dest_relative_path = translated_relative_path(source_path, repo_root, target)
    dest_path = dest_backend_dir / dest_relative_path

    translated = target.translate_text(source_path.read_text())
    if not dry_run:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(translated)

    return source_path, dest_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate selected backend files into backends/nuclei, including ops, passes, "
            "quantizer, and required arm quantizer helper files."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"Repository root (default: {REPO_ROOT})",
    )
    parser.add_argument(
        "--dest-backend-dir",
        type=Path,
        default=DEST_BACKEND_DIR,
        help=f"Destination backend directory (default: {DEST_BACKEND_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned translations without writing files.",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dest_backend_dir = args.dest_backend_dir.resolve()

    if not repo_root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {repo_root}")

    translated_count = 0
    for target, source_path in iter_source_files(repo_root):
        _, dest_path = translate_file(
            source_path, repo_root, dest_backend_dir, target, args.dry_run
        )
        translated_count += 1
        print(
            f"{source_path.relative_to(repo_root)} -> "
            f"{dest_path.relative_to(dest_backend_dir)}"
        )

    print(
        ("Planned" if args.dry_run else "Translated")
        + f" {translated_count} files into {dest_backend_dir} from sources under {repo_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
