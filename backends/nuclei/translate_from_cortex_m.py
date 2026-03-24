#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = SCRIPT_DIR.parent / "cortex_m" / "ops"
DEST_DIR = SCRIPT_DIR / "ops"
FILE_SUFFIXES = {".cpp", ".h"}
REPLACEMENTS = (
    ("cmsis", "nmsis"),
    ("CMSIS", "NMSIS"),
    ("cortex_m", "nuclei"),
    ("arm", "riscv"),
    ("ARM", "RISCV"),
)


def translate_text(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def translate_name(name: str) -> str:
    return translate_text(name)


def iter_source_files(source_dir: Path):
    yield from sorted(
        path for path in source_dir.rglob("*") if path.is_file() and path.suffix in FILE_SUFFIXES
    )


def translate_file(source_path: Path, source_dir: Path, dest_dir: Path, dry_run: bool) -> tuple[Path, Path]:
    relative_path = source_path.relative_to(source_dir)
    dest_relative_path = Path(
        *[translate_name(part) for part in relative_path.parts[:-1]],
        translate_name(relative_path.name),
    )
    dest_path = dest_dir / dest_relative_path

    translated = translate_text(source_path.read_text())
    if not dry_run:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(translated)

    return source_path, dest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate backends/cortex_m/ops .cpp/.h files into "
            "backends/nuclei/ops, replacing cmsis->nmsis and arm->riscv."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help=f"Source ops directory (default: {SOURCE_DIR})",
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=DEST_DIR,
        help=f"Destination ops directory (default: {DEST_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned translations without writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    dest_dir = args.dest_dir.resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    translated_count = 0
    for source_path in iter_source_files(source_dir):
        _, dest_path = translate_file(source_path, source_dir, dest_dir, args.dry_run)
        translated_count += 1
        print(f"{source_path.relative_to(source_dir)} -> {dest_path.relative_to(dest_dir)}")

    print(
        (
            "Planned" if args.dry_run else "Translated"
        )
        + f" {translated_count} files from {source_dir} to {dest_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
