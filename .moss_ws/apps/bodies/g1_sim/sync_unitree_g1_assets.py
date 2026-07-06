from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


RAW_BASE = "https://raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main"
SCENE_URL = f"{RAW_BASE}/resources/robots/g1_description/scene.xml"
MODEL_URL = f"{RAW_BASE}/resources/robots/g1_description/g1_12dof.xml"
POLICY_URL = f"{RAW_BASE}/deploy/pre_train/g1/motion.pt"
CHUNK_SIZE = 1024 * 256
MAX_RETRIES = 3
RETRY_DELAY_SEC = 1.5


def _format_size(size: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{size}B"


@dataclass(slots=True)
class DownloadItem:
    url: str
    dest: Path


@dataclass(slots=True)
class DownloadStats:
    total_files: int
    processed_files: int = 0
    downloaded_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    downloaded_bytes: int = 0


def _print_overall_progress(stats: DownloadStats) -> None:
    percent = (stats.processed_files * 100.0 / stats.total_files) if stats.total_files else 100.0
    print(
        f"overall: {stats.processed_files}/{stats.total_files} "
        f"({percent:6.2f}%) | downloaded={stats.downloaded_files} "
        f"skipped={stats.skipped_files} failed={stats.failed_files} "
        f"bytes={_format_size(stats.downloaded_bytes)}"
    )


def download_bytes(url: str, label: str) -> bytes:
    with urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        chunks: list[bytes] = []
        print(f"downloading {label}")
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)
            if total > 0:
                percent = downloaded * 100.0 / total
                sys.stdout.write(
                    f"\r  progress: {percent:6.2f}% ({_format_size(downloaded)}/{_format_size(total)})"
                )
            else:
                sys.stdout.write(f"\r  progress: {_format_size(downloaded)}")
            sys.stdout.flush()
        if downloaded:
            sys.stdout.write("\n")
        return b"".join(chunks)


def download_file(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = download_bytes(url, str(dest))
    dest.write_bytes(data)
    print(f"saved {dest}")
    return len(data)


def ensure_file(item: DownloadItem, stats: DownloadStats, retries: int) -> bytes:
    stats.processed_files += 1
    prefix = f"[{stats.processed_files}/{stats.total_files}]"
    if item.dest.exists() and item.dest.stat().st_size > 0:
        stats.skipped_files += 1
        print(f"{prefix} skip existing {item.dest}")
        _print_overall_progress(stats)
        return item.dest.read_bytes()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"{prefix} start {item.dest} (attempt {attempt}/{retries})")
            downloaded_size = download_file(item.url, item.dest)
            stats.downloaded_files += 1
            stats.downloaded_bytes += downloaded_size
            _print_overall_progress(stats)
            return item.dest.read_bytes()
        except (OSError, URLError, TimeoutError) as exc:
            last_error = exc
            print(f"retry {attempt}/{retries} failed for {item.dest}: {exc}")
            if attempt < retries:
                time.sleep(RETRY_DELAY_SEC)
            if item.dest.exists() and item.dest.stat().st_size == 0:
                item.dest.unlink()

    stats.failed_files += 1
    _print_overall_progress(stats)
    raise RuntimeError(f"failed to download {item.dest} after {retries} attempts") from last_error


def parse_mesh_names(model_xml: str) -> list[str]:
    return sorted(set(re.findall(r'file="([^"]+\.STL)"', model_xml)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Unitree G1 MuJoCo M1 assets into local g1_sim app")
    parser.add_argument("--skip-policy", action="store_true", help="Do not download motion.pt")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES, help="Retry count for each file download")
    args = parser.parse_args()

    app_dir = Path(__file__).resolve().parent
    g1_dir = app_dir / "assets" / "g1"
    meshes_dir = g1_dir / "meshes"
    policy_path = app_dir / "assets" / "policies" / "g1_motion.pt"

    g1_dir.mkdir(parents=True, exist_ok=True)
    model_item = DownloadItem(MODEL_URL, g1_dir / "g1_12dof.xml")
    stats = DownloadStats(total_files=2 + (0 if args.skip_policy else 1))
    print("preparing download manifest")
    model_xml = ensure_file(model_item, stats, max(1, args.retries))
    mesh_names = parse_mesh_names(model_xml.decode("utf-8"))
    stats.total_files += len(mesh_names)
    print(f"download plan: {stats.total_files} files")

    ensure_file(DownloadItem(SCENE_URL, g1_dir / "scene.xml"), stats, max(1, args.retries))
    for mesh_name in mesh_names:
        ensure_file(
            DownloadItem(f"{RAW_BASE}/resources/robots/g1_description/meshes/{mesh_name}", meshes_dir / mesh_name),
            stats,
            max(1, args.retries),
        )

    if not args.skip_policy:
        ensure_file(DownloadItem(POLICY_URL, policy_path), stats, max(1, args.retries))

    print("done")
    _print_overall_progress(stats)


if __name__ == "__main__":
    main()
