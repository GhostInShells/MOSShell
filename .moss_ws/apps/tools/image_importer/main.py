"""Image Importer — batch import images into the pil-image resource storage.

Usage:
  GUI:     moss apps test tools/image_importer   (primary)
  MOSS:    moss apps start tools/image_importer
           Then via CTML: <apps.tools_image_importer:import_dir directory="/path" />
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.resources.local_image import (
    LocalImageInfo,
    LocalImageItem,
    LocalImageStorage,
)

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".ico"}

_APP_DIR = Path(__file__).resolve().parent


def _find_workspace_root() -> Path:
    """Walk up to find the workspace root (the directory containing .moss_ws)."""
    d = _APP_DIR
    for _ in range(10):
        if (d / ".moss_ws").is_dir():
            return d
        parent = d.parent
        if parent == d:
            break
        d = parent
    return Path.cwd()


def _assets_dir() -> Path:
    return _find_workspace_root() / ".moss_ws" / "assets" / "pil-images"


def scan_images(directory: Path) -> list[Path]:
    """Scan a directory for supported image files (case-insensitive)."""
    files: list[Path] = []
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    return sorted(files)


async def import_images(
    storage: LocalImageStorage,
    image_paths: list[Path],
    tags: list[str] | None = None,
    on_progress=None,
) -> dict:
    """Import images into the storage.  Returns {imported, skipped, errors}."""
    stats = {"imported": 0, "skipped": 0, "errors": 0}
    tags = tags or []

    # Load full index once for dedup checks (JSONL is sized for a few hundred entries)
    all_infos = await storage.list_infos(limit=-1)
    existing_names = {m.file_name for m in all_infos}

    for i, img_path in enumerate(image_paths):
        name = img_path.name
        stem = img_path.stem

        if name in existing_names:
            stats["skipped"] += 1
            if on_progress:
                on_progress(i, len(image_paths), name, "skipped (already exists)")
            continue

        try:
            meta = LocalImageInfo(
                path=name,
                file_name=name,
                description=stem.replace("_", " ").replace("-", " "),
                tags=tags,
            )
            item = LocalImageItem(meta, str(img_path))
            locator = await storage.put(item)
            existing_names.add(meta.file_name or name)
            stats["imported"] += 1
            if on_progress:
                on_progress(i, len(image_paths), name, f"ok → {locator}")

        except Exception as exc:
            stats["errors"] += 1
            if on_progress:
                on_progress(i, len(image_paths), name, f"error: {exc}")

    return stats


# -- GUI ------------------------------------------------------------------

def main_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Image Importer — MOSS pil-image Storage")
    root.geometry("640x460")
    root.resizable(True, True)

    selected_dir = tk.StringVar()
    host_var = tk.StringVar(value="workspace-assets")
    tags_var = tk.StringVar(value="")
    status_var = tk.StringVar(value="Ready.")
    file_count_var = tk.StringVar(value="")

    def select_directory() -> None:
        path = filedialog.askdirectory(title="Select a directory containing images")
        if not path:
            return
        selected_dir.set(path)
        imgs = scan_images(Path(path))
        file_count_var.set(f"{len(imgs)} image(s) found")

    async def do_import() -> None:
        path = selected_dir.get()
        if not path:
            messagebox.showwarning("Warning", "Select a directory first")
            return

        image_paths = scan_images(Path(path))
        if not image_paths:
            messagebox.showinfo("No images", f"No supported images in:\n{path}")
            return

        progress_bar["maximum"] = len(image_paths)
        import_btn["state"] = "disabled"

        tag_list = [t.strip() for t in tags_var.get().split(",") if t.strip()]
        storage = LocalImageStorage(_assets_dir(), host=host_var.get())

        def update(i: int, total: int, name: str, status: str) -> None:
            progress_bar["value"] = i + 1
            status_var.set(f"[{i + 1}/{total}] {name}: {status}")
            root.update_idletasks()

        try:
            stats = await import_images(storage, image_paths, tags=tag_list, on_progress=update)
            messagebox.showinfo(
                "Done",
                f"Imported: {stats['imported']}\nSkipped: {stats['skipped']}\nErrors: {stats['errors']}",
            )
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
        finally:
            progress_bar["value"] = 0
            status_var.set("Ready.")
            import_btn["state"] = "normal"

    def run_import() -> None:
        asyncio.run(do_import())

    # -- layout --
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Batch Import Images to MOSS Resource Storage",
              font=("", 14, "bold")).pack(pady=(0, 20))

    # directory row
    dir_frame = ttk.Frame(frame)
    dir_frame.pack(fill=tk.X, pady=6)
    ttk.Label(dir_frame, text="Directory  ", width=11).pack(side=tk.LEFT)
    ttk.Entry(dir_frame, textvariable=selected_dir).pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
    ttk.Button(dir_frame, text="Browse...", command=select_directory).pack(side=tk.LEFT)

    ttk.Label(frame, textvariable=file_count_var, foreground="gray").pack(anchor="w", padx=11, pady=(0, 6))

    # host row
    host_frame = ttk.Frame(frame)
    host_frame.pack(fill=tk.X, pady=6)
    ttk.Label(host_frame, text="Host       ", width=11).pack(side=tk.LEFT)
    ttk.Entry(host_frame, textvariable=host_var, width=24).pack(side=tk.LEFT, padx=6)

    # tags row
    tags_frame = ttk.Frame(frame)
    tags_frame.pack(fill=tk.X, pady=6)
    ttk.Label(tags_frame, text="Tags       ", width=11).pack(side=tk.LEFT)
    ttk.Entry(tags_frame, textvariable=tags_var, width=32).pack(side=tk.LEFT, padx=6)
    ttk.Label(tags_frame, text="(comma-separated)", foreground="gray").pack(side=tk.LEFT)

    # progress
    progress_bar = ttk.Progressbar(frame, mode="determinate")
    progress_bar.pack(fill=tk.X, pady=(16, 6))

    ttk.Label(frame, textvariable=status_var, wraplength=580).pack(pady=4)

    # import button
    import_btn = ttk.Button(frame, text="Import Images", command=run_import)
    import_btn.pack(pady=12)

    ttk.Label(frame, text="Supported: PNG, JPG, JPEG, WEBP, GIF, BMP, TIFF, ICO",
              foreground="gray").pack(side=tk.BOTTOM, pady=(8, 0))

    root.mainloop()


# -- MOSS Channel ---------------------------------------------------------

async def _do_import_dir(directory: str, host: str, tags: str) -> str:
    dir_path = Path(directory).resolve()
    if not dir_path.is_dir():
        return f"Error: '{directory}' is not a valid directory"

    image_paths = scan_images(dir_path)
    if not image_paths:
        return f"No supported images found in '{directory}'"

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    storage = LocalImageStorage(_assets_dir(), host=host)

    stats = await import_images(storage, image_paths, tags=tag_list)
    return f"{stats['imported']} imported, {stats['skipped']} skipped, {stats['errors']} errors"


async def main(matrix: Matrix) -> None:
    channel = new_channel(
        name="image_importer",
        description="Batch import images into the pil-image resource storage.",
    )

    @channel.build.command()
    async def import_dir(directory: str, host: str = "workspace-assets", tags: str = "") -> str:
        """Import all images from a directory into pil-image resource storage.

        directory: absolute path to a directory with images
        host:      storage host name (default: workspace-assets)
        tags:      comma-separated tags applied to all imported images
        """
        return await _do_import_dir(directory, host, tags)

    @channel.build.command()
    async def list_images(query: str = "", limit: int = 50) -> str:
        """List images currently in pil-image storage.  query matches description/tags."""
        infos = await matrix.resources().list_infos(
            "pil-image", query=query or None, limit=limit
        )
        if not infos:
            return "No images found."
        lines = [f"{len(infos)} image(s):"]
        for info in infos:
            lines.append(f"  {info.locator}  {info.width}x{info.height}  {info.file_name}")
        return "\n".join(lines)

    await matrix.provide_channel(channel)


# -- entry ----------------------------------------------------------------

if __name__ == "__main__":
    main_gui()
