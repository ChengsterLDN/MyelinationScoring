"""
This is just to clean prior pillar folders from previous attempts.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import config


def find_pillar_folders(root_dir: Path) -> list[Path]:
    """Recursively find every folder literally named PILLAR_DIRNAME."""
    name = config.PILLAR_DIRNAME.lower()
    return sorted(p for p in root_dir.rglob("*") if p.is_dir() and p.name.lower() == name)


def unique_destination(used_names: set[str], filename: str, source_label: str) -> str:
    """Return a filename not already in used_names. Tracked in memory, not
    on disk, so --dry-run also catches collisions between planned moves."""
    # No clash: keep the original name
    if filename not in used_names:
        return filename
    # Clash: prefix with the source subfolder label
    candidate = f"{source_label}__{filename}"
    # Still clashing: add a counter until unique
    n = 2
    while candidate in used_names:
        candidate = f"{source_label}__{n}__{filename}"
        n += 1
    return candidate


def flatten_pillar_folder(pillar_dir: Path, dry_run: bool) -> tuple[int, int, int]:
    """Move every matching image under pillar_dir's subfolders up into
    pillar_dir, then remove emptied subfolders.
    Returns (n_moved, n_dirs_removed, n_dirs_left)."""
    # Lower-cased exclusions and extensions for case-insensitive matching
    exclude = {n.lower() for n in config.PILLAR_FLATTEN_EXCLUDE_DIRNAMES}
    extensions = {e.lower() for e in config.PILLAR_FLATTEN_EXTENSIONS}

    # deepest first, so a parent that only contained now-emptied
    # subfolders becomes eligible for removal in the same pass
    subfolders = sorted(
        (p for p in pillar_dir.rglob("*") if p.is_dir()),
        key=lambda p: len(p.relative_to(pillar_dir).parts),
        reverse=True,
    )

    # Names already in pillar_dir; grows as files are (virtually) placed
    used_names = {f.name for f in pillar_dir.iterdir() if f.is_file()}

    # Item counts per subfolder up front, plus files moved out of each, so
    # the "would it be empty" check also works in --dry-run
    initial_count = {sub: sum(1 for _ in sub.iterdir()) for sub in subfolders}
    moved_from: dict[Path, int] = {}

    # --- pass 1: move files ---
    n_moved = 0
    for sub in subfolders:
        rel_parts = sub.relative_to(pillar_dir).parts
        # Skip anything inside an excluded folder
        if any(part.lower() in exclude for part in rel_parts):
            continue
        source_label = "_".join(rel_parts)
        for f in sorted(sub.iterdir()):
            # Only image files with a matching extension
            if not (f.is_file() and f.suffix.lower() in extensions):
                continue
            # Pick a collision-free destination name
            dest_name = unique_destination(used_names, f.name, source_label)
            used_names.add(dest_name)
            dest = pillar_dir / dest_name
            if dest_name != f.name:
                print(f"      ! name collision: {sub.name}/{f.name} -> {dest_name}")
            print(f"      move {f.relative_to(pillar_dir)} -> {dest_name}"
                  f"{' [dry run]' if dry_run else ''}")
            # Only touch the disk outside dry-run mode
            if not dry_run:
                shutil.move(str(f), str(dest))
            moved_from[sub] = moved_from.get(sub, 0) + 1
            n_moved += 1

    # --- pass 2: remove emptied subfolders ---
    n_removed = 0
    n_left = 0
    removed: set[Path] = set()
    for sub in subfolders:
        rel_parts = sub.relative_to(pillar_dir).parts
        if any(part.lower() in exclude for part in rel_parts):
            continue
        # Remaining = original items - files moved out - child folders removed
        # (computed, not re-listed, so --dry-run cascades correctly)
        immediate_child_dirs = [d for d in subfolders
                                 if d.parent == sub]
        removed_children = sum(1 for d in immediate_child_dirs if d in removed)
        remaining = initial_count[sub] - moved_from.get(sub, 0) - removed_children
        # Keep folders that still hold something else
        if remaining > 0:
            n_left += 1
            print(f"      ! not removing {sub.relative_to(pillar_dir)}: "
                  f"{remaining} item(s) would be left in it")
            continue
        print(f"      rmdir {sub.relative_to(pillar_dir)}{' [dry run]' if dry_run else ''}")
        if not dry_run:
            sub.rmdir()
        removed.add(sub)
        n_removed += 1

    return n_moved, n_removed, n_left


def main():
    # Only option: --dry-run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="print what would happen without moving or deleting anything")
    args = parser.parse_args()

    # Find every pillar folder under ROOT_DIR
    root_dir = Path(config.ROOT_DIR)
    print(f"Scanning {root_dir} for '{config.PILLAR_DIRNAME}' folders...")
    pillar_dirs = find_pillar_folders(root_dir)
    if not pillar_dirs:
        print(f"No '{config.PILLAR_DIRNAME}' folders found.")
        return

    print(f"Found {len(pillar_dirs)} pillar folder(s).")
    if args.dry_run:
        print("*** DRY RUN -- nothing will actually be moved or deleted ***")

    # Flatten each pillar folder that has subfolders, keeping totals
    total_moved = total_removed = total_left = 0
    touched_folders = 0
    for pillar_dir in pillar_dirs:
        rel = pillar_dir.relative_to(root_dir)
        if not any(p.is_dir() for p in pillar_dir.iterdir()):
            continue  # nothing nested here, skip quietly
        print(f"\n--- {rel} ---")
        n_moved, n_removed, n_left = flatten_pillar_folder(pillar_dir, args.dry_run)
        print(f"    moved {n_moved} file(s), removed {n_removed} subfolder(s), "
              f"left {n_left} non-empty subfolder(s)")
        total_moved += n_moved
        total_removed += n_removed
        total_left += n_left
        touched_folders += 1

    # Final summary
    print(f"\n{'=' * 50}")
    print("FLATTEN COMPLETE" + (" (DRY RUN)" if args.dry_run else ""))
    print(f"{'=' * 50}")
    print(f"Pillar folders with nested subfolders : {touched_folders}")
    print(f"Files moved                           : {total_moved}")
    print(f"Subfolders removed                    : {total_removed}")
    print(f"Subfolders left (not empty)            : {total_left}")


if __name__ == "__main__":
    main()
