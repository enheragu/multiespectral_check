"""Filesystem loader for dataset images, metadata, and trash operations.

Discovers LWIR/visible pairs, caches metadata, and handles moving or restoring files.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from common.yaml_utils import load_yaml, save_yaml


IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
logger = logging.getLogger(__name__)

# Subdirectories that are managed explicitly and must not be treated as companion dirs.
_MANAGED_SUBDIRS = frozenset({"lwir", "visible", "to_delete", "calibration", "labels"})


class DatasetLoader:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.lwir_dir = self.root_dir / "lwir"
        self.vis_dir = self.root_dir / "visible"
        self.to_delete_dir = self.root_dir / "to_delete"
        self.image_bases: List[str] = []
        self.metadata_cache: Dict[str, Dict] = {}
        self.channel_map: Dict[str, Set[str]] = {}
        self._missing_counts_cache: Optional[Dict[str, int]] = None
        self.companion_dirs: List[str] = []

    def _discover_companion_dirs(self) -> List[str]:
        """Return names of extra synchronized data directories (e.g. lidar_range, odom, dht22).

        Any subdirectory that is not in _MANAGED_SUBDIRS and contains at least one
        file is considered a companion dir.  Files in companion dirs follow the same
        base-name convention as lwir/visible images and are moved together on delete.
        """
        found: List[str] = []
        if not self.root_dir.is_dir():
            return found
        for entry in sorted(self.root_dir.iterdir()):
            if not entry.is_dir() or entry.name in _MANAGED_SUBDIRS:
                continue
            if any(entry.iterdir()):  # non-empty directory
                found.append(entry.name)
        return found

    def load_dataset(self) -> bool:
        self.image_bases.clear()
        self.metadata_cache.clear()
        self.channel_map.clear()
        self._missing_counts_cache = None
        self.companion_dirs = self._discover_companion_dirs()

        lwir_files = self._get_image_files(self.lwir_dir)
        vis_files = self._get_image_files(self.vis_dir)

        for file_path in lwir_files:
            base = self._normalize_base(file_path, self.lwir_dir, "lwir")
            self.channel_map.setdefault(base, set()).add("lwir")

        for file_path in vis_files:
            base = self._normalize_base(file_path, self.vis_dir, "visible")
            self.channel_map.setdefault(base, set()).add("visible")

        self.image_bases = sorted(self.channel_map.keys())
        return len(self.image_bases) > 0

    def missing_channel_counts(self) -> Dict[str, int]:
        if self._missing_counts_cache is None:
            counts = {"lwir": 0, "visible": 0}
            for channels in self.channel_map.values():
                if "lwir" not in channels:
                    counts["lwir"] += 1
                if "visible" not in channels:
                    counts["visible"] += 1
            self._missing_counts_cache = counts
        return dict(self._missing_counts_cache)

    def _get_image_files(self, dir_path: Path) -> List[Path]:
        if not dir_path.exists():
            return []
        files: List[Path] = []
        for path in dir_path.rglob("*"):
            if not path.is_file():
                continue
            if "to_delete" in path.parts:
                continue
            if path.suffix.lower() in IMAGE_EXTS:
                files.append(path)
        return files

    def _normalize_base(self, file_path: Path, root_dir: Path, type_dir: str) -> str:
        rel = file_path.relative_to(root_dir)
        name = rel.stem
        if name.startswith(f"{type_dir}_"):
            name = name[len(type_dir) + 1 :]
        if rel.parent == Path("."):
            return name
        return str(rel.parent / name)

    def get_image_path(self, base: str, type_dir: str) -> Optional[Path]:
        img_dir = self.lwir_dir if type_dir == "lwir" else self.vis_dir
        base_path = Path(base)

        # Resolve by stem and accept extension case variants (.jpg/.JPG, etc.).
        def _resolve_by_stem(stem_path: Path) -> Optional[Path]:
            parent = stem_path.parent
            if not parent.exists():
                return None
            pattern = f"{stem_path.name}.*"
            for candidate in parent.glob(pattern):
                if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
                    return candidate
            return None

        # Try relative path first (supports nested dirs)
        resolved = _resolve_by_stem(img_dir / base_path)
        if resolved is not None:
            return resolved

        # Try prefixed filename variants
        prefixed_name = f"{type_dir}_{base_path.name}"
        resolved = _resolve_by_stem(img_dir / base_path.parent / prefixed_name)
        if resolved is not None:
            return resolved

        # Fallback to flat bare names at root
        resolved = _resolve_by_stem(img_dir / f"{type_dir}_{base}")
        if resolved is not None:
            return resolved

        resolved = _resolve_by_stem(img_dir / base)
        if resolved is not None:
            return resolved

        return None


    def get_metadata(self, base: str, type_dir: str) -> Optional[Dict]:
        img_path = self.get_image_path(base, type_dir)
        if not img_path:
            return None

        yaml_path = img_path.with_suffix('.yaml')
        cache_key = str(yaml_path)
        if cache_key in self.metadata_cache:
            return self.metadata_cache[cache_key]

        if yaml_path.exists():
            try:
                data = load_yaml(yaml_path)
                self.metadata_cache[cache_key] = data
                return data  # type: ignore[return-value]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse metadata %s: %s", yaml_path, exc)
        return None

    def delete_entry(self, base: str, reason: str, auto: bool = False) -> bool:
        move_plan: List[Dict[str, Any]] = []  # Usar Any para aceptar Path y bool
        for type_dir in ("lwir", "visible"):
            img_path = self.get_image_path(base, type_dir)
            if not img_path or not img_path.exists():
                continue
            destination = self._compute_to_delete_destination(img_path, type_dir)
            if destination is None:
                return False
            move_plan.append({"src": img_path, "dst": destination, "type": type_dir})
            yaml_path = img_path.with_suffix('.yaml')
            if yaml_path.exists():
                yaml_destination = self._compute_to_delete_destination(yaml_path, type_dir)
                if yaml_destination is None:
                    return False
                move_plan.append({
                    "src": yaml_path,
                    "dst": yaml_destination,
                    "type": type_dir,
                    "yaml": True,
                })

        # Companion dirs: move any file whose stem matches the base (any extension)
        base_stem = Path(base).name
        for companion in self.companion_dirs:
            companion_dir = self.root_dir / companion
            for src in companion_dir.glob(f"{base_stem}.*"):
                if not src.is_file():
                    continue
                destination = self._compute_to_delete_destination(src, companion)
                if destination is None:
                    return False
                move_plan.append({"src": src, "dst": destination, "type": companion})

        if not move_plan:
            return False
        performed: List[Dict[str, Any]] = []
        try:
            for step in move_plan:
                src = step["src"]
                dst = step["dst"]
                cache_key = str(src) if step.get("yaml") else None
                src.rename(dst)
                performed.append(step)
                if cache_key:
                    self.metadata_cache.pop(cache_key, None)
        except OSError as exc:
            logger.error("Failed to move files for %s: %s", base, exc)
            for step in reversed(performed):
                try:
                    step["dst"].rename(step["src"])
                except OSError as rollback_exc:
                    logger.error("Rollback failed for %s: %s", step["dst"], rollback_exc)
            return False
        if base in self.image_bases:
            self.image_bases.remove(base)
        self._missing_counts_cache = None
        self._write_delete_reason(base, reason, [step["dst"] for step in performed], auto=auto)
        return True

    def restore_from_trash(self) -> int:
        lwir_restored = self._restore_category("lwir")
        vis_restored = self._restore_category("visible")
        restored_pairs = min(lwir_restored, vis_restored)

        # Restore companion dirs: any to_delete subdir that is not a managed channel
        _managed_trash = {"lwir", "visible", "reasons"}
        if self.to_delete_dir.is_dir():
            for entry in self.to_delete_dir.iterdir():
                if entry.is_dir() and entry.name not in _managed_trash:
                    self._restore_companion(entry.name)

        if lwir_restored or vis_restored:
            self.load_dataset()
        return restored_pairs

    def count_trash_pairs(self) -> int:
        lwir_count = self._count_trash_images("lwir")
        vis_count = self._count_trash_images("visible")
        return min(lwir_count, vis_count)

    def _compute_to_delete_destination(self, file_path: Path, category: str) -> Optional[Path]:
        target_dir = self.to_delete_dir / category
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create to_delete directory %s: %s", target_dir, exc)
            return None
        destination = target_dir / file_path.name
        return self._unique_destination(destination)

    @staticmethod
    def _unique_destination(initial: Path) -> Path:
        destination = initial
        counter = 1
        while destination.exists():
            destination = initial.with_name(f"{initial.stem}_{counter}{initial.suffix}")
            counter += 1
        return destination

    def _write_delete_reason(self, base: str, reason: str, files: List[Path], auto: bool = False) -> None:
        reasons_dir = self.to_delete_dir / "reasons"
        try:
            reasons_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create reasons dir %s: %s", reasons_dir, exc)
            return
        payload = {
            "base": base,
            "reason": reason,
            "source": "auto" if auto else "user",
            "timestamp": datetime.now().isoformat(),
            "files": [
                {
                    "path": self._relative_or_str(path),
                    "reason": reason,
                }
                for path in files
            ],
        }
        reason_path = reasons_dir / f"{base}.yaml"
        reason_path = self._unique_destination(reason_path)
        try:
            save_yaml(reason_path, payload)
        except OSError as exc:
            logger.error("Failed to write delete reason %s: %s", reason_path, exc)

    def _relative_or_str(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            return str(path)

    def _restore_companion(self, companion: str) -> None:
        """Restore all files from to_delete/{companion}/ back to {companion}/."""
        src_dir = self.to_delete_dir / companion
        target_dir = self.root_dir / companion
        if not src_dir.is_dir():
            return
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create companion dir %s: %s", target_dir, exc)
            return
        for src in src_dir.iterdir():
            if not src.is_file():
                continue
            destination = self._unique_destination(target_dir / src.name)
            try:
                src.rename(destination)
            except OSError as exc:
                logger.error("Failed to restore companion file %s: %s", src, exc)

    def _restore_category(self, category: str) -> int:
        src_dir = self.to_delete_dir / category
        target_dir = self.lwir_dir if category == "lwir" else self.vis_dir
        if not src_dir.exists():
            return 0
        restored = 0
        image_paths = [p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        if not image_paths:
            return 0
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create target dir %s: %s", target_dir, exc)
            return 0
        for image_path in image_paths:
            destination = self._unique_destination(target_dir / image_path.name)
            try:
                image_path.rename(destination)
                restored += 1
            except OSError as exc:
                logger.error("Failed to restore %s: %s", image_path, exc)
                continue
            yaml_name = image_path.with_suffix('.yaml').name
            yaml_path = src_dir / yaml_name
            if yaml_path.exists():
                yaml_destination = destination.with_suffix('.yaml')
                if yaml_destination.exists():
                    yaml_destination = self._unique_destination(yaml_destination)
                try:
                    yaml_path.rename(yaml_destination)
                except OSError as exc:
                    logger.error("Failed to restore metadata %s: %s", yaml_path, exc)
        return restored

    def _count_trash_images(self, category: str) -> int:
        src_dir = self.to_delete_dir / category
        if not src_dir.exists():
            return 0
        return sum(1 for path in src_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
