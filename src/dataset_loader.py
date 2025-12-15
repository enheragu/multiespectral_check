import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
logger = logging.getLogger(__name__)

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
        
    def load_dataset(self) -> bool:
        self.image_bases.clear()
        self.metadata_cache.clear()
        self.channel_map.clear()
        self._missing_counts_cache = None

        lwir_files = self._get_image_files(self.lwir_dir)
        vis_files = self._get_image_files(self.vis_dir)

        for file_path in lwir_files:
            base = file_path.stem.replace(f"{file_path.parent.name}_", "")
            self.channel_map.setdefault(base, set()).add("lwir")

        for file_path in vis_files:
            base = file_path.stem.replace(f"{file_path.parent.name}_", "")
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
        return [p for p in dir_path.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    
    def get_image_path(self, base: str, type_dir: str) -> Optional[Path]:
        img_dir = self.lwir_dir if type_dir == "lwir" else self.vis_dir
        for ext in IMAGE_EXTS:
            candidate = img_dir / f"{type_dir}_{base}{ext}"
            if candidate.exists():
                return candidate
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
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self.metadata_cache[cache_key] = data
                return data
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse metadata %s: %s", yaml_path, exc)
        return None

    def delete_entry(self, base: str, reason: str) -> bool:
        move_plan: List[Dict[str, Path]] = []
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
        if not move_plan:
            return False
        performed: List[Dict[str, Path]] = []
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
        self._write_delete_reason(base, reason, [step["dst"] for step in performed])
        return True

    def restore_from_trash(self) -> int:
        restored_pairs = 0
        lwir_restored = self._restore_category("lwir")
        vis_restored = self._restore_category("visible")
        restored_pairs = min(lwir_restored, vis_restored)
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

    def _write_delete_reason(self, base: str, reason: str, files: List[Path]) -> None:
        reasons_dir = self.to_delete_dir / "reasons"
        try:
            reasons_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create reasons dir %s: %s", reasons_dir, exc)
            return
        payload = {
            "base": base,
            "reason": reason,
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
            with open(reason_path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, allow_unicode=False)
        except OSError as exc:
            logger.error("Failed to write delete reason %s: %s", reason_path, exc)

    def _relative_or_str(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root_dir))
        except ValueError:
            return str(path)

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
