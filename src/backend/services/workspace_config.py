"""Workspace configuration service for persistent workspace-level settings.

Manages workspace-specific configuration stored in .workspace_config.yaml,
including default calibration paths and other workspace preferences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from common.log_utils import log_debug, log_info, log_warning


WORKSPACE_CONFIG_FILENAME = ".workspace_config.yaml"


@dataclass
class CalibrationConfig:
    """Calibration configuration for workspace defaults."""

    intrinsic_path: Optional[Path] = None  # Absolute path to intrinsic YAML
    extrinsic_path: Optional[Path] = None  # Absolute path to extrinsic YAML
    source_dataset: Optional[str] = None   # Name of dataset these came from (for display)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to YAML-compatible dict."""
        return {
            "intrinsic_path": str(self.intrinsic_path) if self.intrinsic_path else None,
            "extrinsic_path": str(self.extrinsic_path) if self.extrinsic_path else None,
            "source_dataset": self.source_dataset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationConfig":
        """Deserialize from dict."""
        intrinsic = data.get("intrinsic_path")
        extrinsic = data.get("extrinsic_path")
        return cls(
            intrinsic_path=Path(intrinsic) if intrinsic else None,
            extrinsic_path=Path(extrinsic) if extrinsic else None,
            source_dataset=data.get("source_dataset"),
        )

    def is_valid(self) -> bool:
        """Check if calibration paths exist and are valid."""
        if self.intrinsic_path and not self.intrinsic_path.exists():
            return False
        if self.extrinsic_path and not self.extrinsic_path.exists():
            return False
        return self.intrinsic_path is not None or self.extrinsic_path is not None


@dataclass
class WorkspaceConfig:
    """Full workspace configuration."""

    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to YAML-compatible dict."""
        return {
            "version": self.version,
            "calibration": self.calibration.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkspaceConfig":
        """Deserialize from dict."""
        calib_data = data.get("calibration", {})
        return cls(
            version=data.get("version", 1),
            calibration=CalibrationConfig.from_dict(calib_data) if isinstance(calib_data, dict) else CalibrationConfig(),
        )


class WorkspaceConfigService:
    """Service for loading/saving workspace configuration."""

    _instance: Optional["WorkspaceConfigService"] = None
    _workspace_path: Optional[Path] = None
    _config: Optional[WorkspaceConfig] = None

    def __new__(cls) -> "WorkspaceConfigService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def set_workspace(self, workspace_path: Path) -> None:
        """Set the current workspace path and load its configuration."""
        if self._workspace_path == workspace_path:
            return
        self._workspace_path = workspace_path
        self._config = None  # Force reload
        self._load()

    def _get_config_path(self) -> Optional[Path]:
        """Get the path to the workspace config file."""
        if not self._workspace_path:
            return None
        return self._workspace_path / WORKSPACE_CONFIG_FILENAME

    def _load(self) -> None:
        """Load configuration from disk."""
        config_path = self._get_config_path()
        if not config_path or not config_path.exists():
            self._config = WorkspaceConfig()
            log_debug(f"No workspace config found at {config_path}, using defaults", "WORKSPACE_CFG")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._config = WorkspaceConfig.from_dict(data)
            log_info(f"Loaded workspace config from {config_path.name}", "WORKSPACE_CFG")
        except Exception as e:
            log_warning(f"Failed to load workspace config: {e}", "WORKSPACE_CFG")
            self._config = WorkspaceConfig()

    def save(self) -> bool:
        """Save configuration to disk."""
        config_path = self._get_config_path()
        if not config_path or not self._config:
            return False

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self._config.to_dict(), f, default_flow_style=False, sort_keys=False)
            log_info(f"Saved workspace config to {config_path.name}", "WORKSPACE_CFG")
            return True
        except Exception as e:
            log_warning(f"Failed to save workspace config: {e}", "WORKSPACE_CFG")
            return False

    @property
    def config(self) -> WorkspaceConfig:
        """Get the current workspace configuration."""
        if self._config is None:
            self._load()
        return self._config or WorkspaceConfig()

    def set_default_calibration(
        self,
        intrinsic_path: Optional[Path] = None,
        extrinsic_path: Optional[Path] = None,
        source_dataset: Optional[str] = None,
    ) -> bool:
        """Set the default calibration for the workspace.

        Args:
            intrinsic_path: Path to intrinsic calibration YAML
            extrinsic_path: Path to extrinsic calibration YAML
            source_dataset: Name of dataset the calibration came from

        Returns:
            True if saved successfully
        """
        if self._config is None:
            self._load()
        if self._config is None:
            self._config = WorkspaceConfig()

        self._config.calibration = CalibrationConfig(
            intrinsic_path=intrinsic_path,
            extrinsic_path=extrinsic_path,
            source_dataset=source_dataset,
        )
        return self.save()

    def clear_default_calibration(self) -> bool:
        """Clear the default calibration for the workspace."""
        return self.set_default_calibration(None, None, None)

    def get_default_calibration(self) -> Optional[CalibrationConfig]:
        """Get the default calibration if set and valid."""
        calib = self.config.calibration
        if calib.is_valid():
            return calib
        return None

    def get_calibration_for_dataset(self, dataset_path: Path) -> Optional[CalibrationConfig]:
        """Get calibration for a dataset, falling back to workspace default.

        Priority:
        1. Dataset's own calibration files
        2. Workspace default calibration

        Args:
            dataset_path: Path to the dataset

        Returns:
            CalibrationConfig if found, None otherwise
        """
        from config import get_config
        app_config = get_config()

        # Check dataset's own calibration
        intrinsic = dataset_path / app_config.calibration_intrinsic_filename
        extrinsic = dataset_path / app_config.calibration_extrinsic_filename

        if intrinsic.exists() or extrinsic.exists():
            return CalibrationConfig(
                intrinsic_path=intrinsic if intrinsic.exists() else None,
                extrinsic_path=extrinsic if extrinsic.exists() else None,
                source_dataset=dataset_path.name,
            )

        # Fall back to workspace default
        return self.get_default_calibration()


def get_workspace_config_service() -> WorkspaceConfigService:
    """Get the global workspace config service instance."""
    return WorkspaceConfigService()
