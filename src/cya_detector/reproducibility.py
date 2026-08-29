"""Run metadata helpers used by all future training entry points."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def git_commit(repo_root: Path) -> str:
    """Return the current Git commit, or ``unknown`` outside a checkout."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def package_versions(distributions: Iterable[str]) -> dict[str, str]:
    """Return installed versions without importing heavyweight packages."""

    versions: dict[str, str] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def collect_run_metadata(
    *,
    config: dict[str, Any],
    repo_root: Path,
    distributions: Iterable[str],
) -> dict[str, Any]:
    """Collect the minimum provenance record required for a run."""

    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "seed": config["runtime"]["seed"],
        "config": config,
        "packages": package_versions(distributions),
    }


def write_run_metadata(path: str | Path, metadata: dict[str, Any]) -> Path:
    """Write metadata atomically so interrupted runs do not leave partial JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path

