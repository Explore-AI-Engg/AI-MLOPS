from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_validation_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Write a JSON validation report for CI and auditability."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")