"""Architecture rule: lunar_rover_sim.core imports nothing from Isaac, USD, Omniverse or Warp."""
from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "src" / "lunar_rover_sim" / "core"
FORBIDDEN_ROOTS = {"omni", "pxr", "isaacsim", "carb", "warp", "torch", "lunar_rover_sim.runtime"}


def imported_modules(path: Path) -> set[str]:
    """Absolute module names imported by one source file (top level and inside functions)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_core_is_isaac_free():
    """No core module imports a forbidden package."""
    violations = []
    for path in sorted(CORE.rglob("*.py")):
        for name in imported_modules(path):
            if any(name == root or name.startswith(root + ".") for root in FORBIDDEN_ROOTS):
                violations.append(f"{path.relative_to(CORE)}: {name}")
    assert not violations, violations
