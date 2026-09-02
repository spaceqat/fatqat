"""Packaged architecture profiles used by the internal ZAP algorithm."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import resources

_BUILTIN_ARCHITECTURES = frozenset(("default", "scale_to_100", "scale_to_500"))


def load_architecture(name: str = "default") -> dict[str, object]:
    """Load one built-in architecture profile as a fresh JSON value."""
    if name not in _BUILTIN_ARCHITECTURES:
        raise ValueError(f"unknown ZAP architecture {name!r}")
    resource = resources.files(__package__).joinpath("architectures", f"{name}.json")
    return json.loads(resource.read_text(encoding="utf-8"))


def architecture_sites(
    architecture: Mapping[str, object], zone_name: str
) -> list[tuple[float, float]]:
    """Expand unique sites from an architecture zone in ZAP's row-major order."""
    sites = []
    for zone in architecture[zone_name]:
        for slm in zone["slms"]:
            x, y = slm["location"]
            separation_x, separation_y = slm["site_seperation"]
            sites.extend(
                (x + column * separation_x, y + row * separation_y)
                for row in range(slm["r"])
                for column in range(slm["c"])
            )
    return list(dict.fromkeys(sites))
