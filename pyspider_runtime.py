"""Select and snapshot the one pinned PySPIDER dependency; no scientific workflow."""

import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PIN = "ccc892cdc84fb183c13531a7fa654132acbe1818"
DISTRIBUTION = "PySPIDER_SR"


def source_hashes(package):
    """Hash the complete Python source inventory, including added modules."""
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*.py"))}


def resolve_source():
    """Locate a pinned install, clean checkout, or verified scratch snapshot."""
    manifest = os.environ.get("SPIDER_RUNTIME_MANIFEST")
    checkout = os.environ.get("SPIDER_CHECKOUT")
    if manifest:
        manifest = Path(manifest).expanduser().resolve()
        identity = json.loads(manifest.read_text(encoding="utf-8"))
        package = manifest.parent / "PySPIDER"
        if identity.get("revision") != PIN:
            raise RuntimeError("Scratch PySPIDER revision differs from the pin")
        if source_hashes(package) != identity.get("python_sources"):
            raise RuntimeError("Scratch PySPIDER source inventory/hash mismatch")
        identity = {**identity, "mode": "scratch-snapshot",
                    "manifest": str(manifest)}
    elif checkout:
        checkout = Path(checkout).expanduser().resolve()
        package = checkout / "src" / "PySPIDER"
        revision = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        changes = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain",
             "--untracked-files=all", "--", "src/PySPIDER"], text=True).strip()
        if revision != PIN or changes:
            raise RuntimeError(f"SPIDER_CHECKOUT must contain clean PySPIDER source at {PIN}")
        identity = {"revision": PIN, "mode": "git-checkout", "checkout": str(checkout)}
    else:
        try:
            distribution = importlib.metadata.distribution(DISTRIBUTION)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("Install requirements/pyspider.txt or set SPIDER_CHECKOUT") from exc
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct_url.get("vcs_info", {}).get("commit_id") != PIN:
            raise RuntimeError("PySPIDER revision differs; install requirements/pyspider.txt")
        package = Path(distribution.locate_file("PySPIDER")).resolve()
        identity = {"revision": PIN, "mode": "pip-distribution",
                    "distribution": DISTRIBUTION, "version": distribution.version,
                    "direct_url": direct_url}
    if not (package / "__init__.py").is_file():
        raise RuntimeError(f"PySPIDER package source missing: {package}")
    return package, {**identity, "package_path": str(package),
                     "python_sources": source_hashes(package)}


def activate_pyspider():
    """Select one namespace before notebook imports, then check the loaded path."""
    package, identity = resolve_source()
    for name, module in tuple(sys.modules.items()):
        if name == "SPIDER" or name.startswith("SPIDER."):
            raise RuntimeError("Legacy SPIDER is loaded; restart the kernel before migration")
        if name == "PySPIDER" or name.startswith("PySPIDER."):
            source = getattr(module, "__file__", None)
            if source and not Path(source).resolve().is_relative_to(package):
                raise RuntimeError("A different PySPIDER is loaded; restart the kernel")
    sys.path.insert(0, str(package.parent))
    importlib.invalidate_caches()
    module = importlib.import_module("PySPIDER")
    if Path(module.__file__).resolve().parent != package:
        raise RuntimeError("Imported PySPIDER differs from the selected package")
    # The fork calls the NumPy 2.x spelling; preserve older PACE environments.
    import numpy as np
    if not hasattr(np, "trapezoid"):
        np.trapezoid = np.trapz
    return identity


def snapshot_pyspider(work):
    """Freeze the selected source before imports; fork/spawn workers inherit it."""
    package, identity = resolve_source()
    target = Path(work).resolve() / "PySPIDER"
    shutil.copytree(package, target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.ipynb"))
    if source_hashes(target) != identity["python_sources"]:
        raise RuntimeError("PySPIDER changed while creating its scratch snapshot")
    manifest = target.parent / "pyspider-source.json"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, indent=2, sort_keys=True)
    os.environ["SPIDER_RUNTIME_MANIFEST"] = str(manifest)
    # Also covers fresh spawned workers before they execute any notebook cell.
    sys.path.insert(0, str(target.parent))
    return identity
