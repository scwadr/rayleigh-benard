"""Save one Copy2 timestep at every width in a completed figure-analysis plan.

Run on a PACE compute node; no discovery or regression is performed.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4
import zipfile


# Copy2 reverses the stored velocity keys when loading the interpolated NPZ.
FIELDS = {"buoyancy": "B", "pressure": "P", "vorticity": "W",
          "u": "V", "v": "U"}


def notebook_filter(path):
    """Load only imports and function definitions, never notebook execution."""
    content = path.read_bytes()
    notebook = json.loads(content)
    cell = next(c for c in notebook["cells"] if c["id"] == "d5dd5858")
    source = "".join(cell["source"])
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {}
    exec(compile(tree, str(path) + "::d5dd5858", "exec"), namespace)
    return namespace["gaussian_filter_2d_finite_support_over_time"], content, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale-plan", type=Path, required=True)
    parser.add_argument("--time-index", type=int, required=True,
                        help="Absolute zero-based index in the preprocessed NPZ")
    parser.add_argument("--field", choices=FIELDS, default="buoyancy")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="PACE scratch directory; creates a unique child directory")
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--clim", type=float, nargs=2, metavar=("MIN", "MAX"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    jax.config.update("jax_enable_x64", True)  # Same precision as Copy2.
    plan_path = args.scale_plan.expanduser().resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    measurement = json.loads((plan_path.parent / "measurement.json").read_bytes())
    if (not (plan_path.parent / "COMPLETE").is_file()
            or (plan_path.parent / "FAILED").exists()
            or plan.get("schema_version") != 1
            or hashlib.sha256(plan_bytes).hexdigest() != measurement["scale_plan_sha256"]):
        raise ValueError("Expected an intact, completed schema-1 scale plan")
    data_path = Path(plan["data"]["path"])
    stat = data_path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (plan["data"]["size_bytes"], plan["data"]["mtime_ns"]):
        raise ValueError("Input changed since integral-scale measurement")
    widths = np.asarray(plan["delta_dx"], dtype=float)
    if (widths.ndim != 1 or not widths.size or not np.isfinite(widths).all()
            or np.any(widths <= 0) or np.any(np.diff(widths) <= 0)
            or not np.isfinite([plan["alpha"], plan["li"]]).all()
            or min(plan["alpha"], plan["li"]) <= 0):
        raise ValueError("Invalid filter widths, alpha, or integral scale")
    with np.load(data_path, allow_pickle=False) as data:
        x, y, t = [np.asarray(data[key], dtype=float) for key in ("x", "y", "t")]
    for coord in (x, y, t):
        if (coord.ndim != 1 or coord.size < 2 or not np.isfinite(coord).all()
                or np.any(np.diff(coord) <= 0)):
            raise ValueError("Invalid coordinates")
    for coord, spacing in ((x, plan["dx"]), (y, plan["dy"])):
        if not np.allclose(np.diff(coord), spacing, rtol=1e-12, atol=1e-14):
            raise ValueError("Grid differs from scale plan")
    if len(y) != 1536 or not np.allclose(y, np.linspace(0, 1, 1536), rtol=1e-12, atol=1e-14):
        raise ValueError("Expected Copy2's interpolated vertical grid")
    stop = len(t) if plan["time_stop"] is None else plan["time_stop"]
    if not 0 <= plan["time_start"] <= args.time_index < stop <= len(t):
        raise ValueError("Time index must lie inside the plan's analysis window")
    if args.dpi <= 0 or (args.clim is not None and
                        (not np.isfinite(args.clim).all() or args.clim[0] >= args.clim[1])):
        raise ValueError("Require positive DPI and finite increasing color limits")

    repo = Path(__file__).resolve().parents[1]
    filter_field, notebook_bytes, filter_source = notebook_filter(repo / "COnvection-FILTER-Copy2.ipynb")
    output = args.output_root.expanduser().resolve() / (
        f"{args.field}_t{args.time_index:06d}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    # NPZ members cannot be memory-mapped directly. Extract just the chosen field
    # to scratch, then copy only one frame into RAM (no full trajectory allocation).
    with tempfile.TemporaryDirectory(prefix="field_", dir=output) as scratch:
        member = FIELDS[args.field] + ".npy"
        extracted = Path(scratch) / member
        with zipfile.ZipFile(data_path) as archive, archive.open(member) as src, extracted.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        trajectory = np.load(extracted, mmap_mode="r", allow_pickle=False)
        if trajectory.shape != (len(x), len(y), len(t)):
            raise ValueError("Expected field shape (x, y, time)")
        frame = np.array(trajectory[..., args.time_index], dtype=float)
        del trajectory
    if not np.isfinite(frame).all():
        raise ValueError("Nonfinite selected field")
    clim = args.clim or [float(frame.min()), float(frame.max())]
    if clim[0] == clim[1]:
        padding = max(1., abs(clim[0])) * 1e-6
        clim = [clim[0] - padding, clim[1] + padding]

    records = []

    def save(values, xp, yp, label, filename, delta_dx):
        fig, ax = plt.subplots(figsize=(12, 4), layout="constrained")
        mesh = ax.pcolormesh(xp, yp, values.T, shading="nearest", cmap=args.cmap,
                             vmin=clim[0], vmax=clim[1], rasterized=True)
        ax.set(xlabel="x", ylabel="y", aspect="equal",
               title=f"{args.field}, t={t[args.time_index]:.8g} (index {args.time_index}) | {label}")
        fig.colorbar(mesh, ax=ax, label=args.field)
        fig.savefig(output / filename, dpi=args.dpi)
        plt.close(fig)
        records.append(dict(file=filename, delta_dx=delta_dx, shape=list(values.shape),
                            x_range=[float(xp[0]), float(xp[-1])],
                            y_range=[float(yp[0]), float(yp[-1])]))
        print(output / filename, flush=True)

    save(frame, x, y, "unfiltered (interpolated input)", "00_unfiltered.png", 0.)
    for index, width in enumerate(widths, 1):
        delta = float(width * plan["dx"])
        filtered, xp, yp = filter_field(jnp.asarray(frame[..., None]), jnp.asarray(x),
                                        jnp.asarray(y), delta, float(plan["alpha"]))
        save(np.asarray(filtered)[..., 0], np.asarray(xp), np.asarray(yp),
             f"Delta/dx={width:.6g}, Delta/li={delta / plan['li']:.6g}",
             f"{index:02d}_delta_dx_{width:.8g}.png", float(width))
    final_stat = data_path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError("Input changed while plotting")
    (output / "scale_plan.json").write_bytes(plan_bytes)
    (output / "filter_cell.py").write_text(filter_source, encoding="utf-8")
    shutil.copyfile(__file__, output / "plot_filtered_snapshot.py")
    manifest = dict(field=args.field, npz_key=FIELDS[args.field], data=plan["data"],
                    time_index=args.time_index, time=float(t[args.time_index]),
                    color_limits=clim, cmap=args.cmap, dpi=args.dpi, images=records,
                    notebook_sha256=hashlib.sha256(notebook_bytes).hexdigest(),
                    filter="Copy2 cell d5dd5858, including original trimming/fallback",
                    validation="Visualization only; input path/stat checked, not full content hashed")
    (output / "snapshot.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    (output / "COMPLETE").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")


if __name__ == "__main__":
    main()
