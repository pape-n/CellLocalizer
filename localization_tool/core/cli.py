"""Command line (phase 1): analyse cells listed in an analysis file, or create such a file.

    python -m localization_tool init FOLDER --out analysis.lta.json [--roles DAPI=Nucleus,488=Actin,647=Target:TTN]
    python -m localization_tool analyze analysis.lta.json --out results/ [--no-images] [--csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .channels import DISPLAY_COLORS, ROLES, ChannelAssignment
from .conditions import DEFAULT_VOCABULARY, suggest_condition, suggest_experiment
from .files import find_images
from .project_file import AnalysisFile, CellSpec, ImageSpec, load, save
from .reader import ReaderError, read_image

ROLE_COLORS = {"Nucleus": DISPLAY_COLORS["Blue"], "Actin": DISPLAY_COLORS["Green"],
               "Target": DISPLAY_COLORS["Magenta"], "Membrane": DISPLAY_COLORS["Gray"],
               "Other": DISPLAY_COLORS["Orange"], "Ignore": DISPLAY_COLORS["Gray"]}


def parse_roles(text: str) -> list[tuple[str, str, str]]:
    """'DAPI=Nucleus,488=Actin,647=Target:TTN' -> [(key, role, label)]. Keys match the fluorophore or channel name."""
    out = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        if "=" not in part:
            raise ValueError(f"role entry {part!r} needs the form key=Role or key=Role:Label")
        key, rhs = part.split("=", 1)
        role, _, label = rhs.partition(":")
        role = role.strip().capitalize()
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}, allowed: {', '.join(ROLES)}")
        out.append((key.strip().lower(), role, label.strip() or (role if role != "Target" else key.strip())))
    return out


def assign_by_rules(channels, rules) -> list[ChannelAssignment] | None:
    result = []
    target_colors = iter((DISPLAY_COLORS["Magenta"], DISPLAY_COLORS["Orange"], DISPLAY_COLORS["Cyan"],
                          DISPLAY_COLORS["Red"], DISPLAY_COLORS["Yellow"]))
    for ch in channels:
        text = f"{ch.fluor or ''} {ch.name}".lower()
        hit = next(((r, lab) for key, r, lab in rules if key in text), None)
        if hit is None:
            return None
        role, label = hit
        color = next(target_colors, DISPLAY_COLORS["Gray"]) if role == "Target" else ROLE_COLORS[role]
        result.append(ChannelAssignment(role=role, label=label, color=color))
    return result


def cmd_init(args) -> int:
    folder = Path(args.folder)
    files, skipped = find_images(folder)          # same rules as "Add folder" in the program
    for f, reason in skipped:
        print(f"skipped {f.name}: {reason}")
    rules = parse_roles(args.roles) if args.roles else []
    out = Path(args.out)
    af = AnalysisFile()
    for f in files:
        try:
            stack = read_image(f)
        except ReaderError as exc:
            print(f"skipped {f.name}: {exc}")
            continue
        assignments = assign_by_rules(stack.channels, rules) if rules else None
        if rules and assignments is None:
            print(f"note {f.name}: not every channel matched --roles, please set the roles in the file")
        af.images.append(ImageSpec(path=f, experiment=suggest_experiment(f),
                                   condition=suggest_condition(f, DEFAULT_VOCABULARY),
                                   assignments=assignments, cells=[CellSpec(name="cell1")]))
    path = save(af, out)
    print(f"{len(af.images)} images written to {path}")
    return 0


def cmd_analyze(args) -> int:
    from .batch import run

    af = load(args.file)
    summary = run(af, args.out, control_images=not args.no_images, progress=print, csv_tables=args.csv,
                  base_dir=Path(args.file).parent)
    print(f"{summary['rows']} rows written to {summary['workbook']}")
    for e in summary["errors"]:
        print(f"error: {e}")
    return 1 if summary["errors"] and summary["rows"] == 0 else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="localization_tool", description="CellLocalizer command line")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="create an analysis file for all images in a folder")
    p.add_argument("folder")
    p.add_argument("--out", required=True)
    p.add_argument("--roles", default="", help="e.g. DAPI=Nucleus,488=Actin,647=Target:TTN")
    p.set_defaults(func=cmd_init)
    p = sub.add_parser("analyze", help="analyse all cells of an analysis file")
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--csv", action="store_true", help="also write cells.csv and profiles.csv")
    p.set_defaults(func=cmd_analyze)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
