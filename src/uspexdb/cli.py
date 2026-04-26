from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .paths import PROJECT_ROOT


DEFAULT_QUERY_JSON = PROJECT_ROOT / "query.json"
DEFAULT_CONFIG_JSON = PROJECT_ROOT / "config.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uspexdb",
        description=(
            "Unified CLI for USPEX database workflows. "
            "'query' runs MOL2 contact search; 'config' runs metadata search/export."
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("query", "config", "all"),
        default="query",
        help="What to run. Default: query",
    )
    parser.add_argument(
        "--query",
        dest="query_json",
        default=str(DEFAULT_QUERY_JSON),
        help="Path to query JSON. Default: query.json in the project root.",
    )
    parser.add_argument(
        "--config",
        dest="config_json",
        default=str(DEFAULT_CONFIG_JSON),
        help="Path to config JSON. Default: config.json in the project root.",
    )
    parser.add_argument(
        "--rebuild-compact",
        action="store_true",
        help="Rebuild compact cache before query mode.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable query validation even if validation.reference_json is configured.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode in {"query", "all"}:
        from .query_search.runner import run_query_mode

        run_query_mode(
            Path(args.query_json),
            rebuild_compact=bool(args.rebuild_compact),
            validate=False if args.no_validate else None,
        )

    if args.mode in {"config", "all"}:
        from .config_search.runner import run_config_mode

        run_config_mode(Path(args.config_json))

    return 0
