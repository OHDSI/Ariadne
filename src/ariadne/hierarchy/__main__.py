"""CLI entry point for the ariadne.hierarchy sub-package.

Subcommands::

    # Run the evaluation pipeline
    PYTHONPATH=src python -m ariadne.hierarchy run
    PYTHONPATH=src python -m ariadne.hierarchy run --workers 8
    PYTHONPATH=src python -m ariadne.hierarchy run --extraction-model o4-mini

    # Build pgvector indexes
    PYTHONPATH=src python -m ariadne.hierarchy build-index
    PYTHONPATH=src python -m ariadne.hierarchy build-index --if-exists skip
    PYTHONPATH=src python -m ariadne.hierarchy build-index --if-exists rebuild
    PYTHONPATH=src python -m ariadne.hierarchy build-index --attributes-only
    PYTHONPATH=src python -m ariadne.hierarchy build-index --reference-only

    # Via installed entry point (after `pip install -e .`)
    ariadne-build-index [--if-exists {append,skip,rebuild} | --attributes-only | --reference-only]
"""

import argparse
import logging
import sys
from typing import cast

from dotenv import load_dotenv

from ariadne.utils.utils import get_project_root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcommand: run (evaluation pipeline)
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    from ariadne.hierarchy.evaluator import evaluate_results
    from ariadne.hierarchy.runner import process_hierarchy
    from ariadne.hierarchy.searchers import SnomedAttributeSearcher, SnomedReferenceSearcher
    from ariadne.utils.config import load_hierarchy_settings
    from ariadne.utils.settings import HierarchySettings

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = cast(HierarchySettings, load_hierarchy_settings(args.config))

    if args.extraction_model:
        cfg.extraction = args.extraction_model
        logger.info("Extraction model overridden to: %s", args.extraction_model)

    logger.info("Connecting to PostgreSQL (snomed_attribute / snomed_reference)...")
    attribute_searcher = SnomedAttributeSearcher(cfg=cfg)
    reference_searcher = SnomedReferenceSearcher(cfg=cfg)

    with attribute_searcher, reference_searcher:
        results = process_hierarchy(
            cfg.evaluation.attribute_gold_standard_path,
            attribute_searcher,
            reference_searcher=reference_searcher,
            cfg=cfg,
            max_workers=args.workers,
        )
        evaluate_results(results, cfg.evaluation.attribute_gold_standard_path, cfg=cfg)


# ---------------------------------------------------------------------------
# Subcommand: build-index
# ---------------------------------------------------------------------------

def _cmd_build_index(args: argparse.Namespace) -> None:
    from ariadne.hierarchy.attribute_ref_table_builder import build_attribute_reference_tables
    from ariadne.utils.config import load_hierarchy_settings
    from ariadne.utils.settings import HierarchySettings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    cfg = cast(HierarchySettings, load_hierarchy_settings(args.config))

    if_exists = args.if_exists
    # Backward-compatible aliases for existing scripts.
    if args.rebuild:
        if_exists = "rebuild"
    if args.check:
        if if_exists == "rebuild":
            raise ValueError("--check and --rebuild cannot be used together.")
        if_exists = "skip"

    build_attribute_reference_tables(
        cfg=cfg,
        if_exists=if_exists,
        attributes_only=args.attributes_only,
        reference_only=args.reference_only,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    load_dotenv(get_project_root() / ".env")

    parser = argparse.ArgumentParser(
        prog="python -m ariadne.hierarchy",
        description="Ariadne hierarchy sub-package CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- run ----------------------------------------------------------------
    run_p = subparsers.add_parser("run", help="Run the evaluation pipeline.")
    run_p.add_argument("--config", default="config_condition_mapping.yaml",
                       help="Path to config_condition_mapping.yaml (reads 'hierarchy' section).")
    run_p.add_argument("--workers", type=int, default=1,
                       help="Parallel worker threads (default: 1).")
    run_p.add_argument("--extraction-model", default=None, dest="extraction_model",
                       metavar="MODEL",
                       help="Override the extraction model (e.g. o4-mini).")
    run_p.set_defaults(func=_cmd_run)

    # -- build-index --------------------------------------------------------
    bi_p = subparsers.add_parser(
        "build-index",
        help="Build/rebuild the snomed_attribute and snomed_reference pgvector tables.",
    )
    bi_p.add_argument("--config", default="config_condition_mapping.yaml",
                      help="Path to config_condition_mapping.yaml (reads 'hierarchy.index_build').")
    bi_p.add_argument(
        "--if-exists",
        default="append",
        choices=["append", "skip", "rebuild"],
        help=(
            "Behavior when target tables already contain data: "
            "append (default), skip, or rebuild (truncate then insert)."
        ),
    )
    bi_p.add_argument("--rebuild", action="store_true", help=argparse.SUPPRESS)
    bi_p.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    bi_p.add_argument("--attributes-only", action="store_true", dest="attributes_only",
                      help="Only build snomed_attribute.")
    bi_p.add_argument("--reference-only", action="store_true", dest="reference_only",
                      help="Only build snomed_reference.")
    bi_p.set_defaults(func=_cmd_build_index)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

