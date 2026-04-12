"""CLI entry point for the ariadne.hierarchy sub-package.

Subcommands::

    # Run the evaluation pipeline
    PYTHONPATH=src python -m ariadne.hierarchy run
    PYTHONPATH=src python -m ariadne.hierarchy run --workers 8
    PYTHONPATH=src python -m ariadne.hierarchy run --extraction-model o4-mini

    # Build pgvector indexes
    PYTHONPATH=src python -m ariadne.hierarchy build-index
    PYTHONPATH=src python -m ariadne.hierarchy build-index --check
    PYTHONPATH=src python -m ariadne.hierarchy build-index --rebuild
    PYTHONPATH=src python -m ariadne.hierarchy build-index --attributes-only
    PYTHONPATH=src python -m ariadne.hierarchy build-index --reference-only

    # Via installed entry point (after `pip install -e .`)
    ariadne-build-index [--check | --rebuild | --attributes-only | --reference-only]
"""

import argparse
import logging
import sys

from dotenv import load_dotenv

from ariadne.utils.utils import get_project_root

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subcommand: run (evaluation pipeline)
# ---------------------------------------------------------------------------

def _cmd_run(args: argparse.Namespace) -> None:
    from ariadne.hierarchy.config import HierarchyConfig
    from ariadne.hierarchy.evaluator import evaluate_results, process_gold_standard
    from ariadne.hierarchy.searchers import SnomedAttributeSearcher, SnomedReferenceSearcher

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = HierarchyConfig.from_yaml(args.config)

    if args.extraction_model:
        from dataclasses import replace
        cfg = replace(cfg, models=replace(cfg.models, extraction=args.extraction_model))
        logger.info("Extraction model overridden to: %s", args.extraction_model)

    logger.info("Connecting to PostgreSQL (snomed_attribute / snomed_reference)...")
    attribute_index = SnomedAttributeSearcher(cfg=cfg)
    reference_index = SnomedReferenceSearcher(cfg=cfg)

    with attribute_index, reference_index:
        results = process_gold_standard(
            cfg.evaluation.attribute_gold_standard_path, attribute_index,
            reference_index=reference_index, cfg=cfg,
            max_workers=args.workers,
        )
        evaluate_results(results, cfg.evaluation.attribute_gold_standard_path, cfg=cfg)


# ---------------------------------------------------------------------------
# Subcommand: build-index
# ---------------------------------------------------------------------------

def _cmd_build_index(args: argparse.Namespace) -> None:
    from ariadne.hierarchy.index_builder import build
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    build(
        rebuild=args.rebuild,
        attributes_only=args.attributes_only,
        reference_only=args.reference_only,
        check=args.check,
        reference_sample_size=args.reference_sample_size,
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
    run_p.add_argument("--config", default="config.yaml",
                       help="Path to config.yaml (reads 'hierarchy' section).")
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
    bi_p.add_argument("--rebuild", action="store_true",
                      help="Truncate existing data and rebuild from scratch.")
    bi_p.add_argument("--check", action="store_true",
                      help="Skip any table that already contains data (idempotent).")
    bi_p.add_argument("--attributes-only", action="store_true", dest="attributes_only",
                      help="Only build snomed_attribute.")
    bi_p.add_argument("--reference-only", action="store_true", dest="reference_only",
                      help="Only build snomed_reference.")
    bi_p.add_argument("--reference-sample-size", type=int, default=10_000,
                      dest="reference_sample_size", metavar="N",
                      help="Unique source concepts for the reference index (default: 10000).")
    bi_p.set_defaults(func=_cmd_build_index)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])

