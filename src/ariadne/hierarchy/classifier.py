"""SNOMED CT classification via snomed-owl-toolkit (ELK reasoner).

Given an RF2 delta ZIP (produced by :mod:`ariadne.hierarchy.rf2_exporter`)
and a base SNOMED CT International Edition snapshot, this module runs the
ELK OWL EL++ classifier to infer *Is a* (parent) relationships for the
new extension concepts.

Prerequisites
-------------
1. **Java 17+** installed and on ``PATH``.
2. **snomed-owl-toolkit executable JAR** — download from
   https://github.com/IHTSDO/snomed-owl-toolkit/releases and place at the
   path specified by ``SNOMED_OWL_TOOLKIT_JAR`` (env var) or pass explicitly.
3. **SNOMED CT International Edition RF2 snapshot** — obtain from
   https://mlds.ihtsdotools.org/ (free registration, SNOMED Affiliate License).
   Set ``SNOMED_BASE_RELEASE_ZIP`` env var or pass explicitly.

Usage::

    from ariadne.hierarchy.classifier import classify_delta, parse_classification_results

    results_zip = classify_delta("data/rf2_output/snomed_delta_20260404.zip")
    new_is_a, removed = parse_classification_results(results_zip)
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd

from ariadne.hierarchy.rf2_exporter import GS_CATEGORY_TO_TYPE_ID

logger = logging.getLogger(__name__)

# Valid SNOMED relationship type IDs (for pre-classification validation)
_VALID_TYPE_IDS: set[str] = {
    "116680003",  # Is a
    *(str(v) for v in GS_CATEGORY_TO_TYPE_ID.values()),
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _toolkit_jar(explicit: str | None = None) -> str:
    """Resolve the path to the snomed-owl-toolkit executable JAR."""
    path = (
        explicit
        or os.environ.get("SNOMED_OWL_TOOLKIT_JAR")
        or "tools/snomed-owl-toolkit.jar"
    )
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"snomed-owl-toolkit JAR not found at '{path}'.  "
            "Download it from https://github.com/IHTSDO/snomed-owl-toolkit/releases "
            "and set SNOMED_OWL_TOOLKIT_JAR or pass toolkit_jar= explicitly."
        )
    return path


def _base_release(explicit: str | None = None) -> str:
    """Resolve the path to the SNOMED CT base release ZIP."""
    path = explicit or os.environ.get("SNOMED_BASE_RELEASE_ZIP")
    if not path:
        raise ValueError(
            "No base SNOMED CT release specified.  "
            "Set SNOMED_BASE_RELEASE_ZIP env var or pass base_snomed_zip= explicitly."
        )
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"SNOMED CT base release not found at '{path}'.  "
            "Download from https://mlds.ihtsdotools.org/"
        )
    return path


# ---------------------------------------------------------------------------
# Pre-classification validation
# ---------------------------------------------------------------------------

def pre_classification_checks(delta_zip: str | Path) -> list[str]:
    """Run lightweight structural checks on an RF2 delta ZIP.

    Validates:
    - The ZIP contains the expected Terminology files
    - StatedRelationship file has correct TSV headers
    - All ``destinationId`` values look like valid SCTIDs (6+ digit integers)
    - All ``typeId`` values are in the known set of SNOMED attribute types

    Args:
        delta_zip: Path to the RF2 delta ZIP.

    Returns:
        List of issue descriptions.  Empty list means all checks passed.
    """
    issues: list[str] = []
    delta_zip = Path(delta_zip)

    if not delta_zip.is_file():
        issues.append(f"Delta ZIP not found: {delta_zip}")
        return issues

    stated_found = False
    stated_row_count = 0
    owl_axiom_found = False
    owl_axiom_row_count = 0
    concept_found = False

    with zipfile.ZipFile(delta_zip, "r") as zf:
        for name in zf.namelist():
            if "StatedRelationship" in name and name.endswith(".txt"):
                stated_found = True
                with zf.open(name) as f:
                    df = pd.read_csv(f, sep="\t", dtype=str)

                stated_row_count = len(df)

                # Only validate content if file is non-empty
                if stated_row_count > 0:
                    expected_cols = {
                        "id", "effectiveTime", "active", "moduleId",
                        "sourceId", "destinationId", "relationshipGroup",
                        "typeId", "characteristicTypeId", "modifierId",
                    }
                    missing_cols = expected_cols - set(df.columns)
                    if missing_cols:
                        issues.append(f"StatedRelationship missing columns: {missing_cols}")

                    if "destinationId" in df.columns:
                        bad_dests = df[
                            ~df["destinationId"].str.match(r"^\d{6,18}$", na=False)
                        ]
                        if len(bad_dests) > 0:
                            samples = bad_dests["destinationId"].head(5).tolist()
                            issues.append(
                                f"{len(bad_dests)} destinationId values look invalid "
                                f"(expected 6-18 digit SCTIDs). Samples: {samples}"
                            )

                    if "typeId" in df.columns:
                        unknown_types = set(df["typeId"]) - _VALID_TYPE_IDS
                        if unknown_types:
                            issues.append(
                                f"Unknown typeId values (not standard SNOMED attribute types): "
                                f"{unknown_types}"
                            )

            elif "OWLAxiom" in name and name.endswith(".txt"):
                owl_axiom_found = True
                with zf.open(name) as f:
                    owl_df = pd.read_csv(f, sep="\t", dtype=str)

                # Filter to active rows only
                active_owl = owl_df[owl_df.get("active", pd.Series(dtype=str)) == "1"] if "active" in owl_df.columns else owl_df
                owl_axiom_row_count = len(active_owl)

                if owl_axiom_row_count > 0:
                    expected_owl_cols = {
                        "id", "active", "moduleId", "refsetId",
                        "referencedComponentId", "owlExpression",
                    }
                    missing_owl_cols = expected_owl_cols - set(owl_df.columns)
                    if missing_owl_cols:
                        issues.append(f"OWL Axiom refset missing columns: {missing_owl_cols}")
                    elif "owlExpression" in owl_df.columns:
                        # Validate each active expression starts with a known axiom type
                        bad_exprs = active_owl[
                            ~active_owl["owlExpression"].str.match(
                                r"^(EquivalentClasses|SubClassOf|TransitiveObjectProperty|ReflexiveObjectProperty)\(",
                                na=False,
                            )
                        ]
                        if len(bad_exprs) > 0:
                            samples = bad_exprs["owlExpression"].head(3).str[:80].tolist()
                            issues.append(
                                f"{len(bad_exprs)} OWL expressions do not start with a known "
                                f"axiom type (EquivalentClasses/SubClassOf/...). "
                                f"Samples: {samples}"
                            )

                        # Validate referencedComponentId looks like SCTIDs
                        if "referencedComponentId" in owl_df.columns:
                            bad_ids = active_owl[
                                ~active_owl["referencedComponentId"].str.match(
                                    r"^\d{6,18}$", na=False
                                )
                            ]
                            if len(bad_ids) > 0:
                                samples = bad_ids["referencedComponentId"].head(5).tolist()
                                issues.append(
                                    f"{len(bad_ids)} OWL axiom referencedComponentId values "
                                    f"look invalid. Samples: {samples}"
                                )

            if "Concept" in name and name.endswith(".txt") and "OWLAxiom" not in name:
                concept_found = True

    if not stated_found and not owl_axiom_found:
        issues.append(
            "No StatedRelationship or OWL Axiom refset file found in the delta ZIP."
        )
    elif stated_row_count == 0 and owl_axiom_row_count == 0:
        issues.append(
            "No concept definitions found: StatedRelationship is empty and OWL Axiom "
            "refset has no active rows.  At least one must contain data."
        )
    if not concept_found:
        issues.append("No Concept file found in the delta ZIP.")

    return issues


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_delta(
    delta_zip: str | Path,
    base_snomed_zip: str | None = None,
    *,
    toolkit_jar: str | None = None,
    java_xms: str = "4g",
    timeout: int = 600,
    output_dir: str | Path | None = None,
) -> Path:
    """Run ELK classification via snomed-owl-toolkit.

    This calls the snomed-owl-toolkit's ``-classify`` command which:

    1. Converts the base SNOMED RF2 snapshot + your delta to OWL
    2. Runs the ELK reasoner to infer *Is a* relationships
    3. Produces a ``classification-results-*.zip`` with the inferred
       relationship changes

    Args:
        delta_zip: Path to the RF2 delta ZIP (from ``export_to_rf2``).
        base_snomed_zip: Path to the SNOMED CT International Edition RF2
            snapshot ZIP.  Falls back to ``SNOMED_BASE_RELEASE_ZIP`` env var.
        toolkit_jar: Path to the snomed-owl-toolkit executable JAR.
            Falls back to ``SNOMED_OWL_TOOLKIT_JAR`` env var, then
            ``tools/snomed-owl-toolkit.jar``.
        java_xms: JVM initial heap size (default ``4g``).
        timeout: Maximum seconds to wait for classification (default 600).
        output_dir: Directory where the results ZIP will be written.
            Defaults to the parent directory of *delta_zip*.

    Returns:
        Path to the classification results ZIP.

    Raises:
        RuntimeError: If the classification process fails.
        FileNotFoundError: If the toolkit JAR or base release cannot be found.
        subprocess.TimeoutExpired: If classification exceeds *timeout*.
    """
    jar = _toolkit_jar(toolkit_jar)
    base = _base_release(base_snomed_zip)
    delta_zip = Path(delta_zip)
    output_dir = Path(output_dir or delta_zip.parent)

    cmd = [
        "java",
        f"-Xms{java_xms}",
        "--add-opens", "java.base/java.lang=ALL-UNNAMED",
        "-jar", jar,
        "-classify",
        "-rf2-snapshot-archives", base,
        "-rf2-authoring-delta-archive", str(delta_zip),
    ]

    logger.info("Running classification: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(output_dir),
    )

    if result.stdout:
        logger.info("snomed-owl-toolkit stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("snomed-owl-toolkit stderr:\n%s", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Classification failed (exit code {result.returncode}).\n"
            f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )

    # Find the newest results ZIP
    pattern = str(output_dir / "classification-results-*.zip")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(
            f"No classification-results-*.zip found in {output_dir} after "
            "successful classification.  Check snomed-owl-toolkit output above."
        )

    results_zip = Path(candidates[-1])
    logger.info("Classification complete → %s", results_zip)
    return results_zip


# ---------------------------------------------------------------------------
# Results parsing
# ---------------------------------------------------------------------------

def parse_classification_results(
    results_zip: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse the classification results ZIP into DataFrames.

    Args:
        results_zip: Path to the ``classification-results-*.zip`` produced by
            :func:`classify_delta`.

    Returns:
        ``(new_is_a, removed_redundant, equiv_df)`` where:

        - **new_is_a** — DataFrame of newly inferred *Is a* relationships
          (``active == 1``, ``typeId == 116680003``).  Columns:
          ``sourceId``, ``destinationId``, plus all original RF2 columns.
        - **removed_redundant** — DataFrame of relationships marked inactive
          (``active == 0``).  These are stated relationships that became
          redundant after classification.
        - **equiv_df** — DataFrame of equivalent concept pairs from the
          EquivalentConceptSimpleMap refset.  Columns include
          ``referencedComponentId`` and ``mapTarget`` (group UUID).
    """
    results_zip = Path(results_zip)
    rel_dfs: list[pd.DataFrame] = []
    equiv_df = pd.DataFrame()

    with zipfile.ZipFile(results_zip, "r") as zf:
        for name in zf.namelist():
            if "Relationship" in name and name.endswith(".txt"):
                with zf.open(name) as f:
                    df = pd.read_csv(f, sep="\t", dtype=str)
                    rel_dfs.append(df)

            # Parse equivalent concepts refset
            if "equivalent" in name.lower() or "equiv" in name.lower():
                with zf.open(name) as f:
                    equiv_df = pd.read_csv(f, sep="\t", dtype=str)
                    if len(equiv_df) > 0:
                        logger.warning(
                            "EQUIVALENT CONCEPTS FOUND (%d rows)!  "
                            "This usually indicates a modelling error — "
                            "two concepts have identical defining attributes.",
                            len(equiv_df),
                        )

    if not rel_dfs:
        logger.warning("No Relationship files found in %s", results_zip)
        empty = pd.DataFrame(columns=[
            "id", "effectiveTime", "active", "moduleId", "sourceId",
            "destinationId", "relationshipGroup", "typeId",
            "characteristicTypeId", "modifierId",
        ])
        return empty, empty, equiv_df

    all_rels = pd.concat(rel_dfs, ignore_index=True)

    new_is_a = all_rels[
        (all_rels["typeId"] == "116680003") & (all_rels["active"] == "1")
    ].copy()

    removed = all_rels[all_rels["active"] == "0"].copy()

    logger.info(
        "Classification results: %d new 'Is a' relationships, "
        "%d redundant removals, %d equivalent concept rows.",
        len(new_is_a), len(removed), len(equiv_df),
    )

    return new_is_a, removed, equiv_df


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def resolve_parent_names(
    is_a_df: pd.DataFrame,
    source_names: dict[int, str] | pd.DataFrame | None = None,
    id_mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add human-readable names to inferred *Is a* relationships.

    Resolves ``destinationId`` (SNOMED SCTIDs) to concept names via the
    vocabulary database, and ``sourceId`` (synthetic IDs) back to OMOP IDs
    using the ID mapping from ``export_to_rf2``.

    Args:
        is_a_df: DataFrame with ``sourceId`` and ``destinationId`` columns
            (from :func:`parse_classification_results`).
        source_names: Mapping of source concept ID → name.  Can be:
            - ``dict[int, str]`` — direct mapping
            - ``pd.DataFrame`` with ``concept_id_1`` and ``concept_name_1`` columns
            - ``None`` — source names will be left as IDs
        id_mapping: DataFrame with ``synthetic_sctid`` and ``omop_concept_id``
            columns (from ``export_to_rf2``).  Used to translate synthetic
            sourceIds back to OMOP IDs.  If ``None``, sourceIds are used as-is.

    Returns:
        DataFrame with columns: ``source_id``, ``source_name``,
        ``parent_sctid``, ``parent_name``.
    """
    import psycopg
    from pgvector.psycopg import register_vector

    from ariadne.utils.utils import get_environment_variable

    if is_a_df.empty:
        return pd.DataFrame(
            columns=["source_id", "source_name", "parent_sctid", "parent_name"]
        )

    # Build synthetic → OMOP mapping if provided
    synth_to_omop: dict[str, str] = {}
    if id_mapping is not None:
        synth_to_omop = dict(
            zip(
                id_mapping["synthetic_sctid"].astype(str),
                id_mapping["omop_concept_id"].astype(str),
            )
        )

    # Resolve source names (keyed by OMOP concept_id)
    if isinstance(source_names, pd.DataFrame):
        src_map: dict[str, str] = dict(
            zip(
                source_names["concept_id_1"].astype(str),
                source_names["concept_name_1"],
            )
        )
    elif isinstance(source_names, dict):
        src_map = {str(k): v for k, v in source_names.items()}
    else:
        src_map = {}

    # Resolve parent names from DB using concept_code (SCTID)
    parent_sctids = is_a_df["destinationId"].unique().tolist()

    # Some parents may be synthetic delta concepts (inter-delta Is a);
    # resolve those from the id_mapping instead of querying the DB.
    parent_name_map: dict[str, str] = {}
    real_parent_sctids: list[str] = []
    if id_mapping is not None:
        synth_set = set(id_mapping["synthetic_sctid"].astype(str))
        synth_name_lookup = dict(
            zip(
                id_mapping["synthetic_sctid"].astype(str),
                id_mapping["concept_name"].astype(str),
            )
        )
        for sctid in parent_sctids:
            if sctid in synth_set:
                parent_name_map[sctid] = synth_name_lookup.get(sctid, sctid)
            else:
                real_parent_sctids.append(sctid)
    else:
        real_parent_sctids = parent_sctids

    conn_str = get_environment_variable("VOCAB_CONNECTION_STRING")
    conn_str = conn_str.replace("+psycopg", "").replace("+psycopg2", "")
    schema = get_environment_variable("VOCAB_SCHEMA")

    if real_parent_sctids:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT concept_code, concept_name "
                    f"FROM {schema}.concept "
                    f"WHERE vocabulary_id = 'SNOMED' "
                    f"  AND concept_code = ANY(%s)",
                    (real_parent_sctids,),
                )
                for code, name in cur.fetchall():
                    parent_name_map[str(code)] = name

    resolved = len(parent_name_map)
    total = len(parent_sctids)
    if resolved < total:
        unresolved = set(parent_sctids) - set(parent_name_map.keys())
        logger.warning(
            "Could not resolve %d / %d parent SCTIDs to names: %s",
            total - resolved, total, list(unresolved)[:10],
        )

    # Also build a fallback name map from id_mapping for source concepts
    # whose OMOP IDs may not be in source_names (e.g. from attribute_results
    # but absent from the gold standard DataFrame).
    if id_mapping is not None:
        for _, m in id_mapping.iterrows():
            omop_str = str(m["omop_concept_id"])
            if omop_str not in src_map and str(m.get("concept_name", "")):
                src_map[omop_str] = str(m["concept_name"])

    rows = []
    for _, rel in is_a_df.iterrows():
        synth_id = str(rel["sourceId"])
        # Map synthetic ID back to OMOP ID if mapping is available
        omop_id = synth_to_omop.get(synth_id, synth_id)
        dest_id = str(rel["destinationId"])
        rows.append({
            "source_id": omop_id,
            "source_name": src_map.get(omop_id, omop_id),
            "parent_sctid": dest_id,
            "parent_name": parent_name_map.get(dest_id, dest_id),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def classification_summary(
    new_is_a: pd.DataFrame,
    removed: pd.DataFrame,
    total_source_concepts: int | None = None,
) -> dict[str, object]:
    """Compute summary statistics from classification results.

    Args:
        new_is_a: DataFrame of new inferred *Is a* relationships.
        removed: DataFrame of removed redundant relationships.
        total_source_concepts: Total number of source concepts in the delta
            (used to detect orphans).  If ``None``, orphan detection is skipped.

    Returns:
        Dict with keys: ``new_is_a_count``, ``removed_count``,
        ``concepts_with_parents``, ``orphan_concepts``,
        ``avg_parents_per_concept``, ``max_parents``, ``min_parents``.
    """
    stats: dict[str, object] = {
        "new_is_a_count": len(new_is_a),
        "removed_count": len(removed),
    }

    if new_is_a.empty:
        stats.update({
            "concepts_with_parents": 0,
            "orphan_concepts": total_source_concepts or "unknown",
            "avg_parents_per_concept": 0.0,
            "max_parents": 0,
            "min_parents": 0,
        })
        return stats

    parents_per_concept = new_is_a.groupby("sourceId").size()
    stats["concepts_with_parents"] = len(parents_per_concept)
    stats["avg_parents_per_concept"] = round(parents_per_concept.mean(), 2)
    stats["max_parents"] = int(parents_per_concept.max())
    stats["min_parents"] = int(parents_per_concept.min())

    if total_source_concepts is not None:
        classified_sources = set(new_is_a["sourceId"].unique())
        stats["orphan_concepts"] = total_source_concepts - len(classified_sources)
    else:
        stats["orphan_concepts"] = "unknown"

    return stats
