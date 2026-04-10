"""RF2 delta exporter for SNOMED CT classification.

Converts the Step 2 attribute extraction results into a valid RF2 delta ZIP
that can be submitted to the ELK classification service (snomed-owl-toolkit).

Source concepts are remapped to synthetic SCTIDs (starting at 1 000 000 001)
to avoid collisions with real SNOMED IDs in the base release.  An ID-mapping
CSV is written alongside the delta so that classification results can be
mapped back to the original OMOP concept IDs.

Usage::

    from ariadne.hierarchy.rf2_exporter import export_to_rf2

    # From attribute_results.csv
    zip_path, id_map = export_to_rf2(
        "data/notebook_results/attribute_results.csv",
        "data/rf2_output/",
    )

    # From a DataFrame
    zip_path, id_map = export_to_rf2(results_df, "data/rf2_output/")
"""
from __future__ import annotations

import csv
import logging
import os
import uuid
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SNOMED attribute category → relationship type SCTID
# ---------------------------------------------------------------------------

GS_CATEGORY_TO_TYPE_ID: dict[str, int] = {
    "Has asso morph":    116676008,
    "Has finding site":  363698007,
    "Has causative agent": 246075003,
    "Has clinical course": 263502005,
    "Has finding context": 408729009,
    "Has interpretation":  363713009,
    "Has interprets":      363714003,
    "Has occurrence":      246454002,
    "Has pathology":       370135005,
    "Has relat context":   408732007,
    "Has severity":        246112005,
    "Has temporal context": 408731000,
    "Finding asso with":    47429007,   # Associated with
}

# ---------------------------------------------------------------------------
# RF2 constants
# ---------------------------------------------------------------------------

_MODULE_ID_DEFAULT   = 900000000000207008  # SNOMED CT core module
_DEFN_STATUS_SD      = 900000000000073002  # Sufficiently defined
_CHAR_TYPE_STATED    = 900000000000010007  # Stated
_CHAR_TYPE_INFERRED  = 900000000000011006  # Inferred
_MODIFIER            = 900000000000451002  # Existential
_IS_A_TYPE_ID        = 116680003           # Is a (relationship type)
_CLINICAL_FINDING    = 404684003           # Clinical finding (default stated parent)
_CONCEPT_ID_START    = 1_000_000_001       # First synthetic concept ID
_REL_ID_START        = 2_000_000_001       # First synthetic relationship ID
_MODULE_DEP_REFSET   = 900000000000534007  # Module dependency refset
_MODEL_COMPONENT_MOD = 900000000000012004  # SNOMED CT Model Component module
_OWL_AXIOM_REFSET_ID = 733073007           # OWL Axiom Reference Set
_ROLE_GROUP_ID       = 609096000           # SNOMED role group concept

_CONCEPT_HEADER = "id\teffectiveTime\tactive\tmoduleId\tdefinitionStatusId\n"
_REL_HEADER = (
    "id\teffectiveTime\tactive\tmoduleId\t"
    "sourceId\tdestinationId\trelationshipGroup\ttypeId\t"
    "characteristicTypeId\tmodifierId\n"
)
_OWL_AXIOM_HEADER = (
    "id\teffectiveTime\tactive\tmoduleId\trefsetId\t"
    "referencedComponentId\towlExpression\n"
)
_MODULE_DEP_HEADER = (
    "id\teffectiveTime\tactive\tmoduleId\trefsetId\t"
    "referencedComponentId\tsourceEffectiveTime\ttargetEffectiveTime\n"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_owl_expression(
    source_sctid: int,
    parent_sctids: list[int],
    attributes: list[tuple[int, str]],
) -> str:
    """Build a complete OWL Functional Syntax axiom expression for one concept.

    For concepts with attributes (Sufficiently Defined semantics)::

        EquivalentClasses(:src ObjectIntersectionOf(:p1 :p2
            ObjectSomeValuesFrom(:609096000 <role_group_content>)))

    For concepts with no attributes (fallback to primitive; first parent used)::

        SubClassOf(:src :p1)

    Role group content:

    - 1 attribute:  ``ObjectSomeValuesFrom(:typeId :destId)``
    - 2+ attributes: ``ObjectIntersectionOf(ObjectSomeValuesFrom(...) ...)``

    Args:
        source_sctid: Synthetic SCTID of the source concept.
        parent_sctids: One or more SCTIDs for the stated "Is a" parents.
            At least one is required.
        attributes: List of ``(type_id, destination_sctid_str)`` tuples.

    Returns:
        Full OWL Functional Syntax string.
    """
    if not parent_sctids:
        parent_sctids = [_CLINICAL_FINDING]

    src = f":{source_sctid}"
    par_exprs = " ".join(f":{p}" for p in parent_sctids)

    if not attributes:
        # No attributes — can only assert subsumption (use first/only parent)
        return f"SubClassOf({src} :{parent_sctids[0]})"

    attr_exprs = [f"ObjectSomeValuesFrom(:{t} :{d})" for t, d in attributes]
    if len(attr_exprs) == 1:
        role_group = f"ObjectSomeValuesFrom(:{_ROLE_GROUP_ID} {attr_exprs[0]})"
    else:
        inner = " ".join(attr_exprs)
        role_group = (
            f"ObjectSomeValuesFrom(:{_ROLE_GROUP_ID} ObjectIntersectionOf({inner}))"
        )

    # Build intersection: all parents + role group
    if len(parent_sctids) == 1:
        intersection_args = f"{par_exprs} {role_group}"
    else:
        intersection_args = f"{par_exprs} {role_group}"

    return (
        f"EquivalentClasses({src} ObjectIntersectionOf({intersection_args}))"
    )


def _rows_from_source(
    source: Union[str, Path, pd.DataFrame],
) -> list[tuple[int, str, str, str]]:
    """Return (source_omop_id, source_name, dest_sctid, attr_type) rows.

    Args:
        source: Path to ``attribute_results.csv`` or a DataFrame with the same
                columns (``concept_id_1``, ``concept_name_1``,
                ``predicted_concept_code_2``, ``attribute_type``).

    Returns:
        List of tuples, skipping rows with missing ``predicted_concept_code_2``.
    """
    if isinstance(source, (str, Path)):
        df = pd.read_csv(source)
    else:
        df = source.copy()

    # Support both column-name conventions
    col_code = next(
        (c for c in ["predicted_concept_code_2", "concept_code_2"] if c in df.columns),
        None,
    )
    col_type = next(
        (c for c in ["attribute_type", "attribute_category"] if c in df.columns),
        None,
    )
    if col_code is None:
        raise ValueError(
            "Source must have a 'predicted_concept_code_2' column. "
            "Run the notebook pipeline cell first to generate attribute_results.csv."
        )
    if col_type is None:
        raise ValueError("Source must have an 'attribute_type' or 'attribute_category' column.")

    before = len(df)
    df = df.dropna(subset=[col_code])
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "Skipped %d rows with missing predicted_concept_code_2 (no SCTID resolved).",
            dropped,
        )

    rows = []
    for _, row in df.iterrows():
        rows.append((
            int(row["concept_id_1"]),
            str(row.get("concept_name_1", "")),
            str(row[col_code]),
            str(row[col_type]),
        ))
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_rf2(
    source: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    *,
    date: str | None = None,
    module_id: int = _MODULE_ID_DEFAULT,
    stated_parent: Union[int, dict[int, list[str]]] = _CLINICAL_FINDING,
    rel_group: int = 1,
    concept_id_start: int = _CONCEPT_ID_START,
    rel_id_start: int = _REL_ID_START,
    zip_it: bool = True,
) -> tuple[Path, pd.DataFrame]:
    """Export Step 2 attribute predictions as an RF2 delta ZIP.

    Source OMOP ``concept_id_1`` values are remapped to sequential
    synthetic IDs starting at *concept_id_start* to avoid collisions
    with real SNOMED SCTIDs in the base release.

    Args:
        source: Path to ``attribute_results.csv`` **or** a DataFrame with
            columns ``concept_id_1``, ``concept_name_1``,
            ``predicted_concept_code_2`` (SNOMED SCTID string), and
            ``attribute_type``.
        output_dir: Directory where the ``Delta/`` folder and ZIP are written.
        date: Effective date string in ``YYYYMMDD`` format.  Defaults to today.
        module_id: SNOMED module concept ID (default: SNOMED CT core module).
        stated_parent: Either a single SCTID ``int`` applied to all concepts,
            or a ``dict[omop_concept_id, list[sctid_str]]`` for per-concept
            parents (output of ``build_stated_parents_map``).  Defaults to
            404684003 (Clinical finding).
        rel_group: Relationship group number for all predicted attributes
            (default: 1 — grouped).  Pass 0 for ungrouped.
        concept_id_start: First synthetic concept ID (default: 1 000 000 001).
        rel_id_start: Starting integer for generated relationship IDs.
        zip_it: When True (default), produce a ``.zip`` archive in
            *output_dir*.

    Returns:
        ``(zip_path, id_map_df)`` where *id_map_df* is a DataFrame with
        columns ``omop_concept_id``, ``synthetic_sctid``, ``concept_name``
        mapping original OMOP IDs to synthetic delta IDs.
    """
    date = date or datetime.today().strftime("%Y%m%d")
    output_dir = Path(output_dir)
    term_dir = output_dir / "Delta" / "Terminology"
    term_dir.mkdir(parents=True, exist_ok=True)
    refset_content_dir = output_dir / "Delta" / "Refset" / "Content"
    refset_content_dir.mkdir(parents=True, exist_ok=True)
    refset_meta_dir = output_dir / "Delta" / "Refset" / "Metadata"
    refset_meta_dir.mkdir(parents=True, exist_ok=True)

    rows = _rows_from_source(source)
    if not rows:
        raise ValueError("No rows to export — source is empty or all SCTIDs missing.")

    logger.info("Exporting %d attribute rows to RF2 delta (date=%s)…", len(rows), date)

    # ── Build OMOP → synthetic ID mapping ─────────────────────────────────
    unique_sources = dict.fromkeys(r[0] for r in rows)  # ordered dedup
    omop_to_synth: dict[int, int] = {}
    synth_to_omop: dict[int, int] = {}
    for i, omop_id in enumerate(unique_sources):
        synth_id = concept_id_start + i
        omop_to_synth[omop_id] = synth_id
        synth_to_omop[synth_id] = omop_id

    # Save mapping CSV
    id_map_rows = []
    for omop_id in unique_sources:
        name = next((r[1] for r in rows if r[0] == omop_id), "")
        id_map_rows.append({
            "omop_concept_id": omop_id,
            "synthetic_sctid": omop_to_synth[omop_id],
            "concept_name": name,
        })
    id_map_df = pd.DataFrame(id_map_rows)
    id_map_path = output_dir / f"id_mapping_{date}.csv"
    id_map_df.to_csv(id_map_path, index=False)
    logger.info("  ID mapping: %d concepts → %s", len(id_map_df), id_map_path.name)

    # ── Concept file ──────────────────────────────────────────────────────────
    concept_path = term_dir / f"sct2_Concept_Delta_INT_{date}.txt"
    with concept_path.open("w", encoding="utf-8") as fh:
        fh.write(_CONCEPT_HEADER)
        for omop_id in unique_sources:
            fh.write(
                f"{omop_to_synth[omop_id]}\t{date}\t1\t{module_id}\t{_DEFN_STATUS_SD}\n"
            )
    logger.info("  Concept file: %d concepts → %s", len(unique_sources), concept_path.name)

    # ── Group attributes by source concept ──────────────────────────────────
    concept_attrs: dict[int, list[tuple[int, str]]] = defaultdict(list)
    unknown_types: set[str] = set()
    for src_id, _src_name, dest_sctid, attr_type in rows:
        type_id = GS_CATEGORY_TO_TYPE_ID.get(attr_type)
        if type_id is None:
            unknown_types.add(attr_type)
            continue
        concept_attrs[src_id].append((type_id, dest_sctid))

    if unknown_types:
        logger.warning(
            "Skipped rows with unrecognised attribute_type values: %s",
            unknown_types,
        )

    # ── StatedRelationship file (header-only — definitions use OWL axioms) ──
    stated_path = term_dir / f"sct2_StatedRelationship_Delta_INT_{date}.txt"
    with stated_path.open("w", encoding="utf-8") as fh:
        fh.write(_REL_HEADER)
    logger.info("  StatedRelationship file (header-only): %s", stated_path.name)

    # ── Empty inferred Relationship file (required by snomed-owl-toolkit) ──
    rel_path = term_dir / f"sct2_Relationship_Delta_INT_{date}.txt"
    with rel_path.open("w", encoding="utf-8") as fh:
        fh.write(_REL_HEADER)
    logger.info("  Relationship file (header-only placeholder): %s", rel_path.name)

    # ── OWL Axiom refset (one row per source concept) ─────────────────────
    owl_path = refset_content_dir / f"der2_sRefset_OWLAxiomDelta_INT_{date}.txt"
    with owl_path.open("w", encoding="utf-8") as fh:
        fh.write(_OWL_AXIOM_HEADER)
        for omop_id in unique_sources:
            synth_id = omop_to_synth[omop_id]
            attrs = concept_attrs.get(omop_id, [])
            if isinstance(stated_parent, dict):
                raw_parents = stated_parent.get(omop_id, [str(_CLINICAL_FINDING)])
                parent_ids = [int(p) for p in raw_parents]
            else:
                parent_ids = [int(stated_parent)]
            owl_expr = _build_owl_expression(synth_id, parent_ids, attrs)
            axiom_uuid = str(uuid.uuid4())
            fh.write(
                f"{axiom_uuid}\t{date}\t1\t{module_id}\t"
                f"{_OWL_AXIOM_REFSET_ID}\t{synth_id}\t{owl_expr}\n"
            )
    n_with_attrs = sum(1 for omop_id in unique_sources if concept_attrs.get(omop_id))
    logger.info(
        "  OWL Axiom refset: %d concepts (%d with attributes, %d attr-only Is a) → %s",
        len(unique_sources), n_with_attrs,
        len(unique_sources) - n_with_attrs,
        owl_path.name,
    )

    # ── Module Dependency refset ──────────────────────────────────────────
    mod_dep_path = refset_meta_dir / f"der2_ssRefset_ModuleDependencyDelta_INT_{date}.txt"
    with mod_dep_path.open("w", encoding="utf-8") as fh:
        fh.write(_MODULE_DEP_HEADER)
        dep_uuid = str(uuid.uuid4())
        fh.write(
            f"{dep_uuid}\t{date}\t1\t{module_id}\t{_MODULE_DEP_REFSET}\t"
            f"{_MODEL_COMPONENT_MOD}\t\t{date}\n"
        )
    logger.info("  Module Dependency refset: %s", mod_dep_path.name)

    if not zip_it:
        delta_dir = output_dir / "Delta"
        logger.info("RF2 delta written (no ZIP): %s", delta_dir)
        return delta_dir, id_map_df

    # ── ZIP ───────────────────────────────────────────────────────────────────
    zip_path = output_dir / f"snomed_delta_{date}.zip"
    delta_root = output_dir / "Delta"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in delta_root.rglob("*"):
            if fpath.is_file():
                zf.write(fpath, fpath.relative_to(output_dir))
    logger.info("RF2 delta ZIP: %s", zip_path)
    return zip_path, id_map_df
