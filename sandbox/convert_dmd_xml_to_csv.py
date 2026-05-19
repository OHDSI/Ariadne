"""Convert dm+d XML data into analysis-ready CSV outputs.

This script performs three stages:
1) Split each XML file into one CSV per top-level section.
2) Build a denormalized dm+d table for downstream mapping work.
3) Create a random sample of the denormalized table.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import TextIO
import xml.etree.ElementTree as ET

import pandas as pd

def strip_tag(tag: str) -> str:
    """Remove an XML namespace from a tag if present."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def flatten_element(
    elem: ET.Element,
    result: dict[str, str],
    prefix: str = "",
) -> None:
    """Flatten nested XML into dot-notation keys, indexing repeated siblings."""
    for attr_name, attr_value in elem.attrib.items():
        key = f"{prefix}.@{attr_name}" if prefix else f"@{attr_name}"
        result[key] = attr_value

    children = list(elem)
    text = (elem.text or "").strip()

    if not children:
        if text:
            leaf_key = prefix or strip_tag(elem.tag)
            result[leaf_key] = text
        return

    if text and prefix:
        result[f"{prefix}.__text"] = text

    child_name_counts = Counter(strip_tag(child.tag) for child in children)
    child_seen = defaultdict(int)

    for child in children:
        child_name = strip_tag(child.tag)
        child_seen[child_name] += 1

        if child_name_counts[child_name] > 1:
            child_part = f"{child_name}[{child_seen[child_name]}]"
        else:
            child_part = child_name

        child_prefix = f"{prefix}.{child_part}" if prefix else child_part
        flatten_element(child, result, prefix=child_prefix)


def sanitize_name(name: str) -> str:
    """Make a section tag safe for use in a filename."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return safe.strip("_") or "section"


def section_to_rows(section_elem: ET.Element, section_tag: str) -> list[dict[str, str]]:
    """Convert a root-level section element into one or more row dicts."""
    children = list(section_elem)
    child_tags = [strip_tag(child.tag) for child in children]

    # Common dmd pattern: section wraps many repeated same-tag records.
    if len(children) > 1 and len(set(child_tags)) == 1:
        rows: list[dict[str, str]] = []
        for child in children:
            row = {"__record_tag": strip_tag(child.tag)}
            flatten_element(child, row)
            rows.append(row)
        return rows

    row = {"__record_tag": section_tag}
    flatten_element(section_elem, row)
    return [row]


def iter_section_rows(xml_path: Path):
    """Yield (section_tag, row) pairs from one XML file."""
    depth = 0
    direct_child_count = 0

    for event, elem in ET.iterparse(xml_path, events=("start", "end")):
        if event == "start":
            depth += 1
            continue

        if depth == 2:
            direct_child_count += 1
            section_tag = strip_tag(elem.tag)
            for row in section_to_rows(elem, section_tag):
                yield section_tag, row
            elem.clear()

        depth -= 1

    if direct_child_count == 0:
        root = ET.parse(xml_path).getroot()
        section_tag = strip_tag(root.tag)
        for row in section_to_rows(root, section_tag):
            yield section_tag, row


def convert_xml_to_section_csvs(xml_path: Path, output_dir: Path) -> list[dict[str, str | int]]:
    """Convert one XML into one CSV per top-level section.

    Returns:
        A list of section-level report rows.
    """
    section_fields: dict[str, set[str]] = defaultdict(set)

    # Pass 1: discover complete field sets for each section.
    for section_tag, row in iter_section_rows(xml_path):
        section_fields[sanitize_name(section_tag)].update(row.keys())

    writers: dict[str, csv.DictWriter] = {}
    files: dict[str, TextIO] = {}
    rows_by_section: Counter[str] = Counter()
    output_file_by_section: dict[str, Path] = {}
    ordered_fields_by_section: dict[str, list[str]] = {}

    try:
        for section_key, fields in section_fields.items():
            output_file = output_dir / f"{xml_path.stem}__{section_key}.csv"
            output_file_by_section[section_key] = output_file
            handle = output_file.open("w", encoding="utf-8", newline="")
            files[section_key] = handle

            ordered_fields = ["__record_tag"] + sorted(k for k in fields if k != "__record_tag")
            ordered_fields_by_section[section_key] = ordered_fields
            writer = csv.DictWriter(handle, fieldnames=ordered_fields)
            writer.writeheader()
            writers[section_key] = writer

        # Pass 2: write rows with the final schema.
        for section_tag, row in iter_section_rows(xml_path):
            section_key = sanitize_name(section_tag)
            writers[section_key].writerow(row)
            rows_by_section[section_key] += 1
    finally:
        for handle in files.values():
            handle.close()

    report_rows: list[dict[str, str | int]] = []
    for section_key in sorted(section_fields):
        report_rows.append(
            {
                "xml_file": xml_path.name,
                "section": section_key,
                "output_csv": output_file_by_section[section_key].name,
                "row_count": rows_by_section[section_key],
                "column_count": len(ordered_fields_by_section[section_key]),
            }
        )

    return report_rows


def write_summary_report(report_rows: list[dict[str, str | int]], report_path: Path) -> None:
    """Write conversion summary statistics to a CSV file."""
    fieldnames = ["xml_file", "section", "output_csv", "row_count", "column_count"]

    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def convert_all_xml(input_dir: Path, output_dir: Path) -> None:
    """Convert all XML files in ``input_dir`` and write section CSVs to ``output_dir``."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for existing_csv in output_dir.glob("*.csv"):
        existing_csv.unlink()

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {input_dir}")

    all_report_rows: list[dict[str, str | int]] = []

    for xml_file in xml_files:
        report_rows = convert_xml_to_section_csvs(xml_file, output_dir)
        all_report_rows.extend(report_rows)

        section_file_count = len(report_rows)
        row_count = sum(int(row["row_count"]) for row in report_rows)
        print(
            f"Converted {xml_file.name} -> {section_file_count} section CSVs "
            f"({row_count} rows total)"
        )

    summary_report_path = output_dir / "conversion_summary_report.csv"
    write_summary_report(all_report_rows, summary_report_path)
    print(f"Wrote summary report: {summary_report_path}")


def denormalize(output_dir: Path) -> None:
    """Build and export a denormalized dm+d table from section CSVs in ``output_dir``."""
    # ==========================================
    # 1. LOAD CORE CLINICAL HIERARCHY TABLES
    # ==========================================
    # AMPP: Actual Medicinal Product Pack (The granular base table)
    ampp_df = pd.read_csv(output_dir / "f_ampp2_3290525__AMPPS.csv")
    # VMPP: Virtual Medicinal Product Pack (Inherits from VMP)
    vmpp_df = pd.read_csv(output_dir / "f_vmpp2_3290525__VMPPS.csv")
    # AMP: Actual Medicinal Product (Inherits from VMP, contains Supplier)
    amp_df = pd.read_csv(output_dir / "f_amp2_3290525__AMPS.csv")
    # VMP: Virtual Medicinal Product (Generic prescribable product)
    vmp_df = pd.read_csv(output_dir / "f_vmp2_3290525__VMPS.csv")
    # VTM: Virtual Therapeutic Moiety (Abstract substance)
    vtm_df = pd.read_csv(output_dir / "f_vtm2_3290525__VTM.csv")

    # ==========================================
    # 2. LOAD LOOKUP TABLES
    # ==========================================
    uom_df = pd.read_csv(output_dir / "f_lookup2_3290525__UNIT_OF_MEASURE.csv")
    ing_lookup_df = pd.read_csv(output_dir / "f_ingredient2_3290525__ING.csv")
    form_lookup_df = pd.read_csv(output_dir / "f_lookup2_3290525__FORM.csv")
    route_lookup_df = pd.read_csv(output_dir / "f_lookup2_3290525__ROUTE.csv")
    supplier_lookup_df = pd.read_csv(output_dir / "f_lookup2_3290525__SUPPLIER.csv")

    # ==========================================
    # 3. PROCESS 1-TO-MANY RELATIONSHIPS (VMP LEVEL)
    # ==========================================

    # --- A. Ingredients & Strengths ---
    vmp_ing_df = pd.read_csv(output_dir / "f_vmp2_3290525__VIRTUAL_PRODUCT_INGREDIENT.csv")

    # Merge ingredient names
    vmp_ing_df = vmp_ing_df.merge(ing_lookup_df[["ISID", "NM"]], on="ISID", how="left")
    vmp_ing_df.rename(columns={"NM": "INGREDIENT_NAME"}, inplace=True)

    # Merge Unit of Measure for the numerator strength
    vmp_ing_df = vmp_ing_df.merge(uom_df[["CD", "CDDT"]], left_on="STRNT_DNMTR_UOMCD", right_on="CD", how="left")
    vmp_ing_df.rename(columns={"CDDT": "NUMERATOR_UOM"}, inplace=True)

    # Combine ingredient name and strength into a readable string (e.g., "Paracetamol 500 mg")
    vmp_ing_df["ING_STRENGTH_STR"] = (
            vmp_ing_df["INGREDIENT_NAME"].astype(str) + " " +
            vmp_ing_df["STRNT_NMRTR_VAL"].astype(str) + " " +
            vmp_ing_df["NUMERATOR_UOM"].astype(str)
    )

    # Aggregate multiple ingredients into one string per VMP separated by " | "
    vmp_ingredients_agg = vmp_ing_df.groupby("VPID")["ING_STRENGTH_STR"].apply(
        lambda x: " | ".join(x.dropna().astype(str))
    ).reset_index(name="AGGREGATED_INGREDIENTS")

    # --- B. Routes of Administration ---
    vmp_route_df = pd.read_csv(output_dir / "f_vmp2_3290525__DRUG_ROUTE.csv")
    vmp_route_df = vmp_route_df.merge(route_lookup_df[["CD", "DESC"]], left_on="ROUTECD", right_on="CD", how="left")

    # Aggregate multiple routes into one string per VMP
    vmp_routes_agg = vmp_route_df.groupby("VPID")["DESC"].apply(
        lambda x: " | ".join(x.dropna().astype(str))
    ).reset_index(name="AGGREGATED_ROUTES")

    # --- C. Dose Forms ---
    vmp_form_df = pd.read_csv(output_dir / "f_vmp2_3290525__DRUG_FORM.csv")
    vmp_form_df = vmp_form_df.merge(form_lookup_df[["CD", "DESC"]], left_on="FORMCD", right_on="CD", how="left")
    # Typically 1 dose form per VMP, but grouping ensures no duplicates
    vmp_forms_agg = vmp_form_df.groupby("VPID")["DESC"].first().reset_index(name="DOSE_FORM")

    # ==========================================
    # 4. BUILD THE FLATTENED MASTER TABLE
    # ==========================================

    # Start with AMPP and join to AMP via APID.
    flat_df = ampp_df.merge(amp_df, on="APID", how="left", suffixes=("_AMPP", "_AMP"))

    # Join AMPP to VMPP.
    flat_df = flat_df.merge(vmpp_df, on=["VPPID", "VPID", "COMBPACKCD"], how="left")

    # Join AMP to VMP (an AMP inherits VMP-level properties).
    flat_df = flat_df.merge(vmp_df, on="VPID", how="left", suffixes=("", "_VMP"))

    # Join VMP to VTM for therapeutic moiety context.
    flat_df = flat_df.merge(vtm_df, on="VTMID", how="left", suffixes=("", "_VTM"))

    # ==========================================
    # 5. ATTACH AGGREGATED FEATURES & LOOKUPS
    # ==========================================

    # Attach aggregated Ingredients, Routes, and Forms to the VMP level
    flat_df = flat_df.merge(vmp_ingredients_agg, on="VPID", how="left")
    flat_df = flat_df.merge(vmp_routes_agg, on="VPID", how="left")
    flat_df = flat_df.merge(vmp_forms_agg, on="VPID", how="left")

    # Attach supplier name to AMP rows via SUPPCD.
    flat_df = flat_df.merge(supplier_lookup_df[["CD", "DESC"]].rename(columns={"DESC": "SUPPLIER_NAME", "CD": "SUPPCD"}), on="SUPPCD", how="left")

    # ==========================================
    # 6. EXPORT
    # ==========================================
    # Optional: Select and reorder columns relevant for OHDSI mapping here before saving
    flat_df.to_csv(output_dir / "dmd_ohdsi_denormalized.csv", index=False)
    print(f"Denormalization complete. Successfully exported {len(flat_df)} rows for OHDSI mapping.")


def sample_denormalized(output_dir: Path, sample_size: int = 100, random_state: int = 42) -> Path:
    """Create a random sample CSV from dmd_ohdsi_denormalized.csv in the same folder."""
    source_path = output_dir / "dmd_ohdsi_denormalized.csv"
    sample_path = output_dir / "dmd_ohdsi_denormalized_sample_100.csv"

    denormalized_df = pd.read_csv(source_path)
    if denormalized_df.empty:
        sampled_df = denormalized_df
    else:
        n = min(sample_size, len(denormalized_df))
        sampled_df = denormalized_df.sample(n=n, random_state=random_state)

    sampled_df.to_csv(sample_path, index=False)
    print(f"Wrote random sample ({len(sampled_df)} rows): {sample_path}")
    return sample_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for XML input and CSV output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../data/dmd_xml_files"),
        help="Directory containing XML files (default: ../data/dmd_xml_files)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../data/dmd_csv_files"),
        help="Directory for generated CSV files (default: ../data/dmd_csv_files)",
    )
    return parser.parse_args()


def main() -> None:
    """Run conversion, denormalization, and sampling in sequence."""
    args = parse_args()
    convert_all_xml(args.input_dir, args.output_dir)
    denormalize(args.output_dir)
    sample_denormalized(args.output_dir)


if __name__ == "__main__":
    main()






