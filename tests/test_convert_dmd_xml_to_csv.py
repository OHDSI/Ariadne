import csv
import importlib.util
from pathlib import Path


def _load_converter_module():
    module_path = Path(__file__).resolve().parents[1] / "sandbox" / "convert_dmd_xml_to_csv.py"
    spec = importlib.util.spec_from_file_location("convert_dmd_xml_to_csv", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_all_xml_writes_summary_report(tmp_path):
    converter = _load_converter_module()

    input_dir = tmp_path / "xml"
    output_dir = tmp_path / "csv"
    input_dir.mkdir()

    xml_content = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<ROOT>
  <AMPS>
    <AMP><CD>A1</CD></AMP>
    <AMP><CD>A2</CD></AMP>
  </AMPS>
  <LOOKUP>
    <ITEM><ID>1</ID></ITEM>
  </LOOKUP>
</ROOT>
"""
    (input_dir / "sample.xml").write_text(xml_content, encoding="utf-8")

    converter.convert_all_xml(input_dir, output_dir)

    summary_path = output_dir / "conversion_summary_report.csv"
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2

    rows_by_section = {row["section"]: row for row in rows}
    assert rows_by_section["AMPS"]["row_count"] == "2"
    assert rows_by_section["LOOKUP"]["row_count"] == "1"

    assert (output_dir / "sample__AMPS.csv").exists()
    assert (output_dir / "sample__LOOKUP.csv").exists()

