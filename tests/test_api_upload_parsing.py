import io
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from api import server  # noqa: E402


class DummyUpload:
    def __init__(self, filename: str, raw: bytes):
        self.filename = filename
        self._raw = raw

    def read(self) -> bytes:
        return self._raw


def _make_upload(filename: str, raw: bytes) -> DummyUpload:
    return DummyUpload(filename, raw)


def test_parse_upload_accepts_utf8_csv():
    raw = (
        "motion_product_id,mfr_name,mfr_part_number,web_desc\n"
        "1,Acme,ABC-1,Standard bearing\n"
    ).encode("utf-8")

    df, headers = server._parse_upload(_make_upload("products.csv", raw))

    assert headers == ["motion_product_id", "mfr_name", "mfr_part_number", "web_desc"]
    assert df.iloc[0].to_dict()["mfr_name"] == "Acme"


def test_parse_upload_accepts_utf8_bom_csv():
    raw = (
        "\ufeffmotion_product_id,mfr_name,mfr_part_number,web_desc\n"
        "2,Acme,ABC-2,BOM export\n"
    ).encode("utf-8")

    df, headers = server._parse_upload(_make_upload("products.csv", raw))

    assert headers == ["motion_product_id", "mfr_name", "mfr_part_number", "web_desc"]
    assert df.iloc[0].to_dict()["web_desc"] == "BOM export"


def test_parse_upload_accepts_cp1252_csv():
    raw = (
        "motion_product_id,mfr_name,mfr_part_number,web_desc\n"
        "3,Acme,ABC-3,Caf\xe9 motor\n"
    ).encode("cp1252")

    df, _ = server._parse_upload(_make_upload("products.csv", raw))

    assert df.iloc[0].to_dict()["web_desc"] == "Café motor"


def test_parse_upload_rejects_binary_csv_with_clear_error():
    raw = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\xff\xfe\x00"

    with pytest.raises(ValueError, match="binary data|Unable to decode CSV upload|No columns to parse from file|Error tokenizing data"):
        server._parse_upload(_make_upload("products.csv", raw))


def test_parse_upload_accepts_cp1252_json():
    payload = [
        {
            "motion_product_id": "4",
            "mfr_name": "Acme",
            "mfr_part_number": "ABC-4",
            "web_desc": "Caf\xe9 motor",
        }
    ]
    raw = json.dumps(payload, ensure_ascii=False).encode("cp1252")

    df, headers = server._parse_upload(_make_upload("products.json", raw))

    assert headers == ["motion_product_id", "mfr_name", "mfr_part_number", "web_desc"]
    assert df.iloc[0].to_dict()["web_desc"] == "Café motor"


def test_parse_upload_rejects_invalid_json_with_clear_error():
    raw = b'{"products": [invalid]}'

    with pytest.raises(ValueError, match="Invalid JSON upload"):
        server._parse_upload(_make_upload("products.json", raw))


def test_upload_route_traces_and_quarantines_failed_upload(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("UPLOAD_TRACE", "1")
    monkeypatch.setattr(server, "DEBUG_UPLOAD_DIR", tmp_path)

    client = server.app.test_client()
    raw = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\xff\xfe\x00"

    with caplog.at_level("INFO", logger="api"):
        response = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(raw), "broken.csv")},
            content_type="multipart/form-data",
        )

    assert response.status_code == 422
    payload = response.get_json()
    assert payload["error"].startswith("Parse error:")

    saved_files = list(tmp_path.iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].suffix == ".csv"
    assert saved_files[0].read_bytes() == raw

    assert "Incoming request content_type=" in caplog.text
    assert "Parser branch selected: csv" in caplog.text
    assert "Saved failing upload sample to" in caplog.text
