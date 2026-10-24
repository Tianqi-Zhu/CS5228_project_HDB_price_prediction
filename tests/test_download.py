"""The optional download must preserve all pages and refuse incomplete snapshots."""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from public_rent_lookup.download_public import download
from public_rent_lookup.run import sha256


def response(records, total=2):
    return io.BytesIO(json.dumps({"success": True,
        "result": {"total": total, "records": records}}).encode())


def record(identifier):
    return {"_id": identifier, "rent_approval_date": "2025-03", "town": "BEDOK",
            "block": "2", "street_name": "BEDOK NTH ST 3", "flat_type": "4-ROOM",
            "monthly_rent": "4000"}


class PublicDownloadTests(unittest.TestCase):
    def test_pagination_and_saved_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "snapshot.csv"
            with patch("urllib.request.urlopen", side_effect=[response([record(1)]), response([record(2)])]) as fetch, patch("time.sleep"):
                metadata = download(target, page_size=1)
            self.assertEqual(len(pd.read_csv(target)), 2)
            self.assertEqual(metadata["sha256"], sha256(target))
            self.assertIn("offset=1", fetch.call_args_list[1].args[0].full_url)
            with self.assertRaises(FileExistsError):
                download(target)

    def test_incomplete_pagination_creates_no_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "snapshot.csv"
            with patch("urllib.request.urlopen", side_effect=[response([record(1)]), response([])]), patch("time.sleep"):
                with self.assertRaisesRegex(ValueError, "Empty page"):
                    download(target, page_size=1)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
