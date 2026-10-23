"""Downloader contracts using in-memory responses; no internet or model load."""
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.download_checkpoint import download


class CheckpointDownloadTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ.get("OCR_TEST_TMP", Path.home() / ".codex/cache/tasks/ocr-tests"))
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(dir=root))
        self.target = self.directory / "weights.pth"
        self.manifest = {"url": "https://example.invalid/weights", "size_bytes": 4,
                         "sha256": hashlib.sha256(b"data").hexdigest()}

    def run_download(self, data):
        return download(self.manifest, self.target, self.directory / "cache",
                        opener=lambda *args, **kwargs: io.BytesIO(data))

    def test_pending_release_never_opens_network(self):
        with self.assertRaises(RuntimeError):
            download({"publication_status": "pending"}, self.target, self.directory / "cache",
                     opener=lambda *args, **kwargs: self.fail("unexpected network"))
        self.assertFalse(self.target.exists())

    def test_verified_bytes_installed(self):
        self.run_download(b"data")
        self.assertEqual(self.target.read_bytes(), b"data")

    def test_bad_hash_does_not_install(self):
        with self.assertRaises(ValueError): self.run_download(b"xxxx")
        self.assertFalse(self.target.exists())

    def test_oversized_download_rejected(self):
        with self.assertRaises(ValueError): self.run_download(b"datax")
        self.assertFalse(self.target.exists())

    def test_existing_checkpoint_is_preserved(self):
        self.target.write_bytes(b"keep")
        with self.assertRaises(FileExistsError): self.run_download(b"data")
        self.assertEqual(self.target.read_bytes(), b"keep")

    def test_matching_existing_checkpoint_needs_no_download(self):
        self.target.write_bytes(b"data")
        result = download(self.manifest, self.target, self.directory / "cache",
                          opener=lambda *args, **kwargs: self.fail("unexpected network"))
        self.assertEqual(result, self.target)


if __name__ == "__main__": unittest.main()
