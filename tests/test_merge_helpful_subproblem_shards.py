import json
import tempfile
import unittest
from pathlib import Path


class HelpfulShardTest(unittest.TestCase):
    def test_shard_metadata_is_present_in_calibration_source(self):
        source = Path("calibrate_helpful_subproblems.py").read_text(encoding="utf-8")
        self.assertIn("--shard-index", source)
        self.assertIn("--num-shards", source)


if __name__ == "__main__":
    unittest.main()
