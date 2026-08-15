import unittest
from collections import Counter

from prepare_calibration_root_manifests import build_manifests


class PrepareCalibrationRootManifestTest(unittest.TestCase):
    def test_recovers_prior_shards_and_fills_without_overlap(self):
        histories = [
            (["a", "b", "shared"], Counter({"a": 20, "b": 10, "shared": 2})),
            (["c", "shared"], Counter({"c": 30, "shared": 18})),
        ]
        manifests = build_manifests(
            list("abcdefghij") + ["shared"],
            histories,
            [5, 5],
            seed=42,
        )
        self.assertEqual(len(manifests[0]), 5)
        self.assertEqual(len(manifests[1]), 5)
        self.assertIn("a", manifests[0])
        self.assertIn("b", manifests[0])
        self.assertIn("c", manifests[1])
        self.assertNotIn("shared", manifests[0])
        self.assertIn("shared", manifests[1])
        self.assertEqual(len(set(manifests[0]) & set(manifests[1])), 0)


if __name__ == "__main__":
    unittest.main()
