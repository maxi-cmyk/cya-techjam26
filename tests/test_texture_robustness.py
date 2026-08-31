from __future__ import annotations

import copy
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from cya_detector.config import ConfigError, load_config
from cya_detector.evaluation.texture_robustness import (
    STAGE1_CELL_IDS,
    validate_robustness_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/colab.json"


class TextureRobustnessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_texture_robustness_contract_is_exact_and_locked(self) -> None:
        contract = validate_robustness_contract(self.config)

        self.assertEqual(contract.cell_ids, STAGE1_CELL_IDS)
        self.assertEqual(
            contract.variants,
            ("global_only", "local_only", "global_local"),
        )
        self.assertEqual(contract.seeds, (42, 43, 44))
        self.assertEqual(
            contract.controlling_comparators,
            ("global_only", "controlled_rine"),
        )
        self.assertEqual(contract.aggregate_class_tolerance, 0.01)
        self.assertEqual(contract.worst_cell_tolerance, 0.03)

    def test_texture_robustness_contract_is_immutable(self) -> None:
        contract = validate_robustness_contract(self.config)

        with self.assertRaises(FrozenInstanceError):
            contract.worst_cell_tolerance = 0.5

    def test_contract_rejects_cells_absent_from_benchmark_contract(self) -> None:
        candidate = copy.deepcopy(self.config)
        candidate["benchmark_transforms"]["jpeg_quality"] = [70, 50, 30]

        with self.assertRaises(ConfigError):
            validate_robustness_contract(candidate)


if __name__ == "__main__":
    unittest.main()
