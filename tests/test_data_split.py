from __future__ import annotations

import copy
import unittest
from collections import Counter

from cya_detector.data.split import assign_grouped_splits


def records_for_split() -> list[dict[str, str]]:
    records = []
    for label in ("authentic", "ai_generated"):
        for index in range(20):
            source_id = f"{label}-{index:02d}"
            records.append(
                {
                    "source_id": source_id,
                    "duplicate_group_id": "",
                    "eligible_for_split": "true",
                    "label": label,
                    "split": "",
                }
            )
    return records


class SplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_stratified(self) -> None:
        first, _ = assign_grouped_splits(records_for_split(), seed=42)
        second, _ = assign_grouped_splits(records_for_split(), seed=42)
        self.assertEqual(
            [(row["source_id"], row["split"]) for row in first],
            [(row["source_id"], row["split"]) for row in second],
        )

        counts = Counter((row["label"], row["split"]) for row in first)
        for label in ("authentic", "ai_generated"):
            self.assertEqual(counts[(label, "seed_train")], 12)
            self.assertEqual(counts[(label, "self_train_pool")], 5)
            self.assertEqual(
                counts[(label, "selection_val")] + counts[(label, "final_test")], 3
            )

    def test_duplicate_group_never_crosses_splits(self) -> None:
        records = records_for_split()
        duplicate = copy.deepcopy(records[0])
        duplicate["source_id"] = "duplicate-secondary"
        records[0]["duplicate_group_id"] = "group-a"
        duplicate["duplicate_group_id"] = "group-a"
        duplicate["eligible_for_split"] = "false"
        records.append(duplicate)

        assigned, _ = assign_grouped_splits(records, seed=42)
        group_splits = {
            row["split"] for row in assigned if row.get("duplicate_group_id") == "group-a"
        }
        self.assertEqual(len(group_splits), 1)


if __name__ == "__main__":
    unittest.main()

