import unittest
from pathlib import Path

import numpy as np

from dp5.nmr_processing.description_files import parse_manual_description
from dp5.nmr_processing.manual_assignment import (
    NMRAssignmentError,
    assign_manual_carbon_strict,
    build_carbon_groups,
)


FIXTURES = Path(__file__).parent / "fixtures"


class FakeMol:
    def __init__(self):
        self.input_file = "fake.sdf"
        self.atoms = np.array(
            [
                "C",
                "C",
                "C",
                "C",
                "C",
                "N",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "N",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
                "C",
            ]
        )
        self.C_labels = np.array(
            [
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C7",
                "C8",
                "C9",
                "C10",
                "C11",
                "C12",
                "C13",
                "C15",
                "C16",
                "C17",
                "C18",
                "C19",
                "C20",
                "C21",
                "C22",
                "C23",
            ]
        )
        shifts_by_label = {
            "C1": 20.3,
            "C19": 20.3,
            "C2": 126.7,
            "C18": 126.7,
            "C3": 129.7,
            "C17": 129.7,
            "C20": 129.7,
            "C23": 129.7,
            "C4": 113.1,
            "C16": 113.1,
            "C21": 113.1,
            "C22": 113.1,
            "C5": 140.1,
            "C15": 140.1,
            "C7": 50.1,
            "C12": 50.1,
            "C8": 39.5,
            "C11": 39.5,
            "C9": 36.3,
            "C13": 36.3,
            "C10": 29.6,
        }
        self.conformer_C_pred = np.array(
            [[shifts_by_label[label] for label in self.C_labels]]
        )
        self._populations = np.array([1.0])

    @property
    def populations(self):
        return self._populations


class ManualAssignmentTests(unittest.TestCase):
    def test_parser_normalizes_labels_and_sections(self):
        description = parse_manual_description(FIXTURES / "cyclopentane_correct.nmr")

        self.assertEqual(description.carbon_shifts[-1].value, 20.3)
        self.assertEqual(description.carbon_shifts[-1].labels, ())
        self.assertIn(("C1", "C19"), description.equivalence_groups)

    def test_builds_required_groups_from_equivalence_and_singletons(self):
        groups = build_carbon_groups(
            parse_manual_description(FIXTURES / "cyclopentane_correct.nmr"),
            FakeMol(),
        )

        self.assertEqual(len(groups), 9)
        self.assertIn("C1,C19", [group.label for group in groups])
        self.assertIn("C10", [group.label for group in groups])

    def test_wrong_same_count_file_assigns_every_group(self):
        mol = FakeMol()
        description = parse_manual_description(
            FIXTURES / "cyclopentane_wrong_same_count.nmr"
        )

        assign_manual_carbon_strict(mol, description)

        self.assertEqual(mol.C_assignment_coverage["required_group_count"], 9)
        self.assertEqual(mol.C_assignment_coverage["experimental_carbon_signal_count"], 9)
        self.assertFalse(mol.C_assignment_coverage["missing_groups"])
        self.assertFalse(mol.C_assignment_coverage["extra_experimental_shifts"])
        assigned = {
            item["group"]: item["experimental_shift"]
            for item in mol.C_assignment_coverage["assignments"]
        }
        self.assertIn("C1,C19", assigned)
        self.assertTrue(np.isfinite(assigned["C1,C19"]))

    def test_correct_file_has_lower_forced_absolute_error_than_wrong_file(self):
        correct_mol = FakeMol()
        wrong_mol = FakeMol()

        assign_manual_carbon_strict(
            correct_mol,
            parse_manual_description(FIXTURES / "cyclopentane_correct.nmr"),
        )
        assign_manual_carbon_strict(
            wrong_mol,
            parse_manual_description(FIXTURES / "cyclopentane_wrong_same_count.nmr"),
        )

        correct_error = np.abs(
            correct_mol.conformer_C_group_pred[0] - correct_mol.C_group_exp
        ).sum()
        wrong_error = np.abs(
            wrong_mol.conformer_C_group_pred[0] - wrong_mol.C_group_exp
        ).sum()
        self.assertLess(correct_error, wrong_error)

    def test_rejects_duplicate_equivalence_membership(self):
        description = parse_manual_description(FIXTURES / "cyclopentane_correct.nmr")
        bad_description = description.__class__(
            carbon_shifts=description.carbon_shifts,
            proton_shifts=description.proton_shifts,
            equivalence_groups=description.equivalence_groups + (("C1", "C2"),),
            omitted_labels=description.omitted_labels,
        )

        with self.assertRaises(NMRAssignmentError):
            build_carbon_groups(bad_description, FakeMol())

    def test_rejects_shift_assignment_spanning_groups(self):
        description = parse_manual_description(FIXTURES / "cyclopentane_correct.nmr")
        first_shift = description.carbon_shifts[0].__class__(
            value=description.carbon_shifts[0].value,
            labels=("C1", "C2"),
            raw=description.carbon_shifts[0].raw,
        )
        bad_description = description.__class__(
            carbon_shifts=(first_shift,) + description.carbon_shifts[1:],
            proton_shifts=description.proton_shifts,
            equivalence_groups=description.equivalence_groups,
            omitted_labels=description.omitted_labels,
        )

        with self.assertRaises(NMRAssignmentError):
            assign_manual_carbon_strict(FakeMol(), bad_description)


if __name__ == "__main__":
    unittest.main()
