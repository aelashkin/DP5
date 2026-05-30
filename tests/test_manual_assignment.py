import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dp5.analysis.dp5 import (
    DP5,
    ErrorDP5ProbabilityCalculator,
    _skip_secondary_rescaling_for_grouping,
)
from dp5.nmr_processing.description_files import parse_manual_description
from dp5.nmr_processing.manual_assignment import (
    NMRAssignmentError,
    assign_manual_carbon_strict,
    build_carbon_groups,
)


FIXTURES = Path(__file__).parent / "fixtures"


def parse_text(text):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "description.nmr"
        path.write_text(text)
        return parse_manual_description(path)


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


class FakeDP5Mol:
    def __init__(self, strict=False, labels=None):
        self.input_file = "fake.sdf"
        self.C_labels = np.array(labels or ["C1"])
        self.C_exp = np.array([10.0] * len(self.C_labels))
        if strict:
            self.C_assignment_mode = "strict"
            self.C_group_labels = np.array(["C1"])
            self.C_group_members = [["C1"]]
            self.C_group_atom_indices = [[0]]
            self.C_group_exp = np.array([10.0])
            self.C_assignment_coverage = {
                "mode": "strict",
                "required_group_count": 1,
                "experimental_carbon_signal_count": 1,
                "omitted_atoms": [],
                "extra_experimental_shifts": [],
                "missing_groups": [],
                "assignments": [
                    {
                        "group": "C1",
                        "members": ["C1"],
                        "experimental_shift": 10.0,
                        "status": "assigned",
                    }
                ],
                "status": "complete",
                "rescaling_applied": True,
                "secondary_rescaling_skipped_for_grouping": False,
            }


class FakeDP5Calculator:
    def __init__(self, labels=None, skips=None):
        self.calls = 0
        self.labels = labels or np.array(["C1"])
        self.secondary_rescaling_skipped_for_grouping = skips or [False]

    def __call__(self, mols):
        self.calls += 1
        return (
            [self.labels],
            [np.array([10.0])],
            [np.array([10.0])],
            [np.array([0.0])],
            [np.array([[0.2]])],
            [np.array([0.8])],
            [0.8],
        )


def make_dp5(output_folder, calculator, dft_shifts=False):
    (output_folder / "dp5").mkdir()
    dp5 = DP5.__new__(DP5)
    dp5.output_folder = output_folder
    dp5.dft_shifts = dft_shifts
    dp5.C_DP5 = calculator
    return dp5


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

    def test_parser_normalizes_whitespace_case_any_omit_and_explicit_labels(self):
        description = parse_text(
            """
            10.0( c 1 ), 20.0(C2), 30.0(any)

            1.0( h 3 )

            c 1, C2
            omit h 3
            """
        )

        self.assertEqual(description.carbon_shifts[0].labels, ("C1",))
        self.assertEqual(description.carbon_shifts[1].labels, ("C2",))
        self.assertEqual(description.carbon_shifts[2].labels, ())
        self.assertIn(("C1", "C2"), description.equivalence_groups)
        self.assertEqual(description.omitted_labels, ("H3",))

    def test_parser_invalid_label_raises_assignment_error(self):
        with self.assertRaises(NMRAssignmentError):
            parse_text("10.0(C@)\n\n0.0(any)\n")

    def test_parser_malformed_shift_raises_assignment_error(self):
        with self.assertRaises(NMRAssignmentError):
            parse_text("not-a-shift(any)\n\n0.0(any)\n")

    def test_dft_secondary_rescaling_allowed_for_strict_singletons(self):
        mol = FakeDP5Mol(strict=True, labels=["C1", "C2"])

        self.assertFalse(
            _skip_secondary_rescaling_for_grouping(
                mol,
                [np.array([0]), np.array([1])],
            )
        )

    def test_dft_secondary_rescaling_skipped_for_grouping_or_omission(self):
        mol = FakeDP5Mol(strict=True, labels=["C1", "C2"])

        self.assertTrue(
            _skip_secondary_rescaling_for_grouping(mol, [np.array([0, 1])])
        )
        self.assertTrue(
            _skip_secondary_rescaling_for_grouping(mol, [np.array([0])])
        )

    def test_dft_secondary_rescaling_is_per_molecule(self):
        calculator = ErrorDP5ProbabilityCalculator.__new__(
            ErrorDP5ProbabilityCalculator
        )
        calculator.dp5_correct_kde = lambda vector: np.ones_like(vector) * 0.8
        calculator.dp5_incorrect_kde = lambda vector: np.ones_like(vector) * 0.2

        scaled_probs, scaled_total = calculator.rescale_probabilities(
            [np.array([0.2]), np.array([0.3])],
            [0.0, 0.0],
            skip_secondary_rescaling=[False, True],
        )

        self.assertAlmostEqual(scaled_probs[0][0], 0.8)
        self.assertAlmostEqual(scaled_total[0], 0.8)
        self.assertAlmostEqual(scaled_probs[1][0], 0.3)
        self.assertAlmostEqual(scaled_total[1], np.exp(np.log(0.3 + 1e-6)))

    def test_legacy_dp5_data_dictionary_does_not_save_group_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir)
            calculator = FakeDP5Calculator()
            dp5 = make_dp5(output_folder, calculator)

            dp5([FakeDP5Mol(strict=False)])

            with open(output_folder / "dp5" / "data_dic.p", "rb") as f:
                data = pickle.load(f)
            self.assertEqual(calculator.calls, 1)
            self.assertNotIn("Cassignment_coverage", data)
            self.assertNotIn("Cgroup_labels", data)
            self.assertNotIn("cache_signature", data)

    def test_strict_cache_is_not_reused_by_legacy_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_folder = Path(tmpdir)
            strict_calculator = FakeDP5Calculator()
            strict_dp5 = make_dp5(output_folder, strict_calculator)
            strict_dp5([FakeDP5Mol(strict=True)])

            legacy_calculator = FakeDP5Calculator()
            legacy_dp5 = DP5.__new__(DP5)
            legacy_dp5.output_folder = output_folder
            legacy_dp5.dft_shifts = False
            legacy_dp5.C_DP5 = legacy_calculator

            legacy_dp5([FakeDP5Mol(strict=False)])

            with open(output_folder / "dp5" / "data_dic.p", "rb") as f:
                data = pickle.load(f)
            self.assertEqual(legacy_calculator.calls, 1)
            self.assertNotIn("Cassignment_coverage", data)
            self.assertNotIn("Cgroup_labels", data)
            self.assertNotIn("cache_signature", data)


if __name__ == "__main__":
    unittest.main()
