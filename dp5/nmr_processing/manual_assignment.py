"""Strict assignment helpers for manual NMR description files."""

from dataclasses import dataclass, asdict
import logging
import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .errors import NMRAssignmentError


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssignmentGroup:
    """A required experimental signal group for equivalent atoms."""

    label: str
    members: Tuple[str, ...]
    atom_indices: Tuple[int, ...]
    carbon_positions: Tuple[int, ...]


@dataclass
class AssignmentCoverage:
    """Summary of strict manual carbon assignment coverage."""

    mode: str
    required_group_count: int
    experimental_carbon_signal_count: int
    omitted_atoms: List[str]
    extra_experimental_shifts: List[float]
    missing_groups: List[str]
    assignments: List[dict]
    status: str
    rescaling_applied: bool = True
    secondary_rescaling_skipped_for_grouping: bool = False

    def to_dict(self):
        return asdict(self)


def canonical_group_label(labels: Sequence[str]) -> str:
    """Return a stable comma-separated group label ordered by atom index."""

    return ",".join(sorted(labels, key=_label_sort_key))


def atom_labels_from_molecule(mol) -> set:
    """Build the full atom-label set from the molecule atom order."""

    return {f"{symbol}{idx}" for idx, symbol in enumerate(mol.atoms, start=1)}


def validate_manual_constraints(description, mol) -> None:
    """Validate manual labels, equivalence groups, and omissions for a molecule."""

    valid_labels = atom_labels_from_molecule(mol)
    equivalent_members = []

    for group in description.equivalence_groups:
        _validate_labels(group, valid_labels, "equivalence group")
        if len({_label_element(label) for label in group}) > 1:
            raise NMRAssignmentError(
                f"Equivalence group {canonical_group_label(group)} mixes nuclei"
            )
        equivalent_members.extend(group)

    duplicated_equivalents = _duplicates(equivalent_members)
    if duplicated_equivalents:
        raise NMRAssignmentError(
            "Atom appears in more than one equivalence group: "
            + ", ".join(duplicated_equivalents)
        )

    _validate_labels(description.omitted_labels, valid_labels, "OMIT section")
    overlap = sorted(
        set(equivalent_members).intersection(description.omitted_labels),
        key=_label_sort_key,
    )
    if overlap:
        raise NMRAssignmentError(
            "Atom appears in both equivalence and OMIT sections: "
            + ", ".join(overlap)
        )

    for shift in description.carbon_shifts:
        _validate_labels(shift.labels, valid_labels, "carbon shift assignment")
        for label in shift.labels:
            if _label_element(label) != "C":
                raise NMRAssignmentError(
                    f"Carbon shift {shift.value:g} is assigned to non-carbon label {label}"
                )

    for shift in description.proton_shifts:
        _validate_labels(shift.labels, valid_labels, "proton shift assignment")
        for label in shift.labels:
            if _label_element(label) != "H":
                raise NMRAssignmentError(
                    f"Proton shift {shift.value:g} is assigned to non-proton label {label}"
                )


def build_carbon_groups(description, mol) -> List[AssignmentGroup]:
    """Collapse equivalent carbons and singleton carbons into required groups."""

    validate_manual_constraints(description, mol)

    carbon_label_to_position = {label: i for i, label in enumerate(mol.C_labels)}
    omitted = set(description.omitted_labels)
    grouped_members = set()
    groups = []

    for raw_group in description.equivalence_groups:
        group = tuple(sorted(raw_group, key=_label_sort_key))
        if _label_element(group[0]) != "C":
            continue
        if any(label in omitted for label in group):
            continue
        _validate_predicted_carbon_labels(group, carbon_label_to_position)
        grouped_members.update(group)
        groups.append(_make_group(group, carbon_label_to_position))

    for label in mol.C_labels:
        if label in omitted or label in grouped_members:
            continue
        groups.append(_make_group((str(label),), carbon_label_to_position))

    return groups


def assign_manual_carbon_strict(
    mol,
    description,
    allow_extra_peaks: bool = False,
    allow_missing_peaks: bool = False,
):
    """Assign manual carbon shifts to required strict carbon groups."""

    groups = build_carbon_groups(description, mol)
    group_by_member = {
        member: group_index
        for group_index, group in enumerate(groups)
        for member in group.members
    }

    exp_values = np.array([shift.value for shift in description.carbon_shifts], dtype=float)
    group_calc = _group_calculated_shifts(mol.conformer_C_pred, groups)

    required_count = len(groups)
    experimental_count = len(exp_values)
    if experimental_count > required_count and not allow_extra_peaks:
        raise NMRAssignmentError(
            f"Manual carbon description has {experimental_count} shifts but "
            f"{required_count} required carbon groups"
        )
    if experimental_count < required_count and not allow_missing_peaks:
        raise NMRAssignmentError(
            f"Manual carbon description has {experimental_count} shifts but "
            f"{required_count} required carbon groups"
        )

    fixed_group_to_exp = {}
    fixed_exp_to_group = {}
    omitted = set(description.omitted_labels)
    for exp_index, shift in enumerate(description.carbon_shifts):
        if not shift.labels:
            continue
        omitted_shift_labels = [label for label in shift.labels if label in omitted]
        if omitted_shift_labels:
            raise NMRAssignmentError(
                f"Carbon shift {shift.value:g} is assigned to omitted label(s): "
                + ", ".join(omitted_shift_labels)
            )
        constrained_groups = {
            group_by_member[label]
            for label in shift.labels
            if label in group_by_member
        }
        if len(constrained_groups) != 1:
            if not constrained_groups:
                raise NMRAssignmentError(
                    f"Carbon shift {shift.value:g} is assigned only to omitted atoms"
                )
            raise NMRAssignmentError(
                f"Carbon shift {shift.value:g} spans incompatible groups: "
                + ", ".join(shift.labels)
            )
        group_index = constrained_groups.pop()
        if group_index in fixed_group_to_exp:
            raise NMRAssignmentError(
                f"Multiple experimental shifts are assigned to group "
                f"{groups[group_index].label}"
            )
        fixed_group_to_exp[group_index] = exp_index
        fixed_exp_to_group[exp_index] = group_index

    assigned_group_to_exp = dict(fixed_group_to_exp)
    remaining_groups = [
        i for i in range(required_count) if i not in assigned_group_to_exp
    ]
    remaining_exps = [
        i for i in range(experimental_count) if i not in fixed_exp_to_group
    ]

    if remaining_groups and remaining_exps:
        boltzmann_group_calc = _boltzmann_group_shifts(mol, group_calc)
        cost = np.abs(
            boltzmann_group_calc[remaining_groups, np.newaxis]
            - exp_values[np.newaxis, remaining_exps]
        )
        group_rows, exp_cols = linear_sum_assignment(cost)
        for row, col in zip(group_rows, exp_cols):
            assigned_group_to_exp[remaining_groups[row]] = remaining_exps[col]

    assigned_exps = set(assigned_group_to_exp.values())
    missing_groups = [
        groups[i].label for i in range(required_count) if i not in assigned_group_to_exp
    ]
    extra_exps = [
        float(exp_values[i]) for i in range(experimental_count) if i not in assigned_exps
    ]

    if missing_groups and not allow_missing_peaks:
        raise NMRAssignmentError(
            "Missing experimental shifts for required carbon groups: "
            + ", ".join(missing_groups)
        )
    if extra_exps and not allow_extra_peaks:
        raise NMRAssignmentError(
            "Unassigned experimental carbon shifts: "
            + ", ".join(f"{value:g}" for value in extra_exps)
        )

    group_exp = np.full(required_count, np.nan, dtype=float)
    assignments = []
    for group_index, group in enumerate(groups):
        exp_index = assigned_group_to_exp.get(group_index)
        exp_shift = float(exp_values[exp_index]) if exp_index is not None else math.nan
        group_exp[group_index] = exp_shift
        assignments.append(
            {
                "group": group.label,
                "members": list(group.members),
                "experimental_shift": exp_shift,
                "status": "assigned" if exp_index is not None else "missing",
            }
        )

    atom_exp = np.full(len(mol.C_labels), np.nan, dtype=float)
    for group, exp_shift in zip(groups, group_exp):
        for carbon_position in group.carbon_positions:
            atom_exp[carbon_position] = exp_shift

    coverage = AssignmentCoverage(
        mode="strict",
        required_group_count=required_count,
        experimental_carbon_signal_count=experimental_count,
        omitted_atoms=sorted(description.omitted_labels, key=_label_sort_key),
        extra_experimental_shifts=extra_exps,
        missing_groups=missing_groups,
        assignments=assignments,
        status="complete" if not missing_groups and not extra_exps else "partial",
    )

    _store_group_assignment(mol, groups, group_calc, group_exp, coverage)
    logger.info(
        "Strict carbon NMR assignment coverage for %s: %s/%s groups",
        mol,
        required_count - len(missing_groups),
        required_count,
    )
    return atom_exp


def _store_group_assignment(mol, groups, group_calc, group_exp, coverage):
    mol.C_assignment_mode = "strict"
    mol.C_group_labels = np.array([group.label for group in groups])
    mol.C_group_members = [list(group.members) for group in groups]
    mol.C_group_atom_indices = [list(group.atom_indices) for group in groups]
    mol.C_group_carbon_positions = [list(group.carbon_positions) for group in groups]
    mol.conformer_C_group_pred = group_calc
    mol.C_group_exp = group_exp
    mol.C_assignment_coverage = coverage.to_dict()


def _group_calculated_shifts(conformer_C_pred, groups):
    conformer_C_pred = np.array(conformer_C_pred, dtype=float)
    grouped = []
    for group in groups:
        grouped.append(conformer_C_pred[:, group.carbon_positions].mean(axis=1))
    if not grouped:
        return np.empty((conformer_C_pred.shape[0], 0), dtype=float)
    return np.stack(grouped, axis=1)


def _boltzmann_group_shifts(mol, group_calc):
    return (mol.populations[:, np.newaxis] * group_calc).sum(axis=0)


def _make_group(labels, carbon_label_to_position):
    members = tuple(sorted(labels, key=_label_sort_key))
    return AssignmentGroup(
        label=canonical_group_label(members),
        members=members,
        atom_indices=tuple(_label_index(label) - 1 for label in members),
        carbon_positions=tuple(carbon_label_to_position[label] for label in members),
    )


def _validate_labels(labels: Iterable[str], valid_labels: set, context: str) -> None:
    invalid = sorted(set(labels).difference(valid_labels), key=_label_sort_key)
    if invalid:
        raise NMRAssignmentError(
            f"Invalid label(s) in {context}: " + ", ".join(invalid)
        )


def _validate_predicted_carbon_labels(labels, carbon_label_to_position):
    missing = [label for label in labels if label not in carbon_label_to_position]
    if missing:
        raise NMRAssignmentError(
            "Carbon label(s) missing from calculated shift list: "
            + ", ".join(missing)
        )


def _duplicates(labels):
    seen = set()
    duplicated = []
    for label in labels:
        if label in seen and label not in duplicated:
            duplicated.append(label)
        seen.add(label)
    return sorted(duplicated, key=_label_sort_key)


def _label_element(label: str) -> str:
    return "".join(ch for ch in label if ch.isalpha())


def _label_index(label: str) -> int:
    return int("".join(ch for ch in label if ch.isdigit()))


def _label_sort_key(label: str):
    return _label_element(label), _label_index(label)
