from dataclasses import dataclass
import re
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManualShift:
    """One manually described experimental shift."""

    value: float
    labels: tuple
    raw: str


@dataclass(frozen=True)
class ManualNMRDescription:
    """Parsed manual NMR description file."""

    carbon_shifts: tuple
    proton_shifts: tuple
    equivalence_groups: tuple
    omitted_labels: tuple


def process_description(nmr_source):
    """Parse a legacy DP4 description file.

    The text format stores carbon shifts on the first line, proton shifts on the
    third line, and optional equivalence or omission directives below. This
    parser preserves the loose syntax used by historical DP4 workflows so that
    the modern package can still consume manually curated descriptions when raw
    FID data are not available.

    :param nmr_source: Path to the description file.
    :type nmr_source: str or pathlib.Path
    :returns: Carbon labels, carbon shifts, proton labels, proton shifts,
        equivalence groups, and omitted labels.
    :rtype: tuple[list, list, list, list, list, list]
    """
    description = parse_manual_description(nmr_source)
    logger.info("Read carbon NMR shifts")
    C_labels = [list(shift.labels) for shift in description.carbon_shifts]
    C_exp = [shift.value for shift in description.carbon_shifts]
    logger.info("Read proton NMR shifts")
    H_labels = [list(shift.labels) for shift in description.proton_shifts]
    H_exp = [shift.value for shift in description.proton_shifts]

    return (
        C_labels,
        C_exp,
        H_labels,
        H_exp,
        [list(group) for group in description.equivalence_groups],
        list(description.omitted_labels),
    )


def parse_manual_description(nmr_source):
    """Parse a manual NMR description into structured records."""

    with open(nmr_source) as f:
        sections = _split_sections(f.readlines())

    carbon_line = " ".join(sections[0]) if len(sections) > 0 else ""
    proton_line = " ".join(sections[1]) if len(sections) > 1 else ""
    constraint_lines = [line for section in sections[2:] for line in section]

    equivalents = []
    omits = []
    for line in constraint_lines:
        if re.match(r"^\s*OMIT\b", line, flags=re.IGNORECASE):
            omit_text = re.sub(r"^\s*OMIT\b", "", line, flags=re.IGNORECASE).strip()
            omits.extend(_parse_label_list(omit_text))
        else:
            labels = _parse_label_list(line)
            if labels:
                equivalents.append(tuple(labels))

    return ManualNMRDescription(
        carbon_shifts=tuple(_parse_shift_entries(carbon_line)),
        proton_shifts=tuple(_parse_shift_entries(proton_line)),
        equivalence_groups=tuple(equivalents),
        omitted_labels=tuple(omits),
    )


def _parse_description(exp):

    shifts = _parse_shift_entries(exp)
    expLabels = [list(shift.labels) for shift in shifts]
    expShifts = [shift.value for shift in shifts]
    logger.info(", ".join(str(value) for value in expShifts))

    return expLabels, expShifts


def _split_sections(lines):
    sections = []
    current = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                sections.append(current)
                current = []
            continue
        current.append(stripped)
    if current:
        sections.append(current)
    return sections


def _parse_shift_entries(exp):
    if not exp:
        return []

    shifts = []
    for entry in _split_top_level_commas(exp):
        entry = entry.strip()
        if not entry:
            continue
        match = re.match(
            r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"\s*(?:\((.*?)\))?\s*$",
            entry,
        )
        if match is None:
            raise ValueError(f"Could not parse NMR shift entry: {entry}")
        value, label_text = match.groups()
        shifts.append(
            ManualShift(
                value=float(value),
                labels=tuple(_parse_label_list(label_text or "")),
                raw=entry,
            )
        )
    return shifts


def _split_top_level_commas(text):
    entries = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            entries.append(text[start:index])
            start = index + 1
    entries.append(text[start:])
    return entries


def _parse_label_list(text):
    text = re.sub(r"\bor\b", ",", text or "", flags=re.IGNORECASE)
    labels = []
    for raw_label in text.split(","):
        raw_label = raw_label.strip()
        if not raw_label or raw_label.lower() == "any":
            continue
        labels.append(_normalize_label(raw_label))
    return labels


def _normalize_label(label):
    match = re.match(r"^\s*([A-Za-z]+)\s*([0-9]+)\s*$", label)
    if match is None:
        raise ValueError(f"Invalid atom label: {label}")
    element, index = match.groups()
    element = element[0].upper() + element[1:].lower()
    return f"{element}{int(index)}"


def pairwise_assignment(calculated, experimental: list):
    sorted_calc = sorted(calculated, reverse=True)
    sorted_exp = sorted(experimental, reverse=True)
    assigned = [None] * len(calculated)

    for calc, exp in zip(sorted_calc, sorted_exp):
        index = list(calculated).index(calc)
        assigned[index] = exp

    return assigned


def matching_assignment(calculated, experimental, threshold=40):

    scaled = calculated

    # Create a bipartite graph
    G = nx.Graph()

    # Add nodes for calc and exp with a bipartite attribute
    calc_nodes = [("calc", i) for i in range(len(scaled))]
    exp_nodes = [("exp", i) for i in range(len(experimental))]
    G.add_nodes_from(calc_nodes, bipartite=0)
    G.add_nodes_from(exp_nodes, bipartite=1)

    # Add edges for all pairs within the threshold
    for i, c in enumerate(scaled):
        for j, e in enumerate(experimental):
            deviation = abs(c - e)
            if deviation <= threshold:
                G.add_edge(("calc", i), ("exp", j), weight=-deviation)

    # Find the maximum matching
    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)

    matched_pair_indices = set()

    for pair in matching:
        # pair could be (('calc', i), ('exp', j)) or the reverse
        calc, exp = sorted(pair)
        # after sorting, 'calc' goes before 'exp'
        calc_index = calc[1]
        exp_index = exp[1]

        matched_pair_indices.add((calc_index, exp_index))

    # Now, use these indices to access values in calc and exp
    matched_pairs = [(calculated[i], experimental[j]) for i, j in matched_pair_indices]
    assigned = [None] * len(calculated)
    for calc_shift, exp_shift in matched_pairs:
        index = list(calculated).index(calc_shift)
        assigned[index] = exp_shift

    return assigned
