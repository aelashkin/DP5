An improved version of DP5 analysis developed by Howarth (DOI:[10.1039/D1SC04406K](https://doi.org/10.1039/D1SC04406K)). This codebase is refactored for legibility and maintainability.

We strongly recommend using a separate python environment created via `conda`, `uv`, or other solution of your choice to run this programme.
DP5 currently supports `python>=3.9,<=3.11` (due to the TensorFlow dependency range).

If you do not have `uv` installed yet:
- macOS (Homebrew): `brew install uv`
- Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `wget -qO- https://astral.sh/uv/install.sh | sh`)

To get started:
- clone this repository using `git clone https://github.com/ruslankotl/DP5.git`
- navigate to the folder on your machine
- create and activate a compatible environment, for example `uv venv --python 3.10 .venv && source .venv/bin/activate`
- install via `uv pip install -e .` 
- to also install documentation build dependencies, run `uv pip install -e ".[dev]"`
- run `pydp4 -s <SD_FILE> -n <NMR_FILE> -w w`

  For example, run:
```
cd reassignments/S11
pydp4 -s S11a_.sdf S11b_.sdf -n S11_NMR -w w
```

NMR files are provided as a list of shifts with assignments, e.g., `
153.0(any),125.6(any)`

Manual NMR descriptions are strict by default. Carbon equivalence groups collapse
multiple atoms into one required experimental signal, omitted atoms are excluded,
and every non-omitted carbon group must be assigned exactly one 13C signal unless
an allow flag is used.

Manual description format:
```
140.1(any),129.7(any),126.7(C10),20.3(C14,C23)

0.0(any)

C1,C3
C4,C5
C14,C23
OMIT C19,H51
```

Section 1 is comma-separated carbon shifts. Section 2 is comma-separated proton
shifts; carbon-only callers may use `0.0(any)`. Later sections may contain one
equivalence group per line and optional `OMIT` lines. Labels are element symbols
plus 1-based atom indices from the supplied molecule/SDF atom order. Use
`--nmr-assignment-mode legacy` to preserve the previous loose assignment behavior.
Use `--allow-extra-peaks` or `--allow-missing-peaks` only when strict coverage
should be reported as partial instead of aborting.
Further documentation for workflow options is available [here](https://ruslankotl.github.io/DP5/)

Original DP5 code can be found at [https://github.com/Goodman-lab/DP5](https://github.com/Goodman-lab/DP5)
