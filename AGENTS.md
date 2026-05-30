# AGENTS.md

* **Project summary:** DP5 is a Python chemistry/NMR structure-assignment workflow. It prepares molecular structure candidates, can run conformer search and DFT backends, predicts or reads NMR shifts, assigns spectra, and produces DP4/DP5 probability reports for candidate structures.

## Documentation Rule (Context7)

When a request depends on third-party library/framework/API documentation, use Context7 before writing code or giving step-by-step instructions.

* Use Context7 to confirm exact APIs, signatures, config flags, setup steps, and version-specific behavior.
* If Context7 does not provide enough detail, say so and ask what version/environment is in use, or fall back to other authoritative sources when appropriate.
* Do not guess doc-sensitive details when Context7 is available.

## Runtime and Tooling

* **Python:** `>=3.9, <=3.11` in `pyproject.toml`; the README notes this is due to the TensorFlow dependency range.
* **Package layout:** flat package layout, import package `dp5/` at repository root. This is not a `src/` layout.
* **Project/package name:** `DP5`.
* **CLI entry point:** `pydp4 = dp5.load_config:main`.
* **Build backend:** `setuptools.build_meta`; `setup.py` only delegates to `setuptools.setup()`.
* **Environment tooling:** README examples use `uv` with editable installs. There is no `uv.lock`, so do not assume `uv sync` is configured for this repo.
* **Docs tooling:** Sphinx is available through the `dev` extra. The only GitHub Actions workflow builds and publishes Sphinx docs.
* **Not configured in this repo:** no verified `pytest`, lint, format, type-check, coverage, tox, nox, or pre-commit configuration is present.

## Dependency Notes

* Core dependencies include `rdkit`, `numpy<2`, `scipy>=1.10`, `scikit-learn`, `pandas`, `statsmodels`, `nmrglue`, `lmfit`, `matplotlib`, `pathos`, `dill`, `joblib`, `networkx>=3`, `tomli`, and `tqdm`.
* TensorFlow is platform-specific in `pyproject.toml`:
  * `tensorflow>=2.13,<=2.14` except on Apple Silicon macOS.
  * `tensorflow-metal>=1.0` on Apple Silicon macOS.
* Treat the TensorFlow, NumPy, scikit-learn, dill/pickle, and model-asset stack as version-sensitive. Do not update those ranges casually.
* Heavy model and data assets live under `dp5/neural_net/`, `dp5/analysis/`, `nmrdb-dataset/`, `reassignments/`, and `stereo_examples/`. Do not delete, rename, or regenerate them unless the task explicitly requires it.
* No GUI extra is configured in this repository.

## Exact Commands

* **Create a compatible uv environment:** `uv venv --python 3.10 .venv`
* **Activate the environment:** `source .venv/bin/activate`
* **Install editable package:** `uv pip install -e .`
* **Install docs dependencies:** `uv pip install -e ".[dev]"`
* **Show CLI help:** `pydp4 --help`
* **Run the README example:** from `reassignments/S11`, run `pydp4 -s S11a_.sdf S11b_.sdf -n S11_NMR -w w`; this writes outputs into the current directory.
* **Create a temporary output folder:** `mkdir -p /tmp/dp5-check`
* **Run with an explicit existing output folder:** `pydp4 -s <SD_FILE> -n <NMR_FILE> -w <workflow_flags> -o /tmp/dp5-check`
* **Build docs after installing `dev`:** `sphinx-build -M html docs/source/ docs/build/`

## Repository Structure

* `dp5/load_config.py`: CLI parser, config loading, command-line override handling, TMS reference lookup, input preparation, output `config.json` writing, and call into the runner.
* `dp5/runner.py`: top-level workflow orchestration for conformer search, DFT, neural-net shifts, NMR assignment, DP4, and DP5.
* `dp5/run/`: input preparation and core `Molecule`/`Molecules` data containers.
* `dp5/conformer_search/`: conformer-search backends and pruning logic. Backends are dynamically imported by `config["method"]` and must expose `ConfSearchMethod`.
* `dp5/dft/`: DFT backend abstraction plus Gaussian, NWChem, and ORCA implementations. Backends are dynamically imported by `config["method"]` and must expose `DFTMethod`.
* `dp5/nmr_processing/`: proton/carbon NMR processing, solvent removal, peak picking, integration, assignment, and `NMRData` public import.
* `dp5/analysis/`: DP4/DP5 probability code plus serialized KDE/PCA/model artifacts used by analysis.
* `dp5/neural_net/`: neural-network shift prediction code, bundled NFP code, Keras/HDF5 models, preprocessors, and related assets.
* `dp5/config/default_config.toml`: canonical default runtime configuration.
* `docs/source/`: Sphinx documentation for usage, config anatomy, API, DFT architecture, NMR processing, and DP5 analysis.
* `reassignments/` and `stereo_examples/`: representative chemistry examples used for manual checks and examples.
* `.github/workflows/sphinx.yaml`: docs-only CI workflow.

## Working Expectations

* Keep changes repository-specific and consistent with the existing CLI-first workflow.
* Preserve the flat `dp5/` package layout and the public `pydp4` entry point unless the user explicitly asks for packaging changes.
* Treat config as dictionary-driven TOML/JSON config, not Pydantic. If adding or renaming config fields, update `dp5/config/default_config.toml`, CLI override logic when relevant, and `docs/source/config_anatomy.rst`.
* Be careful with path semantics: many inputs are resolved relative to the current working directory, `prepare_inputs` writes SDF files into the current working directory, and `load_config.py` writes `config.json` into `output_folder`.
* Preserve the `Molecule`/`Molecules` attribute contracts for atoms, conformers, energies, labels, shifts, populations, and analysis outputs.
* Keep conformer-search integrations behind the `ConfSearchMethod(config)` backend contract.
* Keep DFT integrations behind the `DFTMethod(config)` backend contract and the shared `BaseDFTMethod` responsibilities documented in `docs/source/dft_architecture.rst`.
* Do not hardcode external chemistry executables or cluster settings; keep Gaussian, NWChem, ORCA, MacroModel, Tinker, and related paths configurable.
* Avoid triggering expensive external DFT or conformer-search programs during routine validation unless the task explicitly requires it.
* Preserve the distinction between DFT-derived shifts and neural-network shifts. Current docs/config note DP5 analysis is implemented for carbon NMR.
* Keep generated report/output filenames stable unless the user asks for a format change: `config.json`, `output.dp4`, `output.dp5`, `dp4/data_dic.p`, and `dp5/data_dic.p`.
* Use the existing logging pattern: module-level `logging.getLogger(__name__)` and `setup_logger(...)` from `dp5.logger` for CLI setup.
* Keep Sphinx docs in sync when public behavior, config, usage, backend architecture, or documented APIs change.

## Testing Expectations

* There is no verified automated test command in this snapshot.
* For small import/config changes, run the smallest practical smoke check, for example: `python -c "import dp5; from dp5.load_config import DEFAULT_BASE_CONFIG_PATH; print(DEFAULT_BASE_CONFIG_PATH)"`
* For CLI changes, run `pydp4 --help` after an editable install.
* For workflow changes that do not require external DFT/MM executables, use a representative example under `reassignments/`, such as the README's `S11` example.
* For DFT backend changes, prefer parsing or dry-run checks against existing files/fixtures when possible. Ask before launching real Gaussian, NWChem, ORCA, MacroModel, or cluster jobs.
* For docs changes, run `sphinx-build -M html docs/source/ docs/build/` when Sphinx dependencies are installed.
* If a validation command is skipped because dependencies are missing or the command would launch expensive external software, state that clearly in the final response.

## Style Rules Used Here

* Follow the existing code style in the repo instead of introducing external defaults.
* Use concise module/class/function docstrings for new public code; many existing modules rely on docstrings for Sphinx output.
* Prefer type hints on new or touched public functions where they clarify contracts, but avoid broad annotation churn.
* Keep imports grouped as standard library, third-party, then project imports.
* Prefer small helper functions for parsing, validation, and format conversion steps.
* Keep dynamically imported backend names, class names, and config keys stable.
* Avoid broad formatting-only changes. Existing files are not uniformly formatted, and repo-wide churn makes chemistry workflow changes harder to review.
* Prefer structured data handling over ad hoc string parsing when the local code or a standard library parser already supports the format.

## Documentation Expectations

* Update `readme.md` when install, quick-start, CLI, or high-level usage behavior changes.
* Update `docs/source/usage.rst` when command-line flags, workflow flags, or installation instructions change.
* Update `docs/source/config_anatomy.rst` when config keys, defaults, or meanings change.
* Update `docs/source/dft_architecture.rst` or `docs/source/dft_api.rst` when DFT backend contracts change.
* Update `docs/source/nmr_processing.rst` or `docs/source/nmr_processing_api.rst` when NMR processing APIs or behavior change.
* No changelog file is present; do not invent one without being asked.

## Ask First Rules

Stop and ask before doing any of the following:

* Changing supported Python versions or major dependency ranges.
* Replacing core chemistry, NMR, ML, or DFT libraries.
* Renaming the package, moving to a `src/` layout, or changing public import paths.
* Changing the `pydp4` command name, workflow flag semantics, CLI flag names, or exit behavior.
* Changing default config shape, default config path, or the meaning of existing config keys.
* Changing output filenames, directory layout, report contents, or serialized analysis data formats.
* Modifying DP4/DP5 probability formulas, statistical parameters, KDE/PCA/model loading, or scaling semantics.
* Regenerating, deleting, or replacing bundled neural-net, KDE, PCA, pickle, or dataset artifacts.
* Launching expensive external DFT/conformer-search jobs or cluster submissions.
* Changing NMR parsing, solvent-removal, peak-picking, integration, or assignment behavior in ways that alter scientific results.

## Never Do

* Never add lint/format/type/test tooling that is not already configured unless the user explicitly asks.
* Never copy README prose wholesale into docs or agent instructions.
* Never remove bundled model/data artifacts because they look large or generated.
* Never make the GUI a requirement; this repo has no GUI extra or GUI entry point.
* Never silently mutate config schema/defaults without updating docs and CLI handling where relevant.
* Never hardcode local machine paths, external executable paths, cluster usernames, queues, or scratch directories.
* Never make noisy repo-wide style churn unrelated to the requested change.
* Never assume DFT, MacroModel, Tinker, ORCA, Gaussian, or NWChem are installed locally.

## Completion Checklist

* [ ] Confirm any commands used are supported by this repository snapshot.
* [ ] Keep changes focused and minimal.
* [ ] Preserve CLI, config, backend, and output contracts unless the task explicitly changes them.
* [ ] Update README/docs/default config when public behavior or config changes.
* [ ] Avoid expensive external chemistry jobs unless explicitly requested.
* [ ] Run the smallest practical validation, or explain why it was not run.