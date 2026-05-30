"""Runs DP5 analysis for organic molecules."""

from pathlib import Path
import logging
import dill as pickle
from abc import abstractmethod

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity

from dp5.neural_net.CNN_model import *
from dp5.analysis.utils import scale_nmr, AnalysisData

logger = logging.getLogger(__name__)


class DP5:
    """Performs DP5 analysis"""

    def __init__(self, output_folder: Path, use_dft_shifts: bool):
        """Initialise the settings.

        Arguments:
          output_folder(Path): path for saved DP5 data
          use_dft_shifts(bool): if set, analyses errors of DFT calculations, compares shifts againt their environments otherwise.

        """
        logger.info("Setting up DP5 method")
        self.output_folder = output_folder
        self.dft_shifts = use_dft_shifts

        if use_dft_shifts:
            # must load model for error prediction
            self.C_DP5 = ErrorDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Error_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_ERRORrep_Error_decomp.p",
                kde_file="pca_10_kde_ERRORrep_Error_kernel.p",
                dp5_correct_scaling="Error_correct_kde.p",
                dp5_incorrect_scaling="Error_incorrect_kde.p",
            )
        else:
            # must load model for shift preiction
            self.C_DP5 = QuantileDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb_CASCADE_99quantiles.zip",
                batch_size=16,
            )

        if not self.output_folder.exists():
            self.output_folder.mkdir()

        if not (self.output_folder / "dp5").exists():
            (self.output_folder / "dp5").mkdir()

    def __call__(self, mols):
        """Runs DP5 calculations.

        Arguments:
          mols: Molecule objects
        """
        data_dic_path = self.output_folder / "dp5" / "data_dic.p"
        expected_signature = _dp5_cache_signature(mols, self.dft_shifts)
        strict_run = any(_is_strict_assignment(mol) for mol in mols)
        dp5_data = DP5Data(mols, data_dic_path)
        cache_valid = False
        if dp5_data.exists:
            logger.info("Found existing DP5 probability file")
            dp5_data.load()
            cache_valid = getattr(dp5_data, "cache_signature", None) == expected_signature
            if not cache_valid:
                logger.info("Existing DP5 probability file does not match assignments")

        if not cache_valid:
            dp5_data.mols = [mol.input_file for mol in mols]
            logger.info("Calculating DP5 probabilites...")
            (
                dp5_data.Clabels,
                dp5_data.Cshifts,
                dp5_data.Cexp,
                dp5_data.Cerrors,
                dp5_data.Cconf_atom_probs,
                dp5_data.CDP5_atom_probs,
                dp5_data.CDP5_mol_probs,
            ) = self.C_DP5(mols)
            if strict_run:
                rescaling_skips = getattr(
                    self.C_DP5,
                    "secondary_rescaling_skipped_for_grouping",
                    [False] * len(mols),
                )
                rescaling_skips = _normalise_rescaling_skips(
                    rescaling_skips, len(mols)
                )
                dp5_data.cache_signature = expected_signature
                dp5_data.Cassignment_coverage = [
                    _coverage_for_dp5(mol, self.dft_shifts, skip)
                    for mol, skip in zip(mols, rescaling_skips)
                ]
                dp5_data.Cassignment_mode = [
                    coverage["mode"] for coverage in dp5_data.Cassignment_coverage
                ]
                dp5_data.Ccoverage_status = [
                    coverage["status"] for coverage in dp5_data.Cassignment_coverage
                ]
                dp5_data.Cgroup_members = [
                    getattr(mol, "C_group_members", [[label] for label in mol.C_labels])
                    for mol in mols
                ]
                dp5_data.Cgroup_atom_indices = [
                    getattr(
                        mol,
                        "C_group_atom_indices",
                        [[int(label[1:]) - 1] for label in mol.C_labels],
                    )
                    for mol in mols
                ]
                dp5_data.Cgroup_labels = dp5_data.Clabels
                dp5_data.Cgroup_calculated_shifts = dp5_data.Cshifts
                dp5_data.Cgroup_experimental_shifts = dp5_data.Cexp
                dp5_data.Cgroup_errors = dp5_data.Cerrors
                dp5_data.Cgroup_probabilities = dp5_data.CDP5_atom_probs
            else:
                _clear_strict_dp5_metadata(dp5_data)
            dp5_data.save()
        return dp5_data.output


class DP5ProbabilityCalculator:
    def __init__(
        self,
        atom_type,
    ):
        """Initialises DP5 Probability calculator for one atom.

        Arguments:
          atom_type (str): atom symbol. Can be 'C' or 'H'
          model_file (str): path for representation generating model to load.
          batch_size (int): batch size for the model.
          transform_file (str): Path to the Scikit-learn PCA file relative to ``dp5/analysis`` folder. Reduces dimensionality of the representation.
          kde_file (str): Path to :py:obj:`scipy.stats.gaussian_kde` or :py:obj:`sklearn.neighbors.KernelDensity` object. Estimates DP5 probabilities
          dp5_correct_scaling (str). Path to :py:obj:`scipy.stats.gaussian_kde` or :py:obj:`sklearn.neighbors.KernelDensity` object. Estimates :math:`P(correct|structure)` for rescaling. Default is None (no scaling)
          dp5_incorrect_scaling (str). Path to :py:obj:`scipy.stats.gaussian_kde` or :py:obj:`sklearn.neighbors.KernelDensity` object. Estimates :math:`P(incorrect|structure)` for rescaling. Default is None (no scaling)

        """
        self.atom_type = atom_type

    @abstractmethod
    def rescale_probabilities(
        self,
        mol_probs,
        errors,
        error_threshold=2,
        grouped=False,
        skip_secondary_rescaling=None,
    ):
        """
        Scales and aggregated atomic probabilities.

        Computes geometric means of atomic probabilities to generate final molecular probabilities.
        """
        total_probs = np.array([np.exp(np.log(arr + 1e-6).mean()) for arr in mol_probs])
        return mol_probs, total_probs

    @abstractmethod
    def probfunction(self, df):
        return NotImplementedError("KDE sampling function not implemented")

    @staticmethod
    def boltzmann_weight(df, col):
        return df.groupby("mol_id")[["conf_population", col]].apply(
            lambda x: (x[col] * x["conf_population"]).sum()
        )

    def __call__(self, mols):
        """Carries out DP5 analysis.

        Arguments:
          mols(list of :py:class:`~dp5.run.data_structures.Molecule`): :py:class:`dp5.run.data_structures.Molecule` objects used in the calculation. Must contain shifts and labels for the provided atom.

        Returns:
          A tuple containing lists of labels of atoms used in the analysis,
          their calculated shifts, their experimental shifts,
          scaled errors, DP5 probabilities for each atom in each conformer,
          Boltzmann-weighted atom DP5 probabilites, and total molecular DP5 probabilities.
        """
        all_labels = []
        rep_df = []
        grouped_mode = False
        secondary_rescaling_skips = []
        for mol_id, mol in enumerate(mols):
            (
                calculated,
                experimental,
                labels,
                atom_indices,
                group_atom_indices,
            ) = self.get_shifts_and_labels(mol)
            # drop unassigned !
            has_exp = np.isfinite(experimental)
            new_calcs = calculated[:, has_exp]
            new_exps = experimental[has_exp]
            new_labs = labels[has_exp]
            if len(new_exps) == 0:
                raise ValueError("No assigned experimental carbon shifts available for DP5")

            if group_atom_indices is None:
                new_inds = atom_indices[has_exp]
                group_positions = None
            else:
                grouped_mode = True
                selected_groups = [
                    group_atom_indices[i] for i in np.where(has_exp)[0].tolist()
                ]
                new_inds, group_positions = _flatten_group_atom_indices(selected_groups)
                secondary_rescaling_skips.append(
                    _skip_secondary_rescaling_for_grouping(mol, selected_groups)
                )
            if group_atom_indices is None:
                secondary_rescaling_skips.append(False)

            # generate scaled errors
            scaled = scale_nmr(new_calcs, new_exps)
            corrected_errors = scaled - new_exps[np.newaxis, :]

            all_labels.append(new_labs)

            rep_df.append(
                (
                    mol_id,
                    range(mol.conformers.shape[0]),
                    mol.rdkit_mols,
                    new_inds,
                    mol.populations,
                    new_calcs,
                    new_exps,
                    corrected_errors,
                    group_positions,
                )
            )

        rep_df = pd.DataFrame(
            rep_df,
            columns=[
                "mol_id",
                "conf_id",
                "Mol",
                "atom_index",
                "conf_population",
                "conf_shifts",
                "exp_shifts",
                "errors",
                "group_positions",
            ],
        )
        # each row of dataframe represents a geometry
        rep_df = rep_df.explode(
            ["conf_id", "Mol", "conf_shifts", "conf_population", "errors"],
            ignore_index=True,
        )
        logger.info("Extracting atomic representations")
        # now return condensed representations! These are now grouped by conformer
        # need to redo to accommodate

        atom_probs = self.probfunction(rep_df)
        # should be abstracted into KDE-based calculator

        weighted_probs = self.boltzmann_weight(rep_df, "atom_probs")
        weighted_probs = 1 - weighted_probs

        weighted_errors = self.boltzmann_weight(rep_df, "errors")
        cmae = weighted_errors.apply(lambda x: np.mean(np.abs(x)))

        # rescale and aggregate probabilities
        weighted_probs, total_probs = self.rescale_probabilities(
            weighted_probs,
            cmae,
            grouped=grouped_mode,
            skip_secondary_rescaling=secondary_rescaling_skips,
        )
        self.secondary_rescaling_skipped_for_grouping = secondary_rescaling_skips

        calc_shifts_analysed = self.boltzmann_weight(rep_df, "conf_shifts")
        exp_shifts_analysed = rep_df.groupby("mol_id")["exp_shifts"].first()

        # eventually return atomic probs, weighted atomic probs, DP5 scores
        logger.info("Atomic probabilities estimated")
        return (
            all_labels,
            calc_shifts_analysed,
            exp_shifts_analysed,
            weighted_errors,
            atom_probs,
            weighted_probs,
            total_probs,
        )

    def get_shifts_and_labels(self, mol):
        """
        Returns calculated and experimental shifts for nuclei in the molecule.

        Arguments:
          self.atom_type(str): nuclei being analysed
          mols(:py:class:`~dp5.run.data_structures.Molecule`): :py:class:`dp5.run.data_structures.Molecule`. Must contain shifts and labels for the provided atom.

        Returns:
          calculated conformer shifts
          assigned experimental shifts
          0-based indices of relevant atoms
        """
        at = self.atom_type
        if (
            at == "C"
            and getattr(mol, "C_assignment_mode", "legacy") == "strict"
            and hasattr(mol, "conformer_C_group_pred")
        ):
            conformer_shifts = mol.conformer_C_group_pred
            assigned_shifts = mol.C_group_exp
            atom_labels = mol.C_group_labels
            group_atom_indices = [
                np.array(indices, dtype=int) for indices in mol.C_group_atom_indices
            ]
            atom_indices = np.array([], dtype=int)
        else:
            conformer_shifts = getattr(mol, "conformer_%s_pred" % at)
            assigned_shifts = getattr(mol, "%s_exp" % at)
            atom_labels = getattr(mol, "%s_labels" % at)
            atom_indices = np.array([int(label[len(at) :]) - 1 for label in atom_labels])
            group_atom_indices = None

        return conformer_shifts, assigned_shifts, atom_labels, atom_indices, group_atom_indices


class ErrorDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(
        self,
        atom_type,
        model_file,
        batch_size,
        transform_file,
        kde_file,
        dp5_correct_scaling=None,
        dp5_incorrect_scaling=None,
    ):
        super().__init__(atom_type)
        self.model = build_model(model_file=model_file)
        self.batch_size = batch_size
        self.transform = _load_pickle(transform_file)
        self.kde = KernelDensityEstimator(kde_file)
        if dp5_correct_scaling is not None:
            self.dp5_correct_kde = KernelDensityEstimator(dp5_correct_scaling)
        if dp5_incorrect_scaling is not None:
            self.dp5_incorrect_kde = KernelDensityEstimator(dp5_incorrect_scaling)

    def probfunction(self, rep_df):
        logger.debug("Transforming representations")
        rep_df["representations"] = extract_representations(
            self.model, rep_df, self.batch_size
        )
        rep_df["representations"] = _aggregate_group_values(
            rep_df, "representations"
        )
        rep_df["representations"] = rep_df["representations"].apply(
            self.transform.transform
        )
        # should be abstracted into KDE-based calculator
        logger.info("Estimating atomic probabilities")
        rep_df["atom_probs"] = self.kde_probfunction(rep_df)
        atom_probs = [np.stack(df) for i, df in rep_df.groupby("mol_id")["atom_probs"]]
        return atom_probs

    def kde_probfunction(self, df):
        # loop through atoms in the test molecule - generate kde for all of them.
        # implement joblib parallel search
        # check if this has been calculated

        min_value = -20
        max_value = 20
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        probs = []
        with Parallel(prefer="threads", n_jobs=-1) as pool:
            for i, (rep, errors) in tqdm(
                df[["representations", "errors"]].iterrows(),
                total=len(df),
                desc="Computing error KDEs",
                leave=True,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool(delayed(self.kde)(atom) for atom in point[:])

                conf_probs = []
                for pdf, error in zip(results, errors):
                    integral = 0
                    if pdf.sum() != 0:
                        max_x = x[np.argmax(pdf)]

                        low_point = max(min_value, max_x - abs(max_x - error))
                        high_point = min(max_value, max_x + abs(max_x - error))

                        low_bound = np.argmin(np.abs(x - low_point))
                        high_bound = np.argmin(np.abs(x - high_point))

                        bound_integral = np.sum(
                            pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
                        )
                        integral = bound_integral / pdf.sum()
                    conf_probs.append(integral)
                probs.append(np.array(conf_probs))

        return probs

    def rescale_probabilities(
        self,
        mol_probs,
        errors,
        error_threshold=2,
        grouped=False,
        skip_secondary_rescaling=None,
    ):
        _, total_probs = super().rescale_probabilities(
            mol_probs,
            errors,
            grouped=grouped,
            skip_secondary_rescaling=skip_secondary_rescaling,
        )
        if skip_secondary_rescaling is None:
            skip_secondary_rescaling = [grouped] * len(mol_probs)
        skip_secondary_rescaling = _normalise_rescaling_skips(
            skip_secondary_rescaling, len(mol_probs)
        )
        scaled_probs = []
        scaled_total = []
        for prob, error, total, skip in zip(
            mol_probs, errors, total_probs, skip_secondary_rescaling
        ):
            if skip:
                scaled_probs.append(prob)
                scaled_total.append(total)
                continue
            if error < error_threshold:
                vector = np.concatenate((prob, np.atleast_1d(total)))
                correct = self.dp5_correct_kde(vector)
                incorrect = self.dp5_incorrect_kde(vector)
                scaled = correct / (correct + incorrect)
                scaled_probs.append(scaled[:-1])
                scaled_total.append(scaled[-1])
            else:
                scaled_probs.append(prob)
                scaled_total.append(total)
        return scaled_probs, scaled_total


class ExpDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(
        self,
        atom_type,
        model_file,
        batch_size,
        transform_file,
        kde_file,
        dp5_correct_scaling=None,
        dp5_incorrect_scaling=None,
    ):
        super().__init__(atom_type)
        self.model = build_model(model_file=model_file)
        self.batch_size = batch_size
        self.transform = _load_pickle(transform_file)
        self.kde = KernelDensityEstimator(kde_file)
        if dp5_correct_scaling is not None:
            self.dp5_correct_kde = KernelDensityEstimator(dp5_correct_scaling)
        if dp5_incorrect_scaling is not None:
            self.dp5_incorrect_kde = KernelDensityEstimator(dp5_incorrect_scaling)

    def probfunction(self, rep_df):
        logger.debug("Transforming representations")
        rep_df["representations"] = extract_representations(
            self.model, rep_df, self.batch_size
        )
        rep_df["representations"] = _aggregate_group_values(
            rep_df, "representations"
        )
        rep_df["representations"] = rep_df["representations"].apply(
            self.transform.transform
        )
        # should be abstracted into KDE-based calculator
        logger.info("Estimating atomic probabilities")
        rep_df["atom_probs"] = self.kde_probfunction(rep_df)
        atom_probs = [np.stack(df) for i, df in rep_df.groupby("mol_id")["atom_probs"]]
        return atom_probs

    def kde_probfunction(self, df):
        """Since the result is compared to the experimental shifts, weights the representations and runs KDE on those."""
        # loop through atoms in the test molecule - generate kde for all of them.
        total_reps = self.boltzmann_weight(df, "representations")
        exp_data = df.groupby("mol_id")["exp_shifts"].first()
        mol_df = pd.DataFrame({"representations": total_reps, "exp_shifts": exp_data})

        min_value = 0
        max_value = 250
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        mol_probs = []
        with Parallel(prefer="threads", n_jobs=-1) as pool:
            for i, (rep, exp) in tqdm(
                mol_df[["representations", "exp_shifts"]].iterrows(),
                total=len(mol_df),
                desc="Computing experimental KDEs",
                leave=True,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool(delayed(self.kde)(atom) for atom in point[:])

                conf_probs = []
                for pdf, value in zip(results, exp):
                    integral = 0
                    if pdf.sum() != 0:
                        max_x = x[np.argmax(pdf)]

                        low_point = max(min_value, max_x - abs(max_x - value))
                        high_point = min(max_value, max_x + abs(max_x - value))

                        low_bound = np.argmin(np.abs(x - low_point))
                        high_bound = np.argmin(np.abs(x - high_point))

                        bound_integral = np.sum(
                            pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
                        )
                        integral = bound_integral / pdf.sum()
                    conf_probs.append(integral)
                mol_probs.append(np.array(conf_probs))
        consistency_hack = {i: probs for i, probs in enumerate(mol_probs)}
        consistent_probs = df["mol_id"].map(consistency_hack)
        return consistent_probs

    def rescale_probabilities(self, *args, **kwargs):
        return super().rescale_probabilities(*args, **kwargs)


class QuantileDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(
        self, atom_type, model_file, batch_size, quantile_regressor="quantile99.zip"
    ):
        super().__init__(atom_type)
        default_path = str(Path(__file__).parent.parent / "neural_net" / model_file)
        self.model = CASCADE_Quantile.load(default_path)
        self.batch_size = batch_size

    def probfunction(self, df):
        # take representations
        df["quantiles"] = extract_representations(self.model, df, self.batch_size)
        df["quantiles"] = _aggregate_group_values(df, "quantiles")
        df[["mu", "sigma"]] = self.generate_distributions(df["quantiles"])
        atom_probs_all = []
        for i, (mus, sigmas, exps) in df[["mu", "sigma", "exp_shifts"]].iterrows():
            atom_probs = []
            for mu, sigma, peak in zip(mus, sigmas, exps):
                perc = norm.cdf(peak, mu, sigma)
                prob = np.abs(1 - 2 * perc)
                atom_probs.append(prob)
            atom_probs = np.array(atom_probs)
            atom_probs_all.append(atom_probs)
        df["atom_probs"] = atom_probs_all
        atom_probs = [np.stack(df) for i, df in df.groupby("mol_id")["atom_probs"]]
        return atom_probs

    def generate_distributions(self, quantile_col):
        # in principle, should be able to explode then reassemble
        # makes for easier handling
        # that or compute lengths
        atom_nums = quantile_col.apply(len).values
        slice_inds = np.cumsum(atom_nums)[:-1]
        quantiles = np.concatenate(quantile_col.values)

        # fit distributions, then spit out mean and std
        mus = []
        sigmas = []
        for q in quantiles:
            percentiles = self.model.quantiles
            dims = len(percentiles)
            median = q[dims // 2]
            std = (q[dims * 2 // 3] - q[dims // 3]) / 2

            mu, sigma = curve_fit(norm.cdf, q, percentiles, p0=[median, std])[0]
            mus.append(mu)
            sigmas.append(sigma)

        mus = np.array(mus)
        sigmas = np.array(sigmas)
        result_df = pd.DataFrame(
            {
                "mu": np.split(mus, slice_inds),
                "sigma": np.split(sigmas, slice_inds),
            },
            index=quantile_col.index,
        )

        return result_df


class KernelDensityEstimator:
    def __init__(self, path_to_pickle):
        self.estimator = _load_pickle(path_to_pickle)
        if type(self.estimator) is gaussian_kde:
            self.evaluate = self._scipy_estimator
        elif type(self.estimator) is KernelDensity:
            self.evaluate = self._sklearn_estimator

    def __call__(self, data):
        return self.evaluate(data)

    def _scipy_estimator(self, data):
        return self.estimator(data)

    def _sklearn_estimator(self, data):
        return self.estimator.score_samples(data.T)


class DP5Data(AnalysisData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def output(self):
        """Uncomment when H-DP5 is implemented"""
        output_dict = dict()
        output_dict["C_output"] = []
        # output_dict["H_output"] = []
        output_dict["CDP5_output"] = []
        # output_dict["HDP5_output"] = []
        # output_dict["DP5_output"] = []
        coverages = getattr(
            self,
            "Cassignment_coverage",
            [_legacy_output_coverage()] * len(self.mols),
        )
        for mol, clab, cshift, cexp, cerr, cpr, coverage in zip(
            self.mols,
            self.Clabels,
            self.Cshifts,
            self.Cexp,
            self.Cerrors,
            self.CDP5_atom_probs,
            coverages,
        ):
            output = ""
            if coverage.get("mode") == "strict":
                output += f"\nCarbon assignment coverage for {mol}:"
                output += self.print_coverage(coverage)
            output += f"\nAssigned C NMR shift for {mol}:"
            output += self.print_assignment(
                clab,
                cshift,
                cexp,
                cerr,
                cpr,
                grouped=coverage.get("mode") == "strict",
            )
            output_dict["C_output"].append(output)

        # for mol, hlab, hshift, hscal, hexp, herr in zip(
        #    self.mols, self.Hlabels, self.Hshifts, self.Hscaled, self.Hexp, self.Herrors
        # ):
        #    output = f"\nAssigned H NMR shift for {mol}:"
        #    output += self.print_assignment(hlab, hshift, hscal, hexp, herr)
        #    output_dict["H_output"].append(output)

        for mol, cdp5 in zip(self.mols, self.CDP5_mol_probs):
            output_dict["CDP5_output"].append(
                f"Carbon DP5 probability for {mol}: {cdp5}"
            )
        t_dic = [
            dict(zip(output_dict.keys(), values))
            for values in zip(*output_dict.values())
        ]
        dp5_output = "\n\n".join([mol["C_output"] for mol in t_dic])
        dp5_output += "\n\n"
        dp5_output += "\n".join([mol["CDP5_output"] for mol in t_dic])
        return dp5_output

    @staticmethod
    def print_coverage(coverage):
        output = f"\nassignment mode: {coverage['mode']}"
        output += f"\nrequired group count: {coverage['required_group_count']}"
        output += (
            "\nexperimental carbon signal count: "
            f"{coverage['experimental_carbon_signal_count']}"
        )
        output += "\nomitted atoms: " + _format_list(coverage["omitted_atoms"])
        output += "\nextra experimental shifts: " + _format_list(
            coverage["extra_experimental_shifts"]
        )
        output += "\nmissing groups: " + _format_list(coverage["missing_groups"])
        output += f"\ncoverage status: {coverage['status']}"
        output += f"\ngroup rescaling applied: {coverage.get('rescaling_applied', True)}"
        output += "\nper-group assignment:"
        for assignment in coverage["assignments"]:
            value = assignment["experimental_shift"]
            if value is not None and np.isfinite(value):
                value_text = f"{value:.2f}"
            else:
                value_text = "missing"
            output += f"\n{assignment['group']} -> {value_text}"
        return output

    @staticmethod
    def print_assignment(labels, calculated, exp, error, probs, grouped=False):
        """Prints table for molecule"""

        s = np.argsort(calculated)
        svalues = calculated[s]
        slabels = labels[s]
        sexp = exp[s]
        serror = error[s]
        sprob = probs[s]

        label_name = "group" if grouped else "label"
        output = f"\n{label_name}, calc, exp, error, prob"

        for lab, calc, ex, er, p in zip(slabels, svalues, sexp, serror, sprob):
            output += f"\n{lab}, {calc:.2f}, {ex:.2f}, {er:.2f}, {p:.2f}"
        return output


def _flatten_group_atom_indices(group_atom_indices):
    flat_indices = []
    group_positions = []
    cursor = 0
    for indices in group_atom_indices:
        indices = list(indices)
        positions = list(range(cursor, cursor + len(indices)))
        group_positions.append(positions)
        flat_indices.extend(indices)
        cursor += len(indices)
    return np.array(flat_indices, dtype=int), group_positions


def _aggregate_group_values(df, column):
    def aggregate(row):
        positions = row.get("group_positions")
        values = np.array(row[column])
        if positions is None:
            return values
        return np.array([values[position].mean(axis=0) for position in positions])

    return df.apply(aggregate, axis=1)


def _is_strict_assignment(mol):
    return getattr(mol, "C_assignment_mode", "legacy") == "strict"


def _skip_secondary_rescaling_for_grouping(mol, selected_group_atom_indices):
    if not _is_strict_assignment(mol):
        return False
    scored_group_count = len(selected_group_atom_indices)
    scored_atom_count = sum(len(indices) for indices in selected_group_atom_indices)
    original_atom_count = len(getattr(mol, "C_labels", []))
    return (
        scored_group_count != scored_atom_count
        or scored_atom_count != original_atom_count
    )


def _clear_strict_dp5_metadata(dp5_data):
    for attr in (
        "cache_signature",
        "Cassignment_coverage",
        "Cassignment_mode",
        "Ccoverage_status",
        "Cgroup_members",
        "Cgroup_atom_indices",
        "Cgroup_labels",
        "Cgroup_calculated_shifts",
        "Cgroup_experimental_shifts",
        "Cgroup_errors",
        "Cgroup_probabilities",
    ):
        if hasattr(dp5_data, attr):
            delattr(dp5_data, attr)


def _normalise_rescaling_skips(rescaling_skips, mol_count):
    if rescaling_skips is None:
        rescaling_skips = []
    rescaling_skips = list(rescaling_skips)
    if len(rescaling_skips) < mol_count:
        rescaling_skips.extend([False] * (mol_count - len(rescaling_skips)))
    return rescaling_skips[:mol_count]


def _dp5_cache_signature(mols, use_dft_shifts):
    molecules = []
    for mol in mols:
        if getattr(mol, "C_assignment_mode", "legacy") == "strict":
            labels = getattr(mol, "C_group_labels", [])
            exp = getattr(mol, "C_group_exp", [])
            members = getattr(mol, "C_group_members", [])
        else:
            labels = getattr(mol, "C_labels", [])
            exp = getattr(mol, "C_exp", [])
            members = [[label] for label in labels]
        molecules.append(
            {
                "input_file": mol.input_file,
                "mode": getattr(mol, "C_assignment_mode", "legacy"),
                "labels": [str(label) for label in labels],
                "members": [[str(label) for label in group] for group in members],
                "exp": _finite_list(exp),
            }
        )
    return {
        "strict_assignment_cache_version": 2,
        "use_dft_shifts": use_dft_shifts,
        "molecules": molecules,
    }


def _legacy_coverage(mol):
    assignments = []
    labels = getattr(mol, "C_labels", [])
    exp = getattr(mol, "C_exp", [])
    for label, value in zip(labels, exp):
        assignments.append(
            {
                "group": str(label),
                "members": [str(label)],
                "experimental_shift": _finite_value(value),
                "status": "assigned" if _finite_value(value) is not None else "missing",
            }
        )
    finite_exp = [_finite_value(value) for value in exp]
    return {
        "mode": "legacy",
        "required_group_count": len(labels),
        "experimental_carbon_signal_count": sum(
            value is not None for value in finite_exp
        ),
        "omitted_atoms": [],
        "extra_experimental_shifts": [],
        "missing_groups": [
            str(label)
            for label, value in zip(labels, finite_exp)
            if value is None
        ],
        "assignments": assignments,
        "status": "legacy",
        "rescaling_applied": True,
    }


def _coverage_for_dp5(mol, use_dft_shifts, secondary_rescaling_skipped=False):
    coverage = getattr(mol, "C_assignment_coverage", _legacy_coverage(mol)).copy()
    if coverage.get("mode") == "strict" and use_dft_shifts:
        coverage["secondary_rescaling_skipped_for_grouping"] = bool(
            secondary_rescaling_skipped
        )
        coverage["rescaling_applied"] = not secondary_rescaling_skipped
    return coverage


def _legacy_output_coverage():
    return {
        "mode": "legacy",
        "required_group_count": 0,
        "experimental_carbon_signal_count": 0,
        "omitted_atoms": [],
        "extra_experimental_shifts": [],
        "missing_groups": [],
        "assignments": [],
        "status": "legacy",
        "rescaling_applied": True,
        "secondary_rescaling_skipped_for_grouping": False,
    }


def _format_list(values):
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _finite_list(values):
    return [_finite_value(value) for value in values]


def _finite_value(value):
    if value is None:
        return None
    value = float(value)
    if np.isfinite(value):
        return value
    return None


def _load_pickle(path: str):
    """
    Loads a pickled object from relative or absolute path.
    Searches within this folder first, then within current folder, then by absolute path.

    Arguments
      path(str): path to the pickled file

    Returns
      Loaded object
    """
    _abs_path = Path(path).resolve()
    _default_path = Path(__file__).parent / path

    if _default_path.exists():
        _path = _default_path
    elif _abs_path.exists():
        _path = _abs_path
    else:
        raise FileNotFoundError("No files found at %s" % (path))
    with open(_path, "rb") as f:
        return pickle.load(f)
