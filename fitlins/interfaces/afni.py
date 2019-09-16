# coding: utf-8
from pathlib import Path
import os
import os.path as op
import numpy as np
import pandas as pd
import subprocess as sp
import io
from warnings import warn


from nipype.interfaces.base import isdefined
import nibabel as nb

from .nistats import FirstLevelModel, prepare_contrasts, _flatten
from .utils import MergeAll

STAT_CODES = nb.volumeutils.Recoder(
    (
        (0, "none", "none"),
        (1, "none", "none"),
        (2, "correlation", "Correl"),
        (3, "t test", "Ttest"),
        (4, "f test", "Ftest"),
        (5, "z score", "Zscore"),
        (6, "chi2", "Chisq"),
        (7, "beta", "Beta"),
        (8, "binomial", "Binom"),
        (9, "gamma", "Gamma"),
        (10, "poisson", "Poisson"),
        (11, "normal", "Normal"),
        (12, "non central f test", "Ftest_nonc"),
        (13, "non central chi2", "Chisq_nonc"),
        (14, "logistic", "Logistic"),
        (15, "laplace", "Laplace"),
        (16, "uniform", "Uniform"),
        (17, "non central t test", "Ttest_nonc"),
        (18, "weibull", "Weibull"),
        (19, "chi", "Chi"),
        (20, "inverse gaussian", "Invgauss"),
        (21, "extreme value 1", "Extval"),
        (22, "p value", "Pval"),
        (23, "log p value", "LogPval"),
        (24, "log10 p value", "Log10Pval"),
    ),
    fields=("code", "label", "stat_code"),
)


class FirstLevelModel(FirstLevelModel):
    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import design_matrix as dm
        from nistats import first_level_model as level1

        info = self.inputs.session_info
        img = nb.load(self.inputs.bold_file)
        vols = img.shape[3]

        if info["sparse"] not in (None, "None"):
            sparse = pd.read_hdf(info["sparse"], key="sparse").rename(
                columns={"condition": "trial_type", "amplitude": "modulation"}
            )
            sparse = sparse.dropna(subset=["modulation"])  # Drop NAs
        else:
            sparse = None

        if info["dense"] not in (None, "None"):
            dense = pd.read_hdf(info["dense"], key="dense")
            column_names = dense.columns.tolist()
            drift_model = None if "cosine_00" in column_names else "cosine"
        else:
            dense = None
            column_names = None
            drift_model = "cosine"

        mat = dm.make_first_level_design_matrix(
            frame_times=np.arange(vols) * info["repetition_time"],
            events=sparse,
            add_regs=dense,
            add_reg_names=column_names,
            drift_model=drift_model,
        )

        mat.to_csv("design.tsv", sep="\t")
        self._results["design_matrix"] = op.join(runtime.cwd, "design.tsv")

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None

        # Create and fit the model, and compute contrasts and tests
        nistats_flm = level1.FirstLevelModel(
            mask_img=mask_file, smoothing_fwhm=smoothing_fwhm
        )

        contrasts = prepare_contrasts(self.inputs.contrast_info, mat.columns.tolist())

        self.reml_fit(runtime, nistats_flm, img, contrasts, design_matrices=mat)

        return runtime

    def reml_fit(
        self,
        run_imgs,
        runtime,
        nistats_flm,
        contrasts,
        events=None,
        confounds=None,
        design_matrices=None,
    ):
        """Fit the GLM
        For each run:
        1. create design matrix X
        2. fit regression using AFNI's 3dREMLfit


        Parameters
        ----------
        run_imgs : Niimg-like object or list of Niimg-like objects,
            See http://nilearn.github.io/manipulating_images/input_output.html#inputing-data-file-names-or-image-objects
            Data on which the GLM will be fitted. If this is a list,
            the affine is considered the same for all.

        runtime : nipype runtime object

        nistats_flm : nistats.first_level_model.First_Level_Model

        contrasts : Object returned by nistats.utils.contrasts.prepare_contrasts
        events : pandas Dataframe or string or list of pandas DataFrames or
                   strings

            fMRI events used to build design matrices. One events object
            expected per run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        confounds : pandas Dataframe or string or list of pandas DataFrames or
                   strings

            Each column in a DataFrame corresponds to a confound variable
            to be included in the regression model of the respective run_img.
            The number of rows must match the number of volumes in the
            respective run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        design_matrices : pandas DataFrame or list of pandas DataFrames,
            Design matrices that will be used to fit the GLM. If given it
            takes precedence over events and confounds.

        """
        import sys
        import time
        from nilearn._utils.niimg_conversions import check_niimg

        from nistats.design_matrix import make_first_level_design_matrix
        from nistats.utils import (
            _check_run_tables,
            _check_events_file_uses_tab_separators,
        )

        # Check arguments
        # Check imgs type
        if events is not None:
            _check_events_file_uses_tab_separators(events_files=events)
        if not isinstance(run_imgs, (list, tuple)):
            run_imgs = [run_imgs]
        if design_matrices is None:
            if events is None:
                raise ValueError("events or design matrices must be provided")
            if nistats_flm.t_r is None:
                raise ValueError(
                    "t_r not given to FirstLevelModel object"
                    " to compute design from events"
                )
        else:
            design_matrices = _check_run_tables(
                run_imgs, design_matrices, "design_matrices"
            )
        # Check that number of events and confound files match number of runs
        # Also check that events and confound files can be loaded as DataFrame
        if events is not None:
            events = _check_run_tables(run_imgs, events, "events")
        if confounds is not None:
            confounds = _check_run_tables(run_imgs, confounds, "confounds")

        # For each run fit the model and keep only the regression results.
        nistats_flm.labels_, nistats_flm.results_, self.design_matrices_ = ([], [], [])
        n_runs = len(run_imgs)
        t0 = time.time()
        for run_idx, run_img in enumerate(run_imgs):
            # Report progress
            if nistats_flm.verbose > 0:
                percent = float(run_idx) / n_runs
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                if run_idx == 0:
                    remaining = "go take a coffee, a big one"
                else:
                    remaining = (100.0 - percent) / max(0.01, percent) * dt
                    remaining = "%i seconds remaining" % remaining

                sys.stderr.write(
                    "Computing run %d out of %d runs (%s)\n"
                    % (run_idx + 1, n_runs, remaining)
                )

            # Build the experimental design for the glm
            run_img = check_niimg(run_img, ensure_ndim=4)
            if design_matrices is None:
                n_scans = run_img.get_data().shape[3]
                if confounds is not None:
                    confounds_matrix = confounds[run_idx].values
                    if confounds_matrix.shape[0] != n_scans:
                        raise ValueError(
                            "Rows in confounds does not match"
                            "n_scans in run_img at index %d" % (run_idx,)
                        )
                    confounds_names = confounds[run_idx].columns.tolist()
                else:
                    confounds_matrix = None
                    confounds_names = None
                start_time = nistats_flm.slice_time_ref * nistats_flm.t_r
                end_time = (n_scans - 1 + nistats_flm.slice_time_ref) * nistats_flm.t_r
                frame_times = np.linspace(start_time, end_time, n_scans)
                design = make_first_level_design_matrix(
                    frame_times,
                    events[run_idx],
                    nistats_flm.hrf_model,
                    nistats_flm.drift_model,
                    nistats_flm.high_pass,
                    nistats_flm.drift_order,
                    nistats_flm.fir_delays,
                    confounds_matrix,
                    confounds_names,
                    nistats_flm.min_onset,
                )
            else:
                design = design_matrices[run_idx]
            self.design_matrices_.append(design)

            img_mat = run_img.get_data()
            if nistats_flm.signal_scaling:
                mean = img_mat.mean(axis=nistats_flm.scaling_axis)
                if (mean == 0).any():
                    warn(
                        "Mean values of 0 observed."
                        "The data have probably been centered."
                        "Scaling might not work as expected"
                    )
                mean = np.maximum(mean, 1)
                img_mat = 100 * (img_mat / mean - 1)

            ##########################
            # not sure how to do this so no caching used currently
            # if nistats_flm.memory:
            #     mem_glm = nistats_flm.memory.cache(run_glm, ignore=["n_jobs"])
            # else:
            #     mem_glm = run_glm

            # compute GLM
            if nistats_flm.verbose > 1:
                t_glm = time.time()
                sys.stderr.write("Performing GLM computation\r")

            os.environ["OMP_NUM_THREADS"] = "1"
            self.run_3dREMLfit(
                runtime,
                img_mat,
                design,
                contrasts,
                noise_model=nistats_flm.noise_model,
                bins=100,
                n_jobs=nistats_flm.n_jobs,
            )

            if nistats_flm.verbose > 1:
                t_glm = time.time() - t_glm
                sys.stderr.write("GLM took %d seconds         \n" % t_glm)

            #########################
            # Results are saved to self instead in run_3dREMLfit, if the
            # memory saving is required it should be implemented there
            # nistats_flm.labels_.append(labels)
            # # We save memory if inspecting model details is not necessary
            # if nistats_flm.minimize_memory:
            #     for key in results:
            #         results[key] = SimpleRegressionResults(results[key])
            # nistats_flm.results_.append(results)

        # Report progress
        if nistats_flm.verbose > 0:
            sys.stderr.write(
                "\nComputation of %d runs done in %i seconds\n\n"
                % (n_runs, time.time() - t0)
            )

    def run_3dREMLfit(
        self, runtime, img_mat, design, contrasts, noise_model, bins, n_jobs
    ):
        import nibabel as nb
        import sys

        tmpdir = op.join(runtime.cwd)
        self.tmpdir = tmpdir
        bold_file = op.join(tmpdir, "input.nii.gz")
        nb.Nifti1Image(img_mat, None).to_filename(bold_file)
        glt = "glt_results"
        extra_vars = "glt_extra_variables"
        t_r = self.inputs.session_info["repetition_time"]
        stim_labels = self.get_stim_labels()
        zscores = "zscore_maps"
        pvals = "pval_maps"

        # Write AFNI style design matrix to file
        afni_design = get_afni_design_matrix(design, contrasts, stim_labels, t_r)
        design_fname = op.join(tmpdir, "design.xmat.1D")
        Path(design_fname).write_text(afni_design)

        # use mask if available:
        if self.inputs.mask_file:
            mask_param = f"-mask {self.inputs.mask_file}"
        else:
            mask_param = ""

        # Define 3dREMLfit command
        reml_cmd = f"""\
        3dREMLfit
            -matrix {design_fname}
            -input {bold_file}
            -Rbuck {glt}
            -Rvar {extra_vars}
            -tout
            -fout
            -verb
            {mask_param}
        """

        # Reformat command string
        reml_cmd = " ".join(reml_cmd.split())

        # Define commands to get pvals and zscores
        pval_cmd = f"3dPval -prefix {pvals} {glt}+orig"
        z_cmd = f"3dPval -prefix {zscores} -zscore {glt}+orig"

        # Execute commands
        sys.stderr.write(
            f"3dREMLfit and 3dPval computation will be performed in: {tmpdir}\n"
        )
        sp.check_output(reml_cmd, shell=True, cwd=tmpdir, stderr=sp.STDOUT)
        sp.check_output(pval_cmd, shell=True, cwd=tmpdir, stderr=sp.STDOUT)
        sp.check_output(z_cmd, shell=True, cwd=tmpdir, stderr=sp.STDOUT)

        pval_bucket, pval_labels = load_bucket_by_prefix(tmpdir, pvals)
        zscore_bucket, z_labels = load_bucket_by_prefix(tmpdir, zscores)
        extra_bucket, extra_labels = load_bucket_by_prefix(tmpdir, extra_vars)
        stat_bucket, stat_labels = load_bucket_by_prefix(tmpdir, glt)

        # create maps object
        maps = {
            "effect_size": stat_bucket,
            "effect_variance": extra_bucket,
            "z_score": zscore_bucket,
            "p_value": pval_bucket,
            "stat": stat_bucket,
        }
        self.save_remlfit_results(maps, contrasts, runtime)

    def save_remlfit_results(self, maps, contrasts, runtime):
        """Parse  the AFNI "bucket" datasets written by 3dREMLfit and
        subsequently read using nibabel. Save the results to disk according to
        fitlins expectation.
        Parameters
        ----------
        maps : A dictionary of nibabel.interfaces.brikhead.AFNIImage objects
            keyed by output map type Description contrasts : TYPE Description
            runtime : TYPE Description
        contrasts : Object returned by nistats.contrasts.prepare_constrasts
        runtime : nipype runtime object
        """
        import nibabel as nb

        contrast_metadata = []
        effect_maps = []
        variance_maps = []
        stat_maps = []
        zscore_maps = []
        pvalue_maps = []
        fname_fmt = op.join(runtime.cwd, "{}_{}.nii.gz").format

        out_ents = self.inputs.contrast_info[0]["entities"]

        stat_types = np.array(
            [
                x.split("(")[0].replace("none", "").replace("test", "")
                for x in maps["effect_size"].header.info["BRICK_STATSYM"].split(";")
            ]
        )
        vol_labels = maps["effect_size"].header.get_volume_labels()

        effect_bool = np.array([x.endswith("Coef") for x in vol_labels])
        for (name, weights, contrast_type) in contrasts:
            contrast_metadata.append(
                {"contrast": name, "stat": contrast_type, **out_ents}
            )

            # Get boolean to index appropriate values
            stat_bool = stat_types == contrast_type.upper()
            contrast_bool = np.array([name in x for x in vol_labels])
            stdev_bool = np.array(
                [
                    "StDev" == x
                    for x in maps["effect_variance"].header.get_volume_labels()
                ]
            )

            # Indices for multi image nibabel object  should have length 1 and be integers
            stat_idx = np.where(contrast_bool & stat_bool)[0]
            stdev_idx = np.where(stdev_bool)[0]
            # For multirow ftests there will be more than one index
            effect_idx = np.where(contrast_bool & effect_bool)[0]

            # Append maps:
            # for each index into the result objects stored in maps, apply a
            # modifying function as required and then append it to the
            # appropriate output list type represented by map_list and write
            # map_list to disk
            for map_type, map_list, idx_list, mod_func in (
                ("effect_size", effect_maps, effect_idx, None),
                ("effect_variance", variance_maps, stdev_idx, np.square),
                ("z_score", zscore_maps, stat_idx, None),
                ("p_value", pvalue_maps, stat_idx, None),
                ("stat", stat_maps, stat_idx, None),
            ):

                if len(effect_idx) > 1:
                    continue
                # If not defined mod_func will be identity
                if not mod_func:

                    def mod_func(x):
                        return x

                # Extract maps and info from bucket and append to relevant
                # list of maps
                for idx in idx_list:
                    img = maps[map_type].slicer[..., int(idx)]
                    intent_info = get_intent_info_for_subvol(img, idx)
                    intent_info = (*intent_info, f"{map_type} of contrast {name}")

                    outmap = nb.Nifti1Image(img.get_data(), img.affine)
                    outmap = set_intents([outmap], [intent_info])[0]

                    # Write maps to disk
                    fname = fname_fmt(name, map_type)
                    outmap.to_filename(fname)

                    map_list.append(fname)

        self._results["effect_maps"] = effect_maps
        self._results["variance_maps"] = variance_maps
        self._results["stat_maps"] = stat_maps
        self._results["zscore_maps"] = zscore_maps
        self._results["pvalue_maps"] = pvalue_maps
        self._results["contrast_metadata"] = contrast_metadata

    def get_stim_labels(self):
        # Iterate through all weight specifications to get a list of stimulus
        # column labels.
        weights = _flatten([x["weights"] for x in self.inputs.contrast_info])
        return list(set(_flatten([x.keys() for x in weights])))


class AFNIMergeAll(MergeAll):
    def _list_outputs(self):
        outputs = self._outputs().get()
        for key in self._fields:
            val = getattr(self.inputs, key)
            outputs[key] = [elem for sublist in val for elem in sublist]

        return outputs


def get_afni_design_matrix(design, contrasts, stim_labels, t_r):
    """Add appropriate metadata to the design matrix and write to file for
    calling 3dREMLfit.  For a description of the target format see
    https://docs.google.com/document/d/1zpujpZYuleB7HuIFjb2vC4sYXG5M97hJ655ceAj4vE0/edit

    Parameters
    ----------
    design : pandas.DataFrame
        Matrix containing regressor for model fit.
    contrasts : output of nistats.contrasts.prepare_contrasts
    stim_labels : List of strings specifying model conditions
    t_r : float
        TR in seconds

    Returns
    -------
    str
        Design matrix with AFNI niml header
    """

    cols = list(design.columns)
    stim_col_nums = sorted([cols.index(x) for x in stim_labels])

    # Currently multi-column stimuli not supported. If they were stim_tops
    # would need to be computed
    stim_pos = "; ".join([str(x) for x in stim_col_nums])

    column_labels = "; ".join(cols)
    test_info = create_glt_test_info(design, contrasts)
    design_vals = design.to_csv(sep=" ", index=False, header=False)

    design_mat = f"""\
        # <matrix
        # ni_type = "{design.shape[1]}*double"
        # ni_dimen = "{design.shape[0]}"
        # RowTR = "{t_r}"
        # GoodList = "0..{design.shape[0] - 1}"
        # NRowFull = "{design.shape[0]}"
        # CommandLine = "{' '.join(sys.argv)}"
        # ColumnLabels = "{column_labels}"
        # {test_info}
        # Nstim = {len(stim_labels)}
        # StimBots = "{stim_pos}"
        # StimTops = "{stim_pos}"
        # StimLabels = "{'; '.join(stim_labels)}"
        # >
        {design_vals}
        # </matrix>
        """

    design_mat = "\n".join([line.lstrip() for line in design_mat.splitlines()])
    return design_mat


def create_glt_test_info(design, contrasts):

    labels, wts_arrays, test_vals = zip(*contrasts)

    # Start defining a list containing the rows for the glt values in the
    # afni design matrix header:
    glt_list = [f'Nglt = "{len(labels)}"', f'''GltLabels = "{'; '.join(labels)}"''']

    # Convert weight arrays to csv strings
    glt_list += get_glt_rows(wts_arrays)

    # Add an empty line and start all other lines with #
    test_info = "\n# ".join([""] + glt_list)
    return test_info


def get_glt_rows(wt_arrays):
    """Generates the appropriate text for generalized linear testing in 3dREMLfit.

    Parameters
    ----------
    wt_arrays : tuple of np.arrays One of the
        Description

    Returns
    -------
    TYPE
        Description
    """
    glt_rows = []
    for ii, wt_array in enumerate(wt_arrays):
        bio = io.BytesIO()
        np.savetxt(bio, wt_array, delimiter="; ", fmt="%g", newline="; ")
        wt_str = bio.getvalue().decode("latin1")

        glt_rows.append(
            f'GltMatrix_{ii:06d} = "{wt_array.shape[0]}; {wt_array.shape[1]}; {wt_str}"'
        )

    return glt_rows


def load_bucket_by_prefix(tmpdir, prefix):
    import nibabel as nb

    bucket_path = list(Path(tmpdir).glob(prefix + "*HEAD"))
    if not len(bucket_path) == 1:
        paths = ", ".join(bucket_path)
        raise ValueError(
            f"""
            Only one file should be found for {prefix}. Instead found '{paths}'
            """
        )
    bucket_path = str(bucket_path[0])
    bucket = nb.load(bucket_path)
    bucket_labels = bucket.header.get_volume_labels()
    return bucket, bucket_labels


def set_intents(img_list, intent_info):
    for img, intent in zip(img_list, intent_info):
        img.header.set_intent(*intent)
    return img_list


def get_intent_info_for_subvol(brick, idx=0):
    intent_info = get_intent_info(brick)
    return intent_info[idx]


def get_intent_info(brick):
    intent_info = []
    if "BRICK_STATSYM" not in brick.header.info:
        intent_info = [
            ("none", ()) for x in range(len(brick.header.get_volume_labels()))
        ]
        return intent_info

    statsyms = brick.header.info["BRICK_STATSYM"].split(";")
    nlabels = len(brick.header.get_volume_labels())
    if nlabels != len(statsyms):
        raise ValueError(
            f"Unexpected number of BRICK_STATSYM values : '{nlabels}' instead of '{len(statsyms)}'"
        )
    for statsym in statsyms:
        val = statsym.replace(")", "").split("(")
        if val == ["none"]:
            val.append(tuple())
            intent_info.append(tuple(val))
        else:
            params = [x for x in val[1].split(",")]
            intent_info.append(
                (STAT_CODES.label[val[0]], tuple([int(x) for x in params if x]))
            )

    return intent_info
