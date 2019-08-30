# coding: utf-8
import os
import numpy as np
import pandas as pd
from warnings import warn

from nipype.interfaces.base import isdefined

from .nistats import FirstLevelModel, prepare_contrasts


class FirstLevelModel(FirstLevelModel):
    def _run_interface(self, runtime):
        import nibabel as nb
        from nistats import design_matrix as dm
        from nistats import first_level_model as level1
        info = self.inputs.session_info
        img = nb.load(self.inputs.bold_file)
        vols = img.shape[3]

        if info['sparse'] not in (None, 'None'):
            sparse = pd.read_hdf(info['sparse'], key='sparse').rename(
                columns={'condition': 'trial_type',
                         'amplitude': 'modulation'})
            sparse = sparse.dropna(subset=['modulation'])  # Drop NAs
        else:
            sparse = None

        if info['dense'] not in (None, 'None'):
            dense = pd.read_hdf(info['dense'], key='dense')
            column_names = dense.columns.tolist()
            drift_model = None if 'cosine_00' in column_names else 'cosine'
        else:
            dense = None
            column_names = None
            drift_model = 'cosine'

        mat = dm.make_first_level_design_matrix(
            frame_times=np.arange(vols) * info['repetition_time'],
            events=sparse,
            add_regs=dense,
            add_reg_names=column_names,
            drift_model=drift_model,
        )

        mat.to_csv('design.tsv', sep='\t')
        self._results['design_matrix'] = os.path.join(runtime.cwd,
                                                      'design.tsv')

        mask_file = self.inputs.mask_file
        if not isdefined(mask_file):
            mask_file = None
        smoothing_fwhm = self.inputs.smoothing_fwhm
        if not isdefined(smoothing_fwhm):
            smoothing_fwhm = None
        flm = level1.FirstLevelModel(
            mask_img=mask_file, smoothing_fwhm=smoothing_fwhm)
        flm.fit(img, design_matrices=mat)

        effect_maps = []
        variance_maps = []
        stat_maps = []
        zscore_maps = []
        pvalue_maps = []
        contrast_metadata = []
        out_ents = self.inputs.contrast_info[0]['entities']
        fname_fmt = os.path.join(runtime.cwd, '{}_{}.nii.gz').format
        for name, weights, contrast_type in prepare_contrasts(
                self.inputs.contrast_info, mat.columns.tolist()):
            maps = flm.compute_contrast(weights, contrast_type, output_type='all')
            contrast_metadata.append(
                {'contrast': name,
                 'stat': contrast_type,
                 **out_ents}
                )

            for map_type, map_list in (('effect_size', effect_maps),
                                       ('effect_variance', variance_maps),
                                       ('z_score', zscore_maps),
                                       ('p_value', pvalue_maps),
                                       ('stat', stat_maps)):
                fname = fname_fmt(name, map_type)
                maps[map_type].to_filename(fname)
                map_list.append(fname)

        self._results['effect_maps'] = effect_maps
        self._results['variance_maps'] = variance_maps
        self._results['stat_maps'] = stat_maps
        self._results['zscore_maps'] = zscore_maps
        self._results['pvalue_maps'] = pvalue_maps
        self._results['contrast_metadata'] = contrast_metadata

        return runtime

    def fit(self, run_imgs, events=None, confounds=None,
            design_matrices=None):
        """ Fit the GLM

        For each run:
        1. create design matrix X
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        run_imgs: Niimg-like object or list of Niimg-like objects,
            See http://nilearn.github.io/manipulating_images/input_output.html#inputing-data-file-names-or-image-objects
            Data on which the GLM will be fitted. If this is a list,
            the affine is considered the same for all.

        events: pandas Dataframe or string or list of pandas DataFrames or
                   strings
                   
            fMRI events used to build design matrices. One events object
            expected per run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        confounds: pandas Dataframe or string or list of pandas DataFrames or
                   strings
                   
            Each column in a DataFrame corresponds to a confound variable
            to be included in the regression model of the respective run_img.
            The number of rows must match the number of volumes in the
            respective run_img. Ignored in case designs is not None.
            If string, then a path to a csv file is expected.

        design_matrices: pandas DataFrame or list of pandas DataFrames,
            Design matrices that will be used to fit the GLM. If given it
            takes precedence over events and confounds.

        """
        # Check arguments
        # Check imgs type
        if events is not None:
            _check_events_file_uses_tab_separators(
                events_files=events)
        if not isinstance(run_imgs, (list, tuple)):
            run_imgs = [run_imgs]
        if design_matrices is None:
            if events is None:
                raise ValueError('events or design matrices must be provided')
            if self.t_r is None:
                raise ValueError('t_r not given to FirstLevelModel object'
                                 ' to compute design from events')
        else:
            design_matrices = _check_run_tables(run_imgs, design_matrices,
                                                'design_matrices')
        # Check that number of events and confound files match number of runs
        # Also check that events and confound files can be loaded as DataFrame
        if events is not None:
            events = _check_run_tables(run_imgs, events, 'events')
        if confounds is not None:
            confounds = _check_run_tables(run_imgs, confounds, 'confounds')

        # Learn the mask
        if self.mask_img is False:
            # We create a dummy mask to preserve functionality of api
            ref_img = check_niimg(run_imgs[0])
            self.mask_img = Nifti1Image(np.ones(ref_img.shape[:3]),
                                        ref_img.affine)
        if not isinstance(self.mask_img, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask_img, smoothing_fwhm=self.smoothing_fwhm,
                target_affine=self.target_affine,
                standardize=self.standardize, mask_strategy='epi',
                t_r=self.t_r, memory=self.memory,
                verbose=max(0, self.verbose - 2),
                target_shape=self.target_shape,
                memory_level=self.memory_level)
            self.masker_.fit(run_imgs[0])
        else:
            if self.mask_img.mask_img_ is None and self.masker_ is None:
                self.masker_ = clone(self.mask_img)
                for param_name in ['target_affine', 'target_shape',
                                   'smoothing_fwhm', 't_r', 'memory',
                                   'memory_level']:
                    our_param = getattr(self, param_name)
                    if our_param is None:
                        continue
                    if getattr(self.masker_, param_name) is not None:
                        warn('Parameter %s of the masker'
                             ' overriden' % param_name)
                    setattr(self.masker_, param_name, our_param)
                self.masker_.fit(run_imgs[0])
            else:
                self.masker_ = self.mask_img

        # For each run fit the model and keep only the regression results.
        self.labels_, self.results_, self.design_matrices_ = [], [], []
        n_runs = len(run_imgs)
        t0 = time.time()
        for run_idx, run_img in enumerate(run_imgs):
            # Report progress
            if self.verbose > 0:
                percent = float(run_idx) / n_runs
                percent = round(percent * 100, 2)
                dt = time.time() - t0
                # We use a max to avoid a division by zero
                if run_idx == 0:
                    remaining = 'go take a coffee, a big one'
                else:
                    remaining = (100. - percent) / max(0.01, percent) * dt
                    remaining = '%i seconds remaining' % remaining

                sys.stderr.write(
                    "Computing run %d out of %d runs (%s)\n"
                    % (run_idx + 1, n_runs, remaining))

            # Build the experimental design for the glm
            run_img = check_niimg(run_img, ensure_ndim=4)
            if design_matrices is None:
                n_scans = run_img.get_data().shape[3]
                if confounds is not None:
                    confounds_matrix = confounds[run_idx].values
                    if confounds_matrix.shape[0] != n_scans:
                        raise ValueError('Rows in confounds does not match'
                                         'n_scans in run_img at index %d'
                                         % (run_idx,))
                    confounds_names = confounds[run_idx].columns.tolist()
                else:
                    confounds_matrix = None
                    confounds_names = None
                start_time = self.slice_time_ref * self.t_r
                end_time = (n_scans - 1 + self.slice_time_ref) * self.t_r
                frame_times = np.linspace(start_time, end_time, n_scans)
                design = make_first_level_design_matrix(frame_times, events[run_idx],
                                                        self.hrf_model, self.drift_model,
                                                        self.high_pass, self.drift_order,
                                                        self.fir_delays, confounds_matrix,
                                                        confounds_names, self.min_onset)
            else:
                design = design_matrices[run_idx]
            self.design_matrices_.append(design)

            # Mask and prepare data for GLM
            if self.verbose > 1:
                t_masking = time.time()
                sys.stderr.write('Starting masker computation \r')

            Y = self.masker_.transform(run_img)

            if self.verbose > 1:
                t_masking = time.time() - t_masking
                sys.stderr.write('Masker took %d seconds       \n' % t_masking)

            if self.signal_scaling:
                Y, _ = mean_scaling(Y, self.scaling_axis)
            if self.memory:
                mem_glm = self.memory.cache(run_glm, ignore=['n_jobs'])
            else:
                mem_glm = run_glm

            # compute GLM
            if self.verbose > 1:
                t_glm = time.time()
                sys.stderr.write('Performing GLM computation\r')
            labels, results = mem_glm(Y, design.values,
                                      noise_model=self.noise_model,
                                      bins=100, n_jobs=self.n_jobs)
            if self.verbose > 1:
                t_glm = time.time() - t_glm
                sys.stderr.write('GLM took %d seconds         \n' % t_glm)

            self.labels_.append(labels)
            # We save memory if inspecting model details is not necessary
            if self.minimize_memory:
                for key in results:
                    results[key] = SimpleRegressionResults(results[key])
            self.results_.append(results)
            del Y

        # Report progress
        if self.verbose > 0:
            sys.stderr.write("\nComputation of %d runs done in %i seconds\n\n"
                             % (n_runs, time.time() - t0))

        return self