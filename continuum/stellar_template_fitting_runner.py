import numpy as np
#import ullyses_utils as uu
import matplotlib.pyplot as plt
import pandas as pd
from astropy.constants import c
c_in_km_s = c.to('km/s').value

from scipy import optimize
from scipy import interpolate

#import celerite
import stellar_template_fitting_atoms as stfa
from linetools.spectra.xspectrum1d import XSpectrum1D

import toml
import argparse


def define_exclusions(include_wave_range, exclude_wave_ranges,
                      exclude_line_vranges, dwave_pad=0.3):
    for name, vrange_def in exclude_line_vranges.items():
        exclude_llist = stfa.get_overlapping_lines(include_wrange=include_wave_range,
                                                   expand_vrange=vrange_def['vrange'],
                                                   dwave_pad=dwave_pad,
                                                   linelist=vrange_def['linelist'],
                                                   drop_species=vrange_def['drop_species'])
        these_ewrs = stfa.get_wranges_around_lines(expand_vrange=vrange_def['vrange'],
                                                   dwave_pad=dwave_pad, linelist=exclude_llist)
        exclude_wave_ranges[f'vrange_{name}'] = stfa.simplify_overlaps(these_ewrs)

    all_ewrs = np.concatenate(list(exclude_wave_ranges.values()), axis=0)
    all_ewrs = stfa.simplify_overlaps(all_ewrs)
    exclude_wave_ranges['all_merged'] = all_ewrs
    return exclude_wave_ranges


def read_spec(fname, autoscale=True):
    xspec = XSpectrum1D.from_file(fname)
    spec_df = pd.DataFrame(data=dict(wave=xspec.wavelength.value,
                                     flux=xspec.flux.value,
                                     err=xspec.sig.value))

    err_is_zero = spec_df.err==0
    spec_df.loc[err_is_zero, ['err', 'flux']] = np.nan
    if autoscale:
        medval = np.nanmedian(spec_df.flux.values)
        spec_df[['flux', 'err']] /= medval

    return spec_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instruction_file')
    args = parser.parse_args()

    ### reading in instruction file
    instruction_file = args.instruction_file#.replace('scripts/', '') #snakefile is one directory up
    with open(instruction_file, 'r') as f:
        instructions = toml.load(f)

    print(instructions)
    # should do some validating: no calling wranges "all_merged"

    # unpacking instructions
    target_info = instructions['target_info']
    target = target_info['target']
    problem = target_info['problem']
    #project = target_info['project']
    #target = target_info['target']
    #galaxy = target_info['galaxy']
    instrument = target_info['instrument']
    spec_file_name = target_info['spec_file']
    model_file_name = target_info['model_file']
    plot_ranges_kwargs = target_info['plot_ranges_kwargs']
    if 'dwave_pad' in target_info:
        dwave_pad = target_info['dwave_pad']
    else:
        dwave_pad = 0.3

    if 'gp_sigma' in target_info:
        log_sigma = np.log(target_info['gp_sigma'])
    else:
        log_sigma = np.log(0.5)
    
    if 'gp_rho' in target_info:
        log_rho = np.log(target_info['gp_rho'])
    else:
        log_rho = np.log(10)


    #log_rho = np.log(10)

    #dr = target_info['dr']

    include_wave_range = instructions['include_wave_range']['wave_range']
    exclude_wave_ranges = instructions['exclude_wave_ranges']
    exclude_line_vranges = instructions['exclude_vranges']

    vshift_def = instructions['vshift']
    vsini_def = instructions['vsini']


    ### defining some names
    fig_name = f'{problem}.pdf'
    fit_save_name = f'{problem}.csv'

    ### reading in spectra and applying exclusions
    exclude_wranges = define_exclusions(include_wave_range, exclude_wave_ranges,
                                        exclude_line_vranges, dwave_pad=dwave_pad)
    all_ewrs = exclude_wranges['all_merged']
    spec = read_spec(spec_file_name)
    #spec = uu.get_cspec(target, instrument, dr=dr)

    full_inc_mask = stfa.get_wrange_mask([include_wave_range], [], wave_ax=spec.wave.values)
    full_inc_spec = spec.iloc[full_inc_mask]

    full_to_no_line_mask = stfa.get_wrange_mask([include_wave_range], all_ewrs,
                                               wave_ax=full_inc_spec.wave.values)
    full_to_no_line_mask = full_to_no_line_mask & ~full_inc_spec.flux.isna().values

    if 'holdout' in exclude_wranges:
        full_inc_spec['is_held_out'] = ~stfa.get_wrange_mask([include_wave_range],
                                                              exclude_wranges['holdout'],
                                                              wave_ax=full_inc_spec.wave.values)
    full_eval_wave = full_inc_spec.wave.values
    if np.count_nonzero(full_to_no_line_mask) == 0:
        raise(ValueError(problem))

    ### initial fit
    dwave_pad = 15

    v_shifts, vsinis = [np.arange(vdef['min'], vdef['max']+0.1*vdef['delta'], vdef['delta'])
                        for vdef in [vshift_def, vsini_def]]

    models = stfa.load_models(model_file_name, instrument)
    wmin = full_eval_wave[0] / (1+vshift_def['max']/c_in_km_s) - dwave_pad
    wmax = full_eval_wave[-1] / (1+vshift_def['min']/c_in_km_s) + dwave_pad
    models = models.sel(wavelength=slice(wmin, wmax))

    polynomial_design = stfa.get_cheb_vander(full_eval_wave, 3)
    bounds = stfa.define_bounds(models, polynomial_design)

    best_fits = []
    best_fit_coeffs = []
    scores = []

    for vsini in vsinis:
        conv_models = stfa.get_convolved_models(vsini, models)
        conv_models = interpolate.interp1d(models.wavelength.values, conv_models.data, kind='linear')

        for v_shift in v_shifts:
            design_full = stfa.assemble_design_matrix(v_shift, conv_models, full_eval_wave, polynomial_design)
            design_no_line = design_full[full_to_no_line_mask]

            output = optimize.lsq_linear(design_no_line, full_inc_spec.iloc[full_to_no_line_mask].flux.values,
                                                bounds=bounds)
            coeffs = output.x
            score = output.cost
            best_fits.append(design_full @ coeffs)
            best_fit_coeffs.append(coeffs)
            scores.append(score)


    ### refinement of initial fit
    best_fit_idx = np.argmin(scores)
    vsini_idx, v_shift_idx = np.unravel_index(best_fit_idx, [vsinis.size, v_shifts.size])
    x0 = [vsinis[vsini_idx], v_shifts[v_shift_idx]]
    best_fit_coeff = best_fit_coeffs[best_fit_idx]

    top5 = np.sort(best_fit_coeff[:-polynomial_design.shape[1]])[-5]
    keep_coeffs = best_fit_coeff[:-polynomial_design.shape[1]]>=top5
    if np.count_nonzero(keep_coeffs) > 2:
        refine_models = models.isel(labels=keep_coeffs)
    else:
        refine_models = models.isel(labels=np.arange(4))

    refine_bounds = stfa.define_bounds(refine_models, polynomial_design)


    args = (full_inc_spec.flux.values[full_to_no_line_mask], refine_models, full_eval_wave,
            polynomial_design, full_to_no_line_mask, refine_bounds)

    refined_solution = optimize.fmin_l_bfgs_b(stfa.refine_cost, x0, approx_grad=True,
                                                  args=args, bounds=([4, 500], [-500, 1000]))

    coeffs, design_full = stfa.refine_cost(refined_solution[0], *args, get_solution=True)
    best_fit_spec = design_full @ coeffs


    if False:
        ### now add the GP bit and save
        resid = (full_inc_spec.flux.values-best_fit_spec)[full_to_no_line_mask]

        kern = celerite.terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)
        gp = celerite.GP(kern)
        gp.compute(full_eval_wave[full_to_no_line_mask], yerr=full_inc_spec.err.values[full_to_no_line_mask])

        gp_correction = gp.predict(resid, full_eval_wave, return_cov=False, return_var=False)
        cont_model = gp_correction+best_fit_spec

    full_inc_spec['best_fit_spec'] = best_fit_spec
    #full_inc_spec['cont_model'] = cont_model
    full_inc_spec.to_csv(fit_save_name, index=False)

    ### plot and save
    stfa.plot_solution(full_inc_spec, exclude_wranges, plot_ranges_kwargs=plot_ranges_kwargs,
         max_span=30)
    plt.savefig(fig_name)
