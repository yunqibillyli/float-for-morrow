import os
import numpy as np
import xarray
import pandas as pd
import ullyses_utils as uu
import matplotlib.pyplot as plt
from astropy.constants import c
c_in_km_s = c.to('km/s').value

from scipy import optimize
from scipy import interpolate
from scipy.signal import convolve2d
#import celerite

from linetools.lists.linelist import LineList
ism_linelist = LineList('ISM')._data.to_pandas()
ism_linelist['species'] = ism_linelist['name'].str.split(' ').str[0]
ism_linelist.set_index(['species', 'name'], inplace=True)

h2_linelist = LineList('H2')._data.to_pandas()
h2_linelist['species'] = 'H2_J' + h2_linelist.Jk.astype('int').astype('str')
h2_linelist.set_index(['species', 'name'], inplace=True)

# CO_linelist = pd.read_excel(os.path.expanduser('~/Dropbox/METAL/data/linelists/CO-lines.xlsx'))
# CO_linelist.rename(columns={'wavelength':'wrest'}, inplace=True)
# CO_linelist['species'] = 'CO'
# CO_linelist.set_index(['species'], inplace=True)

# linelists = {'CO':CO_linelist, 'H2':h2_linelist, 'ISM':ism_linelist}

linelists = {'H2':h2_linelist, 'ISM':ism_linelist}

### kernel things
def rotation_kernel(vsini, regrid_dv, ld_eps=0.5):
    dx = regrid_dv/vsini
    x = np.arange(0, 1 + 2*dx, dx)
    kernel = (2 * (1-ld_eps) * np.sqrt((1-x**2)) + 0.5*np.pi*ld_eps * (1-x**2))/(np.pi*(1-ld_eps/3))
    kernel = np.where(np.abs(x) < 1, kernel, 0)
    kernel = np.concatenate([np.flip(kernel[1:]), kernel])
    kernel /= kernel.sum()
    return kernel

def gaussian_kernel(fwhm, regrid_dv, clip=5):
    sd = fwhm / 2.355
    x = np.arange(0, sd*5 + 2*regrid_dv, regrid_dv) / regrid_dv
    gauss = np.exp(-0.5 * x**2)
    gauss = np.concatenate([gauss[1:][::-1], gauss])
    gauss /= gauss.sum()
    return gauss

def get_lsf_fwhm(instrument):
    """
    approximate the lsf as a gaussian; it's OK for the purpose of making a star template
    """
    if ('stis_e140m' in instrument) or ('stis_e230m' in instrument):
        return 8
    elif 'stis_e140h' in instrument:
        return 1.6
    elif 'cos' in instrument:
        return 16
    else:
        raise(ValueError(f'Unrecognized instrument: {instrument}'))


### wave range things
def simplify_overlaps(ranges):
    dstate = np.repeat([[1, -1]], ranges.shape[0], axis=0)
    x_flat = ranges.ravel()
    dstate_flat = dstate.ravel()

    order = np.argsort(x_flat)
    x_flat = x_flat[order]
    dstate_flat = dstate_flat[order]

    old_state = 0
    current_range = []
    ranges = []

    for x, dstate in zip(x_flat, dstate_flat):
        new_state = old_state + dstate
        if new_state<0:
            raise(ValueError('state went negative, this shouldn\'t happen'))
        elif (old_state == 0) and (new_state == 1):
            # start of a merged range
            current_range.append(x)
        elif (old_state == 1) and (new_state == 0):
            # end of a merged range
            current_range.append(x)
            ranges.append(current_range)
            current_range = []
        old_state = new_state
    ranges = np.array(ranges)
    return ranges


def get_overlapping_lines(include_wrange, expand_vrange, dwave_pad, linelist, drop_species=[]):
    linelist = linelists[linelist]
    expand_vrange = np.asarray(expand_vrange)
    zp1 = 1+expand_vrange/c_in_km_s

    overlap_llist = linelist.query(f'{include_wrange[0]*zp1[0]-dwave_pad}'
                                       f'< wrest < {include_wrange[1]*zp1[1]+dwave_pad}')
    overlap_llist = overlap_llist.loc[~overlap_llist.index.get_level_values(0).isin(drop_species)]
    return overlap_llist


def get_wranges_around_lines(expand_vrange, dwave_pad, linelist):
    expand_vrange = np.asarray(expand_vrange)
    zp1 = 1+expand_vrange/c_in_km_s
    wranges = linelist.wrest.values[:, None] * zp1
    wranges[:, 0] -= dwave_pad
    wranges[:, 1] += dwave_pad
    return wranges


def get_wrange_mask(inclusions, exclusions, wave_ax):
    mask = np.full(wave_ax.size, False)
    n_wave = wave_ax.size
    for wlo, whi in inclusions:
        idxs = np.clip(np.searchsorted(wave_ax, [wlo, whi]), 0, n_wave)
        mask[idxs[0]:idxs[1]] = True
    for wlo, whi in exclusions:
        idxs = np.clip(np.searchsorted(wave_ax, [wlo, whi]), 0, n_wave)
        mask[idxs[0]:idxs[1]] = False
    return mask


def drop_exclusions_outside_range(inclusions, exclusions):
    exclusions = np.asarray(exclusions)
    keep = np.full(exclusions.shape[0], True)
    for lo, hi in inclusions:
        to_left = exclusions[:, -1] < lo
        to_right = hi < exclusions[:, 0]
        keep = keep & ~(to_left|to_right)
    return exclusions[keep]


### plotting
def plot_spectrum(spec, ax=None, figsize=[9, 3], apply_tight_layout=True, add_labels=True, yq='flux',
                  color='dimgray', lw=1, ds='steps-mid', **plot_kwargs):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    ax.plot(spec.wave, spec[yq], color=color, lw=lw, ds=ds, **plot_kwargs)
    ax.set_ylim(0)
    if add_labels:
        ax.set_xlabel('Wavelength (Å)', fontsize=13)
        ax.set_ylabel('Flux (arbitrary)', fontsize=13)
    if apply_tight_layout:
        plt.tight_layout()
    return ax


def plot_ranges(ranges, ax=plt, color='tomato', alpha=0.6, zorder=1, **axvspan_kwargs):
    for lo, hi in ranges:
        ax.axvspan(lo, hi, color=color, alpha=alpha, zorder=zorder, **axvspan_kwargs)


def plot_solution(full_solution_df, exclude, max_span=50, wpad=0.2, plot_yqs_kwargs={},
                  plot_ranges_kwargs={}):
    wmin, wmax = full_solution_df.wave.min(), full_solution_df.wave.max()
    n_segments = int(np.ceil((wmax-wmin)/max_span))
    w_edges = np.linspace(wmin, wmax, n_segments+1)

    # fig, axes = plotch.sized_subplots(ny=n_segments, dx=8, dy=3, sharey=False)
    fig, axes = plt.subplots(nrows = n_segments, ncols = 1, figsize = (8, 3 * n_segments))

    axes = np.atleast_1d(axes)
    for seg_i in range(n_segments):
        lo, hi = w_edges[seg_i:seg_i+2]
        sub_solution_df = full_solution_df.query(f'{lo-wpad}<wave<{hi+wpad}')

        ax = axes[seg_i]
        ax.set_xlim(lo-wpad, hi+wpad)
        plot_spectrum(sub_solution_df, ax, add_labels=False, apply_tight_layout=False)
        #plot_spectrum(sub_solution_df, ax, add_labels=False, apply_tight_layout=False, yq='cont_model',
        #              color='C1', lw=2)
        plot_spectrum(sub_solution_df, ax, add_labels=False, apply_tight_layout=False, yq='best_fit_spec',
                      color='aquamarine', lw=1.5, ls='--')
        for range_name, kwargs in plot_ranges_kwargs.items():
            plot_ranges(exclude[range_name], ax, **kwargs)


### parts of fitting
def load_models(model_file, instrument):
    models = xarray.load_dataarray(model_file)
    lsf = gaussian_kernel(get_lsf_fwhm(instrument), models.attrs['regrid_dv'])
    labels = models.labels.values

    # little hack
    if "UVBLUE" in model_file:
        # these are already pretty low resolution, no need to make things worse
        pass
    else:
        for label in labels:
            models.loc[label] = np.convolve(models.loc[label], lsf, mode='same')

    return models


def get_convolved_models(vsini, models, ld_eps=0.5, wrange=None):
    rot_kern = rotation_kernel(vsini, models.attrs['regrid_dv'], ld_eps)
    models = models.data.copy()
    for e, m in enumerate(models):
        models[e] = np.convolve(m, rot_kern, mode='same')

    return models


def eval_shifted_models(v_shift, model_interpolator, eval_wave):
    return model_interpolator(eval_wave/(1+v_shift/c_in_km_s))


def get_cheb_vander(x, degree):
    x_min, x_max = x.min(), x.max()
    x_remap = 2 * (x - x_min)/(x_max-x_min) - 1
    return np.polynomial.chebyshev.chebvander(x_remap, degree)


def assemble_design_matrix(v_shift, models, eval_wave, polynomial_part):
    shifted_models = eval_shifted_models(v_shift, models, eval_wave)
    design = np.append(shifted_models.T, polynomial_part, axis=1)
    return design


def define_bounds(models, polynomial_part):
    bounds = (np.zeros(models.shape[0] + polynomial_part.shape[1]),
              np.full(models.shape[0] + polynomial_part.shape[1], np.inf))
    bounds[0][models.shape[0]:] = -np.inf
    return bounds


def refine_design(params, models, eval_wave, polynomial_part):
    vsini, v_shift = params
    conv_models = get_convolved_models(vsini, models)
    conv_models = interpolate.interp1d(models.wavelength.values, conv_models.data, kind='linear')
    design_full = assemble_design_matrix(v_shift, conv_models, eval_wave, polynomial_part)
    return design_full


def refine_cost(params, obs, models, eval_wave, polynomial_part, full_to_no_line_mask, bounds,
                get_solution=False):
    design_full = refine_design(params, models, eval_wave, polynomial_part)
    design_no_line = design_full[full_to_no_line_mask]

    output = optimize.lsq_linear(design_no_line, obs, bounds=bounds)
    if get_solution:
        return output.x, design_full
    else:
        return output.cost
