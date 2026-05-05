'''
Utilities to visualize data vectors in Cobaya-CosmoLike Joint Architecture.

Functions
=========
get_starting_index : Get the starting index for given function and bin pair.
confidence_ellipse : Draw a confidence ellipse on a 2D plot.
plot_mu_and_cov : Plot the mean and covariance of two parameters.
make_corner_plots : Create corner plots for specified parameters.

Classes
=======
DVViz : Visualize best-fit data vector and given model vector.
DVDeriv : Compute and store partial derivatives of the data vector.

'''

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from itertools import combinations_with_replacement, product
from scipy.stats import linregress
from cocoa_dvcalc import SPACE, ci, Provider, DVCalc, Params


ntomo = 8
n_coord = 15  # coord = theta or ell.
if SPACE == "real": ggl_exclude = [(6,0),(7,0),(7,1)]
elif SPACE == "fourier": ggl_exclude = [(7,0),(7,1)]

pairs_ss = list(combinations_with_replacement(range(ntomo), 2))
pairs_gs = [(l, s) for (l, s) in product(range(ntomo), repeat=2)
            if (l, s) not in ggl_exclude]
pairs_gg = [(l, l) for l in range(ntomo)]
n_ss, n_gs, n_gg = len(pairs_ss), len(pairs_gs), len(pairs_gg)

def get_starting_index(func, b1, b2):
    try:
        if func in ['xi_p', 'xi_m', 'C_ss']:
            return n_coord * (pairs_ss.index((b1, b2)) +\
                n_ss * (func == 'xi_m'))
        elif func in ['w_gammat', 'C_gs']:
            return n_coord * (pairs_gs.index((b1, b2)) +\
                n_ss * (1 + (not func.startswith('C_'))))
        elif func in ['w_gg', 'C_gg']:
            return n_coord * (pairs_gg.index((b1, b2)) +\
                n_ss * (1 + (not func.startswith('C_'))) + n_gs)
    except:
        return None


if SPACE == "fourier":
    mask_fourier = np.where(pd.read_csv(
        "./projects/roman_cpip_data_challenge/data_challenge1_fourier_medium/dc1.mask",
        delimiter='\s+', skiprows=0, header=None)[1].values)[0]
    df = pd.read_csv("./projects/roman_cpip_data_challenge/data_challenge1_fourier_medium/cov_dc1",
        delimiter='\s+', skiprows=0, header=None, names=
        ['i', 'j', 'ell_i', 'ell_j', 'b1_i', 'b2_i', 'b1_j', 'b2_j', 'g', 'ng'])
    df = df[df.i == df.j].sort_values(by='i', ascending=True)
    df = df[df.i == df.j].sort_values(by='i', ascending=True)
    error_fourier = np.sqrt((df.g + df.ng).values); del df


class DVViz:
    POSTERIOR = True  # Whether to use MAP solution instead of ML solution.

    def __init__(self, i_mv: int = 1):
        Provider.INPUT_FILE = os.path.join(
            os.path.dirname(Provider.INPUT_FILE),
            f"EXAMPLE_EVALUATE_v{i_mv}.yaml".replace("_v1.", "."))
        self.filename = f"./projects/cocoa_dvcalc/dc1_{SPACE}_v{i_mv}.json"
        if self.POSTERIOR: self.filename = self.filename.replace(".json", "_map.json")
        self.params = Params(DVCalc(), self.filename)
        self.datavector = self.params.datavector

        if SPACE == "real":
            self.coord = ci.get_binning_real_space()
            self.mask = np.where(ci.get_mask())[0]
            self.error = np.sqrt(ci.get_cov_masked().diagonal())
        elif SPACE == "fourier":
            self.coord = ci.get_binning_fourier_space()
            self.mask = mask_fourier
            self.error = error_fourier

        self.modelvector = pd.read_csv(
            f"./projects/roman_cpip_data_challenge/data_challenge1_{SPACE}_medium/" +\
            ("challenge1.modelvector" if i_mv == 1 else f"alternates/dc1_v{i_mv}.modelvector"),
            delimiter='\s+', skiprows=0, header=None)[1].values

    def plot_segment(self, ax, s_idx, frac: bool = False):
        if s_idx is None:
            ax.set_axis_off(); return
        l_idx = self.mask[np.searchsorted(self.mask, s_idx)]
        r_idx = self.mask[np.searchsorted(self.mask, s_idx+n_coord)-1]

        my_coord = self.coord[l_idx%n_coord:r_idx%n_coord+1]
        my_slice = np.s_[l_idx:r_idx+1]
        if not frac:
            ax.plot(my_coord, self.datavector[my_slice])
            ax.plot(my_coord, self.modelvector[my_slice])
            ax.plot(my_coord, (self.datavector+self.error)[my_slice], ls=':')
            ax.plot(my_coord, (self.datavector-self.error)[my_slice], ls=':')
        else:
            ax.plot(my_coord, np.zeros_like(my_coord))
            ax.plot(my_coord, self.modelvector[my_slice] / self.datavector[my_slice] - 1)
            ax.plot(my_coord, self.error[my_slice] / self.datavector[my_slice], ls=':')
            ax.plot(my_coord, -self.error[my_slice] / self.datavector[my_slice], ls=':')
        ax.set_xscale("log")

    def plot_func(self, func, frac: bool = False):
        if func not in ['w_gg', 'C_gg']:
            fig, axs = plt.subplots(ntomo, ntomo, figsize=(14, 12), sharex=True)
            for b1 in range(ntomo):
                for b2 in range(ntomo):
                    self.plot_segment(axs[b2, b1], get_starting_index(func, b1, b2), frac)

        else:
            fig, axs = plt.subplots(1, ntomo, figsize=(14, 1.2), sharex=True)
            for b1 in range(ntomo):
                self.plot_segment(axs[b1], get_starting_index(func, b1, b1), frac)

        plt.show()
    
    def plot_all(self, frac: bool = False):
        print(self.params.logp)
        if SPACE == "real": all_funcs = ['xi_p', 'xi_m', 'w_gammat', 'w_gg']
        elif SPACE == "fourier": all_funcs = ['C_ss', 'C_gs', 'C_gg']
        for func in all_funcs: self.plot_func(func, frac)


class DVDeriv(DVViz):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dv0 = self.params.datavector

    def get_datavector(self, param, value, delta):
        self.params.set_value(param, value + delta)
        datavector = self.params.datavector
        self.params.set_value(param, value)
        return datavector

    def five_point_derivative(self, param, shrink=0, mode='stencil'):
        value = self.params.params[param]
        hstep = self.params.scales[param] / 2.0 ** shrink

        dvs = np.zeros((5, len(self.dv0)))
        dvs[2] = self.dv0
        for sigma in [-2, -1, 1, 2]:
            dvs[sigma + 2] = self.get_datavector(param, value, sigma * hstep)

        if mode == 'stencil':
            deriv = (dvs[0] - dvs[1] * 8 + dvs[3] * 8 - dvs[4]) / 12
        elif mode == 'regression':
            deriv = np.zeros(len(self.dv0))
            for i in range(len(self.dv0)):
                deriv[i] = linregress(np.arange(-2, 3), dvs[:, i]).slope

        deriv_ = np.mean(np.diff(dvs, axis=0), axis=0)  # Quality check.
        quality = np.linalg.norm((deriv - deriv_)) / np.linalg.norm(deriv)
        return deriv / hstep, quality

    def compute_derivatives(self, filename=None, verbose=True):
        self.derivs = {}

        for param in Params.ALLPARAMS:
            mode = 'regression' if param in Params.COSMOLOGY else 'stencil'
            deriv, quality = self.five_point_derivative(param, mode=mode)
            if quality > 1e-6:
                for shrink in range(2, 32, 2):
                    deriv_, quality_ = self.five_point_derivative(param, shrink=shrink, mode=mode)
                    if quality_ < quality:
                        deriv, quality = deriv_, quality_
            self.derivs[param] = deriv
            if verbose: print(param, quality)

        if filename is not None:
            np.savez(filename, base=self.dv0, **self.derivs)


# https://matplotlib.org/3.1.1/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(mu, cov, ax, n_std=3.0, facecolor='none', zorder=1, **kwargs):
    assert mu.shape == (2,), "mu must be a 2D vector"
    assert cov.shape == (2, 2), "cov must be a 2x2 covariance matrix"

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        zorder=zorder,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mu[0]

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mu[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


labels = dict(
    omegam=r"$\Omega_m$", sigma8=r"$\sigma_8$",
    ns=r"$n_s$", omegab=r"$\Omega_b$", h0=r"$h_0$",
    **{"roman_B1_"+str(i+1): rf"$b^{i+1}$" for i in range(8)},
    **{"roman_DZ_S"+str(i+1): rf"$\Delta_{{z}}^{i+1}$" for i in range(8)},
    **{"roman_M"+str(i+1): rf"$m_{i+1}$" for i in range(8)},
    roman_A1_1=r"$A_{\rm IA}$", roman_A1_2=r"$\eta_{\rm IA}$"
)


def plot_mu_and_cov(case, mode, param_j, param_i, ax,
                    c, marker="+", s=50, n_std=1.0, ls="-", lw=1, zorder=1):
    idx_j = case.allparams.index(param_j)
    idx_i = case.allparams.index(param_i)
    if mode in ["ML", "MAP"]:
        mu = case.values_ML[[idx_j, idx_i]]
    elif mode in ["MLp"]:
        mu = case.values_MAP[[idx_j, idx_i]]

    if mode in ["ML"]:
        cov = case.invfsh[np.ix_([idx_j, idx_i], [idx_j, idx_i])]
    elif mode in ["MLp", "MAP"]:
        cov = case.invfshp[np.ix_([idx_j, idx_i], [idx_j, idx_i])]

    ax.scatter(*mu, c=c, marker=marker, s=s, zorder=zorder)
    confidence_ellipse(mu, cov, ax, n_std=n_std, zorder=zorder, edgecolor=c, ls=ls, lw=lw)


def make_corner_plots(ML, MAP, params):
    fig, axs = plt.subplots(*(len(params)-1,)*2, figsize=((len(params)-1)*2,)*2,
                            sharex="col", sharey="row")

    for j, param_j in enumerate(params[:-1]):
        for i, param_i in enumerate(params[1:]):
            ax = axs[i, j]
            if i < j:
                ax.set_axis_off()
                continue

            plot_mu_and_cov(ML, "ML", param_j, param_i, ax, "C0")
            plot_mu_and_cov(ML, "MLp", param_j, param_i, ax, "C1")
            plot_mu_and_cov(MAP, "MAP", param_j, param_i, ax, "C2")

    for j, param_j in enumerate(params[:-1]):
        axs[-1, j].set_xlabel(labels.get(param_j))
    for i, param_i in enumerate(params[1:]):
        axs[i, 0].set_ylabel(labels.get(param_i))

    fig.subplots_adjust(hspace=0, wspace=0)
    plt.show()

