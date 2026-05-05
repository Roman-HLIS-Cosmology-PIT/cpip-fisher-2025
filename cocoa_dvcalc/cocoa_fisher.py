"""
Utilities to make Fisher matrix forecasts for Roman HLIS cosmology.

Functions
---------
get_cov : Read in the covariance matrices from CosmoCov output files.
get_mask : Read in the mask from CosmoCov output files.
show_matrix : Display a matrix using matplotlib with appropriate normalization.
invert_matrix : Invert a covariance matrix using the correlation trick.

Classes
-------
FisherMeta : Handle meta information about real space or Fourier space.
FisherCase : A specific choice of space, tomographic bins, and angular scales.
FisherViz : Visualize partial derivatives of the data vector.

Functions
---------
explore_tomographic_bins : Explore interesting subsets of tomographic bins.
explore_angular_scales : Explore interesting subsets of angular scales.
explore_NG_coefficient : Explore coefficient for the non-Gaussian covariance.
explore_nuisance_priors : Explore different priors on nuisance parameters.
explore_all_variations : Driver function to explore all variations.

"""

import os
import json
from itertools import combinations_with_replacement, product, combinations
from pprint import pprint
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
from scipy.stats import linregress

import camb
from cobaya.input import load_input_file


def get_cov(filename, load_or_save: bool = True):
    if load_or_save and os.path.exists(filename+".npy"):
        with open(filename+".npy", "rb") as f:
            cov_g = np.load(f)
            cov_ng = np.load(f)
        ndata = cov_g.shape[0]
        return cov_g, cov_ng, ndata

    df = pd.read_csv(filename, delimiter="\s+", skiprows=0, header=None,
                     comment="#", usecols=[0, 1, 8, 9], names=["i", "j", "g", "ng"])
    ndata = int(np.max(df.i))+1
    print("Dimension of cov: %dx%d"%(ndata,ndata))
    cov_g = np.zeros((ndata,ndata))
    cov_ng = np.zeros((ndata,ndata))

    for row in df.itertuples():
        cov_g [row.i, row.j] = cov_g [row.j, row.i] = row.g
        cov_ng[row.i, row.j] = cov_ng[row.j, row.i] = row.ng
    del df

    if load_or_save:
        with open(filename+".npy", "wb") as f:
            np.save(f, cov_g)
            np.save(f, cov_ng)
    return cov_g, cov_ng, ndata


def get_mask(filename):
    df = pd.read_csv(filename, delimiter="\s+", skiprows=0, header=None)
    mask = np.where(df[1].values)[0]; del df
    return mask


def show_matrix(fig, ax, mat, symlog=False, oom=8):
    vmin, vmax = mat.min(), mat.max()
    gain = max(abs(vmin), abs(vmax))
    if symlog:
        norm = SymLogNorm(vmin=-gain, vmax=gain,
                          linthresh=max(10**(np.rint(np.log10(gain)-oom)), 1e-24))
    else:
        norm = Normalize(vmin=-gain, vmax=gain)

    im = ax.imshow(mat, cmap="bwr", norm=norm, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylim((vmin, vmax))


def invert_matrix(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:  # Singular matrix
        inv = np.empty_like(corr)
        inv.fill(np.nan)
    return inv / np.outer(std, std)


As = lambda As_1e9: 1e-9 * As_1e9
wa = lambda w0pwa, w: w0pwa - w
omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

class FisherMeta:
    COSMOLOGY = ["As_1e9", "ns", "H0", "omegab", "omegam"]
    ALLPARAMS = COSMOLOGY +\
        ["roman_DZ_S"+str(i+1) for i in range(8)] +\
        ["roman_M"+str(i+1) for i in range(8)] +\
        ["roman_A1_"+str(i+1) for i in range(2)] +\
        ["roman_B1_"+str(i+1) for i in range(8)]
    ntomo = 8
    n_coord = 15  # coord = theta or ell.
    POSTERIOR = True  # Whether to use MAP solution instead of ML solution.

    def __init__(self, space, i_mv=1, to_sigma8=True):
        self.space = space
        self.cov_g, self.cov_ng, self.ndata = get_cov(
            f"../roman_cpip_data_challenge/data_challenge1_{space}_medium/cov_dc1")
        self.mask = get_mask(
            f"../roman_cpip_data_challenge/data_challenge1_{space}_medium/dc1.mask")
        self.info_dict = load_input_file(
            f"../roman_cpip_data_challenge/data_challenge1_{space}_medium/" +\
            f"EXAMPLE_EVALUATE_v{i_mv}.yaml".replace("_v1.", "."))

        filestem = f"dc1_{space}_v{i_mv}" + ("_map" if self.POSTERIOR else "")
        with open(f"{filestem}.json", "r") as f: self.params = json.load(f)
        self.scales = {param: self.info_dict["params"][param]["ref"].get("scale") for param in self.ALLPARAMS}
        self.priors = {param: self.info_dict["params"][param]["prior"].get("scale") for param in self.ALLPARAMS}
        with open(f"{filestem}.npz", "rb") as f: self.derivs = dict(np.load(f))
        self.datav = self.derivs.pop("base")
        self.to_sigma8 = False
        if to_sigma8: self.switch_to_sigma8()

        self.ggl_exclude = list(map(tuple, self.info_dict["likelihood"]
                                    [f"roman_{space}.roman_{space}_3x2pt"]["ggl_exclude"]))
        self.pairs_ss = list(combinations_with_replacement(range(self.ntomo), 2))
        self.pairs_gs = [(l, s) for (l, s) in product(range(self.ntomo), repeat=2)
                         if (l, s) not in self.ggl_exclude]
        self.pairs_gg = [(l, l) for l in range(self.ntomo)]
        self.n_ss, self.n_gs, self.n_gg = len(self.pairs_ss), len(self.pairs_gs), len(self.pairs_gg)

    def get_starting_index(self, func, b1, b2):
        if self.space == "real":
            assert func in ["xi_p", "xi_m", "w_gammat", "w_gg"]
        elif self.space == "fourier":
            assert func in ["C_ss", "C_gs", "C_gg"]
        assert min(b1, b2) >= 0 and max(b1, b2) < self.ntomo

        try:
            if func in ["xi_p", "xi_m", "C_ss"]:
                return self.n_coord * (self.pairs_ss.index((b1, b2)) +\
                    self.n_ss * (func == "xi_m"))
            elif func in ["w_gammat", "C_gs"]:
                return self.n_coord * (self.pairs_gs.index((b1, b2)) +\
                    self.n_ss * (1 + (not func.startswith("C_"))))
            elif func in ["w_gg", "C_gg"]:
                return self.n_coord * (self.pairs_gg.index((b1, b2)) +\
                    self.n_ss * (1 + (not func.startswith("C_"))) + self.n_gs)
        except:
            return None

    def get_camb_params(self):
        # Provider.set_params from cocoa_dvcalc.py.
        likelihood_key = f"roman_{self.space}.roman_{self.space}_3x2pt"
        kmax = self.info_dict["likelihood"][likelihood_key]["kmax_boltzmann"] \
            * self.info_dict["likelihood"][likelihood_key]["accuracyboost"]

        H0, omegab, omegam = self.params["H0"], self.params["omegab"], self.params["omegam"]
        mnu = self.info_dict["params"]["mnu"]["value"]
        self.camb_params = dict(
            H0=H0, ombh2=omegabh2(omegab, H0), omch2=omegach2(omegam, omegab, mnu, H0),
            mnu=mnu, omk=0, tau=self.info_dict["params"]["tau"]["value"],
            As=As(self.params["As_1e9"]), ns=self.params["ns"], lmax=10, kmax=kmax,
            **{key: value for key, value in self.info_dict["theory"]["camb"]["extra_args"].items()
            if key != "dark_energy_model"})

        # Provider.get_camb_results from cocoa_dvcalc.py.
        w = self.info_dict["params"]["w"]["value"]
        w0pwa = self.info_dict["params"]["w0pwa"]["value"]
        self.dark_energy = dict(w=w, wa=wa(w0pwa, w), dark_energy_model="ppf")

    def get_sigma8(self, param: str, value: float) -> float:
        if param in self.COSMOLOGY:
            value_ = self.params[param]
            self.params[param] = value
            self.get_camb_params()
        elif not hasattr(self, "camb_params"):
            self.get_camb_params()

        pars = camb.set_params(**self.camb_params)
        pars.set_dark_energy(**self.dark_energy)
        pars.set_matter_power(redshifts=[0.0], kmax=self.camb_params["kmax"], silent=True)
        results = camb.get_results(pars)
        sigma8 = results.get_sigma8_0()

        if param in self.COSMOLOGY:
            self.params[param] = value_
            self.get_camb_params()

        return sigma8

    def get_psigma8_pparam(self, RES: int = 17) -> None:
        self.psigma8_pparam = {}
        factors = np.linspace(-1, 1, RES)

        for param in self.COSMOLOGY:
            value = self.params[param]
            scale = self.scales[param]
            sigma8s = np.zeros(RES)

            for i, factor in enumerate(factors):
                sigma8s[i] = self.get_sigma8(param, value + scale * factor)
                deriv, _, _, _, _ = linregress(scale * factors, sigma8s)
                self.psigma8_pparam[param] = deriv

    def switch_to_sigma8(self):
        self.to_sigma8 = True
        if not hasattr(self, "psigma8_pparam"): self.get_psigma8_pparam()
        self.params["sigma8"] = self.get_sigma8("As_1e9", self.params["As_1e9"])
        del self.params["As_1e9"]
        # self.params["sigma8"] = self.As_1e9_to_sigma8(self.params.pop("As_1e9"))

        psigma8_pAs_1e9 = self.psigma8_pparam["As_1e9"]
        self.scales["sigma8"] = self.scales.pop("As_1e9") * psigma8_pAs_1e9
        self.priors["sigma8"] = None; assert self.priors.pop("As_1e9") is None
        self.derivs["sigma8"] = self.derivs.pop("As_1e9") / psigma8_pAs_1e9

        for param in self.COSMOLOGY:
            if param == "As_1e9": continue
            self.derivs[param] -= self.derivs["sigma8"] * self.psigma8_pparam[param]


class FisherCase:

    def __init__(self, meta, tomos=list(range(FisherMeta.ntomo)),
                 coords=list(range(FisherMeta.n_coord))):
        self.meta = meta; self.space = meta.space
        self.tomos = tomos; self.ntomo = len(tomos)
        self.coords = coords; self.n_coord = len(coords)
        self.mask = np.zeros(meta.ndata, dtype=bool)

        # cosmic shear
        for func in [["xi_p", "xi_m"], ["C_ss"]][self.space != "real"]:
            for b1, b2 in combinations_with_replacement(tomos, 2):
                s = meta.get_starting_index(func, b1, b2)
                for c in coords: self.mask[s + c] = True

        # galaxy-galaxy lensing
        func = "w_gammat" if self.space == "real" else "C_gs"
        for b1, b2 in product(tomos, repeat=2):
            if (s := meta.get_starting_index(func, b1, b2)) is not None:
                for c in coords: self.mask[s + c] = True

        # galaxy clustering
        func = "w_gg" if self.space == "real" else "C_gg"
        for b in tomos:
            s = meta.get_starting_index(func, b, b)
            for c in coords: self.mask[s + c] = True

        self.mask = np.where(self.mask)[0]
        self.mask = np.array([idx for idx in self.mask if idx in meta.mask])
        self.bdry_ss_gs = np.searchsorted(self.mask, meta.n_coord *\
            meta.n_ss * (1 + (self.space == "real")))
        self.bdry_gs_gg = np.searchsorted(self.mask, meta.n_coord *\
            (meta.n_ss * (1 + (self.space == "real")) + meta.n_gs))

    def get_submask(self, probes):
        submask = np.array([], dtype=self.mask.dtype)
        if "ss" in probes:
            submask = np.concatenate((submask, self.mask[:self.bdry_ss_gs]))
        if "gs" in probes:
            if self.bdry_ss_gs == self.bdry_gs_gg: return None
            submask = np.concatenate((submask, self.mask[self.bdry_ss_gs:self.bdry_gs_gg]))
        if "gg" in probes:
            if self.bdry_gs_gg == len(self.mask): return None
            submask = np.concatenate((submask, self.mask[self.bdry_gs_gg:]))
        return submask

    def get_priors_submat(self, prefix, var_all=1.0, var_which=-1,
                          var_single=1.0, corr_all=0.0, corr_pair=0.0):
        assert abs(corr_all) < 1.0, "corr_all must be in (-1, 1)"
        subcov = np.zeros((self.ntomo, self.ntomo))
        for itomo in range(self.ntomo):
            subcov[itomo, itomo] = self.meta.priors[f"{prefix}{itomo+1}"] ** 2

        if var_all != 1.0:
            subcov *= var_all ** 2
        if var_which >= 0:
            subcov[var_which, var_which] *= var_single ** 2

        if corr_all != 0.0:
            for j in range(self.ntomo):
                for i in range(j):
                    subcov[j, i] = subcov[i, j] =\
                        corr_all * np.sqrt(subcov[i, i] * subcov[j, j])

        if corr_pair != 0.0:
            for j in range(self.ntomo - 1):
                subcov[j, j + 1] = subcov[j + 1, j] =\
                    corr_pair * np.sqrt(subcov[j, j] * subcov[j + 1, j + 1])

        return invert_matrix(subcov)

    def get_priors_mat(self, subparams, **kwargs):
        priors = np.zeros((len(subparams),) * 2)
        for prefix in ["roman_DZ_S", "roman_M"]:
            try:
                s_idx = subparams.index(f"{prefix}{min(self.tomos)+1}")
                priors[s_idx:s_idx + self.ntomo, s_idx:s_idx + self.ntomo] =\
                    self.get_priors_submat(prefix, **{key[len(prefix)+1:]: value for key, value
                                                      in kwargs.items() if key.startswith(prefix)})
            except AssertionError as e:
                print(f"Error in {prefix}: {e}")
            except ValueError:
                continue
        return priors

    def explore_subcase(self, probes, coef_ng=1.0, visualize=False, **kwargs):
        submask = self.get_submask(probes)
        if submask is None:
            results = {}
            for i_ng in range(2):
                prefix = "".join(probes) + ("NG" if i_ng else "G")
                results[f"{prefix}_SNR2"] = np.nan
                for i_fom in range(1, 3):
                    for suffix in ["U", "", "P", "N", "NP"]:
                        results[f"{prefix}_FoM{i_fom}{suffix}"] = np.nan
            self.results.update(results)
            return

        subdatav = self.meta.datav[submask]
        subderivs = {param: deriv[submask] for param, deriv in self.meta.derivs.items()}

        subparams = ["sigma8"] + FisherMeta.ALLPARAMS[1:]
        for b in range(FisherMeta.ntomo):
            if b in self.tomos: continue
            for prefix in ["roman_DZ_S", "roman_M", "roman_B1_"]:
                subparams.remove(f"{prefix}{b+1}")
        if probes == ("ss",):
            subparams = [param for param in subparams
                         if not param.startswith("roman_B1_")]
        elif probes == ("gg",):
            subparams = [param for param in subparams
                         if not (param.startswith("roman_M") or param.startswith("roman_A1_"))]

        if visualize:
            print(f"Probes: {probes}")
            fig, axs = plt.subplots(1, 2, figsize=(8, 4))

        for i_ng, subinv in enumerate([invert_matrix(self.meta.cov_g[submask][:, submask]),
                                       invert_matrix((self.meta.cov_g + coef_ng *\
                                           self.meta.cov_ng)[submask][:, submask])]):
            prefix = "".join(probes) + ("NG" if i_ng else "G")
            results = {}  # Dictionary to store results.

            results[f"{prefix}_SNR2"] = np.einsum("i,ij,j->", subdatav, subinv, subdatav)
            fisher = np.zeros((len(subparams),) * 2)
            for j, param1 in enumerate(subparams):
                for i, param2 in enumerate(subparams):
                    if i < j:
                        fisher[j, i] = fisher[i, j]
                        continue
                    fisher[j, i] = np.einsum("i,ij,j->", subderivs[param1], subinv, subderivs[param2])

            if visualize:
                show_matrix(fig, axs[i_ng], fisher, symlog=True)
                axs[i_ng].set_title(f"Fisher matrix ({'G+NG' if i_ng else 'G'})")

            priors = self.get_priors_mat(subparams, **kwargs)
            maskgb = np.array([idx for idx, param in enumerate(subparams) if (idx in [0, 4]
                               or param.startswith("roman_B1_"))])  # gb: galaxy biases only.
            maskpm = np.array([idx for idx in range(len(subparams))
                               if not 1 <= idx <= 3])  # pm: partial marginalization.

            # Unmarginalized figure of metrits.
            # Priors on nuisance parameters do not matter.
            # results[f"{prefix}_FoM1U"] = fisher[0, 0]  # sigma8
            # results[f"{prefix}_FoM2U"] = np.sqrt(np.linalg.det(fisher[0:5:4, 0:5:4]))  # sigma8 and omegam

            # Update: always marginalizing over galaxy biases. (8/21/2025)
            invfshgb = invert_matrix(fisher[maskgb][:, maskgb])
            results[f"{prefix}_FoM1U"] = 1 / invfshgb[0, 0]
            results[f"{prefix}_FoM2U"] = 1 / np.sqrt(np.linalg.det(invfshgb[:2, :2]))

            # Marginalized figure of metrits, without priors.
            invfsh = invert_matrix(fisher)
            results[f"{prefix}_FoM1"] = 1 / invfsh[0, 0]
            results[f"{prefix}_FoM2"] = 1 / np.sqrt(np.linalg.det(invfsh[0:5:4, 0:5:4]))

            # Marginalized figure of metrits, with priors.
            invfshp = invert_matrix(fisher + priors)
            results[f"{prefix}_FoM1P"] = 1 / invfshp[0, 0]
            results[f"{prefix}_FoM2P"] = 1 / np.sqrt(np.linalg.det(invfshp[0:5:4, 0:5:4]))

            # Marginalized over nuisance parameters, without priors.
            invfshn = invert_matrix(fisher[maskpm][:, maskpm])
            results[f"{prefix}_FoM1N"] = 1 / invfshn[0, 0]
            results[f"{prefix}_FoM2N"] = 1 / np.sqrt(np.linalg.det(invfshn[:2, :2]))

            # Marginalized over nuisance parameters, with priors.
            invfshnp = invert_matrix((fisher + priors)[maskpm][:, maskpm])
            results[f"{prefix}_FoM1NP"] = 1 / invfshnp[0, 0]
            results[f"{prefix}_FoM2NP"] = 1 / np.sqrt(np.linalg.det(invfshnp[:2, :2]))

            if visualize and i_ng:
                fig_, axs_ = plt.subplots(2, 2, figsize=(8, 8))
                for i_mat, mat in enumerate(["invfsh", "invfshp", "invfshn", "invfshnp"]):
                    ax = axs_.ravel()[i_mat]
                    show_matrix(fig_, ax, vars()[mat], symlog=True)
                    ax.set_title(f"{mat} (logdet: {np.linalg.slogdet(vars()[mat])[1]:.4f})")

            if visualize: pprint(results)
            self.results.update(results)
        if visualize: plt.show()
        # return fisher, priors, maskpm

    def __call__(self, coef_ng=1.0, visualize=False, **kwargs):
        self.results = {}
        for i_sub in range(7, 0, -1):
            probes = []
            if i_sub & 4: probes.append("ss")
            if i_sub & 2: probes.append("gs")
            if i_sub & 1: probes.append("gg")

            self.explore_subcase(tuple(probes), coef_ng=coef_ng,
                                 visualize=visualize, **kwargs)

    def compute_values_and_errors(self, probes=("ss", "gs", "gg"), store_mats=False):
        assert self.tomos == list(range(FisherMeta.ntomo)), "Not the base case."
        assert self.coords == list(range(FisherMeta.n_coord)), "Not the base case."

        submask = self.get_submask(probes)
        subderivs = {param: deriv[submask] for param, deriv in self.meta.derivs.items()}
        # subparams = ["sigma8"] + FisherMeta.ALLPARAMS[1:]
        subparams = FisherMeta.ALLPARAMS.copy()
        if self.meta.to_sigma8: subparams[0] = "sigma8"

        subinv = invert_matrix((self.meta.cov_g + self.meta.cov_ng)[submask][:, submask])
        fisher = np.zeros((len(subparams),) * 2)
        for j, param1 in enumerate(subparams):
            for i, param2 in enumerate(subparams):
                if i < j:
                    fisher[j, i] = fisher[i, j]
                    continue
                fisher[j, i] = np.einsum("i,ij,j->", subderivs[param1], subinv, subderivs[param2])

        # Convert H0 to h0.
        idx = subparams.index("H0")
        fisher[idx, :] *= 100.0
        fisher[:, idx] *= 100.0

        priors = self.get_priors_mat(subparams)
        invfsh = invert_matrix(fisher)
        invfshp = invert_matrix(fisher + priors)
        if store_mats:
            self.fisher = fisher
            self.priors = priors
            self.invfsh = invfsh
            self.invfshp = invfshp

        self.allparams = subparams.copy(); self.allparams[idx] = "h0"
        self.values_ML = np.array([self.meta.params[param] for param in subparams])
        self.errors_ML = np.sqrt(np.diagonal(invfsh))
        self.values_MAP = invfshp @ (fisher @ self.values_ML + priors @ np.zeros_like(self.values_ML))
        self.errors_MAP = np.sqrt(np.diagonal(invfshp))

        # Convert H0 to h0.
        idx = subparams.index("H0")
        self.allparams[idx] = "h0"
        for arr in [self.values_ML, self.values_MAP]:
            arr[idx] /= 100.0


def explore_tomographic_bins(meta, df):
    print("explore_tomographic_bins", flush=True)

    # 4 consecutive tomographic bins.
    for i_bin in tqdm(range(5)):
        tomos = list(range(i_bin, i_bin+4))
        case = FisherCase(meta, tomos=tomos); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"tomo_{i_bin}to{i_bin+3}").T])

    # All combinations of 2 tomographic bins.
    for pair in tqdm(list(combinations(range(8), 2))):
        case = FisherCase(meta, tomos=list(pair)); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"tomo_{pair[0]}and{pair[1]}").T])

    # Jackknife studies for tomographic bins.
    for i_bin in tqdm(range(8)):
        case = FisherCase(meta, tomos=sorted(set(range(8))-{i_bin})); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"tomo_no{i_bin}").T])

    # Cumulative studies for tomographic bins.
    for i_bin in tqdm(range(0, 6)):
        if f"tomo_0to{i_bin}" in df.index: continue
        case = FisherCase(meta, tomos=list(range(i_bin+1))); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"tomo_0to{i_bin}").T])
    # tomo_0to6 = tomo_no7, tomo_0to7 = base

    for i_bin in tqdm(range(7, 1, -1)):
        if f"tomo_{i_bin}to7" in df.index: continue
        case = FisherCase(meta, tomos=list(range(i_bin, 8))); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"tomo_{i_bin}to7").T])
    # tomo_1to7 = tomo_no0, tomo_0to7 = base

    return df


def explore_angular_scales(meta, df):
    print("explore_angular_scales", flush=True)

    # Jackknife studies for angular scales.
    for i_coord in tqdm(range(15)):
        case = FisherCase(meta, coords=sorted(set(range(15))-{i_coord})); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"coord_no{i_coord}").T])

    # Cumulative studies for angular scales.
    for i_coord in tqdm(range(2, 15-1)):
        case = FisherCase(meta, coords=list(range(i_coord+1))); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"coord_0to{i_coord}").T])

    for i_coord in tqdm(range(14-1, -1, -1)):
        case = FisherCase(meta, coords=list(range(i_coord, 15))); case()
        df = pd.concat([df, pd.Series(case.results).to_frame(
            name=f"coord_{i_coord}to14").T])

    return df


def explore_NG_coefficient(base, df):
    print("explore_NG_coefficient", flush=True)

    for coef in tqdm(np.linspace(0.1, 1.0, 9, endpoint=False)):
        base(coef_ng=coef)
        df = pd.concat([df, pd.Series(base.results).to_frame(
            name=f"coef_ng_{coef:.1f}").T])

    return df


def explore_nuisance_priors(base, prefix, df):
    print("explore_nuisance_priors:", prefix, flush=True)

    for scale in tqdm(np.geomspace(0.1, 10, 9)):
        base(**{f"{prefix}_var_all": scale})
        df = pd.concat([df, pd.Series(base.results).to_frame(
            name=f"{prefix[6:]}_var_all_{scale:.1f}").T])

    for i_bin in range(8):
        for scale in tqdm(np.geomspace(0.1, 10, 9)):
            base(**{f"{prefix}_var_which": i_bin, f"{prefix}_var_single": scale})
            df = pd.concat([df, pd.Series(base.results).to_frame(
                name=f"{prefix[6:]}_var_{i_bin}_{scale:.1f}").T])

    for symbol in [+1, -1]:
        for coef in tqdm(symbol * np.linspace(0.1, 1.0, 9, endpoint=False)):
            base(**{f"{prefix}_corr_all": coef})
            df = pd.concat([df, pd.Series(base.results).to_frame(
                name=f"{prefix[6:]}_corr_all_{coef:.1f}").T])

    for symbol in [+1, -1]:
        for coef in tqdm(symbol * np.linspace(0.1, 1.0, 9, endpoint=False)):
            base(**{f"{prefix}_corr_pair": coef})
            df = pd.concat([df, pd.Series(base.results).to_frame(
                name=f"{prefix[6:]}_corr_pair_{coef:.1f}").T])

    return df


def explore_all_variations(space):
    print("explore_all_variations", flush=True)

    meta = FisherMeta(space, to_sigma8=True)
    base = FisherCase(meta); base(visualize=False)
    df = pd.Series(base.results).to_frame(name="base").T

    df = explore_tomographic_bins(meta, df)
    df = explore_angular_scales(meta, df)
    df = explore_NG_coefficient(base, df)
    for prefix in ["roman_DZ_S", "roman_M"]:
        df = explore_nuisance_priors(base, prefix, df)

    df.to_csv(f"cocoa_fisher_{space}.csv")
    # df = pd.read_csv(f"cocoa_fisher_{space}.csv", index_col=0)


class FisherViz(FisherMeta):

    def __init__(self, space, to_sigma8=True):
        super().__init__(space, to_sigma8=to_sigma8)
        self.scales = dict(As_1e9=0.06, sigma8=0.03, ns=0.06, H0=1.2, omegab=0.004, omegam=0.015,
                           roman_A1_1=0.04, roman_A1_2=0.24)  # Somewhat arbitrary.
        for i in range(8):
            self.scales["roman_DZ_S"+str(i+1)] = 0.003
            self.scales["roman_M"+str(i+1)] = 0.005
            self.scales["roman_B1_"+str(i+1)] = 0.01
        self.deltas = {param: self.derivs[param] * self.scales[param]
                       for param in ["sigma8" if to_sigma8 else "As_1e9"] + self.ALLPARAMS[1:]}

        # From DVViz in cocoa_dvviz.py.
        if space == "real": self.coord = [
            2.97200132245545, 4.0400089913792465, 5.491812041638084, 7.465329796304372, 10.148043768622234,
            13.794808151791631, 18.75206062208114, 25.490733448767543, 34.65099142176315, 47.1030466394507,
            64.02982747918657, 87.03935519068173, 118.31750373639919, 160.835654856868, 218.63297531081636]
        elif space == "fourier": self.coord = [
            35.31445806308525, 48.93449698153415, 67.80749659411795, 93.95941264291237, 130.19756911313132,
            180.4120154240586, 249.99311070922303, 346.4101615137752, 480.0132278028123, 665.1441685740268,
            921.6761942439148, 1277.147191799222, 1769.7158282998685, 2452.257760926468, 3398.0416685322934]
        self.error = np.sqrt((self.cov_g + self.cov_ng).diagonal())

    def plot_segment(self, ax, s_idx, frac: bool = False, params: list[str] = []):
        if s_idx is None:
            ax.set_axis_off(); return
        l_idx = self.mask[np.searchsorted(self.mask, s_idx)]
        r_idx = self.mask[np.searchsorted(self.mask, s_idx+self.n_coord)-1]

        my_coord = self.coord[l_idx%self.n_coord:r_idx%self.n_coord+1]
        my_slice = np.s_[l_idx:r_idx+1]
        if not frac:
            ax.plot(my_coord, self.datav[my_slice], ls="-", c="C0")
            for i, param in enumerate(params, start=1):
                ax.plot(my_coord, (self.datav + self.deltas[param])[my_slice], ls="--", c=f"C{i}")
            ylim = ax.get_ylim()
            for symbol in [+1, -1]:
                ax.plot(my_coord, (self.datav + symbol * self.error)[my_slice], ls=":", c="C9")
            ax.set_ylim(ylim)
        else:
            ax.plot(my_coord, np.zeros_like(my_coord), ls="-", c="C0")
            for i, param in enumerate(params, start=1):
                ax.plot(my_coord, self.deltas[param][my_slice] / self.datav[my_slice], ls="--", c=f"C{i}")
            ylim = ax.get_ylim()
            for symbol in [+1, -1]:
                ax.plot(my_coord, symbol * self.error[my_slice] / self.datav[my_slice], ls=":", c="C9")
            ax.set_ylim(ylim)
        ax.set_xscale("log")

    def plot_func(self, func, frac: bool = False, params: list[str] = []):
        if func not in ["w_gg", "C_gg"]:
            fig, axs = plt.subplots(self.ntomo, self.ntomo, figsize=(14, 12), sharex=True)
            for b1 in range(self.ntomo):
                for b2 in range(self.ntomo):
                    self.plot_segment(axs[b2, b1], self.get_starting_index(func, b1, b2), frac, params)

            ax = axs[0, -1]
            ax.plot([], [], ls="-", c="C0", label="base")
            for i, param in enumerate(params, start=1):
                ax.plot([], [], ls="--", c=f"C{i}", label=param)
            ax.plot([], [], ls=":", c="C9", label="error")
            ax.legend(loc="upper left")

        else:
            fig, axs = plt.subplots(1, self.ntomo, figsize=(14, 1.2), sharex=True)
            for b1 in range(self.ntomo):
                self.plot_segment(axs[b1], self.get_starting_index(func, b1, b1), frac, params)

        plt.show()

    def plot_all(self, frac: bool = False, params: list[str] = []):
        # print(self.space, "space:"); print({param: self.scales[param] for param in params})
        if self.space == "real": all_funcs = ["xi_p", "xi_m", "w_gammat", "w_gg"]
        elif self.space == "fourier": all_funcs = ["C_ss", "C_gs", "C_gg"]
        if all(param.startswith("roman_B1_") for param in params): all_funcs = all_funcs[-2:]
        if all(param.startswith("roman_M") for param in params): all_funcs = all_funcs[:-1]
        for func in all_funcs: self.plot_func(func, frac, params)

