'''
Utilities to calculate data vectors in Cobaya-CosmoLike Joint Architecture.

Classes
=======
Provider : Parse YAML file and retrieve CAMB power spectra.
DVCalc : Data vector calculator for specified base class.
Params : Collection of cosmological and nuisance parameters to vary.

CoordDesc : Coordinate descent optimizer for parameters.
CoordDescPCA : Principal Component Analysis version of CoordDesc.

Functions
=========
show_matrix : Display a covariance matrix using matplotlib.
get_corr : Calculate the correlation matrix from a covariance matrix.
generate_dataset_and_yaml : Generate dataset and YAML files for alternates.

'''

import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
from matplotlib.colors import SymLogNorm, Normalize
from contextlib import redirect_stdout

PATH = os.path.dirname(__file__)
SPACE = "fourier"  # "real" or "fourier"
os.chdir("../..")
sys.path.append(f"./projects/roman_{SPACE}/interface")

import camb
print("Using CAMB %s installed at %s"%(camb.__version__,os.path.dirname(camb.__file__)))
if SPACE == "real":
    import cosmolike_roman_real_interface as ci
    from cobaya.likelihoods.roman_real.roman_real_3x2pt import roman_real_3x2pt as DVCalcBase
elif SPACE == "fourier":
    import cosmolike_roman_fourier_interface as ci
    from cobaya.likelihoods.roman_fourier.roman_fourier_3x2pt import roman_fourier_3x2pt as DVCalcBase
from cobaya.input import load_input_file


As = lambda As_1e9: 1e-9 * As_1e9
wa = lambda w0pwa, w: w0pwa - w
omegabh2 = lambda omegab, H0: omegab*(H0/100)**2
omegach2 = lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708
omegamh2 = lambda omegam, H0: omegam*(H0/100)**2

class Provider:
    INPUT_FILE = f"./projects/roman_cpip_data_challenge/data_challenge1_{SPACE}_medium/EXAMPLE_EVALUATE.yaml"
    LIKELIHOOD_KEY = f"roman_{SPACE}.roman_{SPACE}_3x2pt"

    def __init__(self, dvcalc):
        self.dvcalc = dvcalc  # For z_interp_2D.
        for key, value in load_input_file(self.INPUT_FILE).items():
            setattr(self, key, value)
        self.values = self.sampler["evaluate"]["override"].copy()

        self.priors = {}
        for param in self.params:
            if (prior := self.params[param].get("prior")) is None: continue
            if not prior.get("dist") == "norm": continue
            assert prior["loc"] == 0.0, \
                f"Prior for {param} is not centered at zero: {prior['loc']}"
            self.priors[param] = prior["scale"]

    def get_param(self, param):
        if param in self.values: return self.values[param]
        if param in ["w0pwa", "w", "mnu", "tau"]: return self.params[param]["value"]
        if param == "As": return As(self.get_param("As_1e9"))
        if param == "wa": return wa(self.get_param("w0pwa"), self.get_param("w"))
        if param == "omegabh2": return omegabh2(self.get_param("omegab"), self.get_param("H0"))
        if param == "omegach2": return omegach2(self.get_param("omegam"), self.get_param("omegab"),
                                                self.get_param("mnu"), self.get_param("H0"))
        if param == "omegamh2": return omegamh2(self.get_param("omegam"), self.get_param("H0"))
        if param.startswith("roman_DZ_L"): return self.get_param("roman_DZ_S" + param[-1])

    def set_params(self, **params):
        self.values.update(params)
        kmax = self.likelihood[self.LIKELIHOOD_KEY]["kmax_boltzmann"] \
             * self.likelihood[self.LIKELIHOOD_KEY]["accuracyboost"]
        camb_params = dict(
            H0=self.get_param("H0"), ombh2=self.get_param("omegabh2"), omch2=self.get_param("omegach2"),
            mnu=self.get_param("mnu"), omk=0, tau=self.get_param("tau"),
            As=self.get_param("As"), ns=self.get_param("ns"), lmax=10, kmax=kmax,
            **{k: v for k, v in self.theory["camb"]["extra_args"].items() if k != "dark_energy_model"})
        if camb_params != getattr(self, "camb_cache", None):
            self.get_camb_results(kmax, camb_params)
            self.camb_cache = camb_params

    def get_camb_results(self, kmax, camb_params):
        pars = camb.set_params(**camb_params)
        pars.set_dark_energy(w=self.get_param("w"), wa=self.get_param("wa"), dark_energy_model="ppf")
        pars.set_matter_power(redshifts=self.dvcalc.z_interp_2D, kmax=kmax, silent=True)
        self.results = camb.get_results(pars)

    def get_Pk_interpolator(self, var_pair, nonlinear, extrap_kmax):
        PK = self.results.get_matter_power_interpolator(
            var1=var_pair[0], var2=var_pair[1], nonlinear=nonlinear,
            extrap_kmax=extrap_kmax, hubble_units=False, k_hunit=False)
        PK.logP = lambda z, k: np.log(PK.P(z, k))
        return PK

    def get_comoving_radial_distance(self, z):
        return self.results.comoving_radial_distance(z)


class DVCalc(DVCalcBase):
    INPUT_FILE = f"./projects/roman_{SPACE}/likelihood/roman_{SPACE}_3x2pt.yaml"
    LIKELIHOOD_KEY = f"roman_{SPACE}.roman_{SPACE}_3x2pt"
    class log: info = staticmethod(lambda *args, **kwargs: None)

    def __init__(self):
        for key, value in load_input_file(self.INPUT_FILE).items():
            setattr(self, key, value)
        self.provider = Provider(self)
        for key, value in self.provider.likelihood[self.LIKELIHOOD_KEY].items():
            setattr(self, key, value)
        self.initialize()


class Params:
    COSMOLOGY = ['As_1e9', 'ns', 'H0', 'omegab', 'omegam']
    ALLPARAMS = COSMOLOGY +\
        ["roman_DZ_S"+str(i+1) for i in range(8)] +\
        ["roman_M"+str(i+1) for i in range(8)] +\
        ["roman_A1_"+str(i+1) for i in range(2)] +\
        ["roman_B1_"+str(i+1) for i in range(8)]
    ZEROS = dict.fromkeys(
        ["roman_A1_"+str(i+1) for i in range(2, 8)] +
        ["roman_A2_"+str(i+1) for i in range(8)] +
        ["roman_BTA_"+str(i+1) for i in range(8)] +
        ["roman_BARYON_Q"+str(i+1) for i in range(4)] +
        ["roman_B2_"+str(i+1) for i in range(8)] +
        ["roman_BMAG_"+str(i+1) for i in range(8)] +
        ["roman_PM"+str(i+1) for i in range(8)], 0.0)
    POSTERIOR = True  # Whether to use posterior probability.

    def __init__(self, dvcalc, filename):
        self.dvcalc = dvcalc
        self.provider = dvcalc.provider
        if os.path.exists(filename): self.load_params(filename)
        else: self.params = self.provider.sampler["evaluate"]["override"].copy()
        self.scales = {param: self.provider.params[param]["ref"]["scale"]
                       for param in self.params}

    def load_params(self, filename):
        with open(filename, "r") as f:
            self.params = json.load(f)

    def save_params(self, filename):
        # if self.POSTERIOR:
        #     filename = filename.replace(".json", "_map.json")
        with open(filename, "w") as f:
            json.dump(self.params, f, indent=2)

    @property
    def param_values(self):
        values = self.params.copy()
        for param in ["As_1e9", "ns"]: del values[param]
        for param in ["w", "mnu", "As"]: values[param] = self.provider.get_param(param)
        for i in range(8): values["roman_DZ_L"+str(i+1)] = self.params["roman_DZ_S"+str(i+1)]
        values["_derived"] = {}; values.update(self.ZEROS)
        return values

    @property
    def logp(self):
        self.provider.set_params(**self.params)
        logp = self.dvcalc.logp(**self.param_values)
        if self.POSTERIOR: logp += self.logprior
        return logp

    @property
    def logprior(self):
        logp = 0.0
        for param, value in self.params.items():
            if (scale := self.provider.priors.get(param)) is None: continue
            logp += -0.5 * (value / scale) ** 2
        return logp

    @property
    def datavector(self):
        self.provider.set_params(**self.params)
        return self.dvcalc.get_datavector(**self.param_values)

    @property
    def datavector_and_logp(self):
        datavector = self.datavector
        return (datavector, self.dvcalc.compute_logp(datavector))

    def set_value(self, param, value):
        self.params[param] = value

    def try_value(self, param, value):
        value_ = self.params[param]
        self.set_value(param, value)
        try:
            logp = self.logp
        except camb.baseconfig.CAMBError:
            logp = -np.inf
        self.set_value(param, value_)
        return logp

    def set_values(self, params, values):
        for param, value in zip(params, values):
            self.set_value(param, value)

    def try_values(self, params, values):
        values_ = {param: self.params[param] for param in params}
        self.set_values(params, values)
        try:
            logp = self.logp
        except camb.baseconfig.CAMBError:
            logp = -np.inf
        self.set_values(values_.keys(), values_.values())
        return logp

    def explore_variants(self):
        print("original param_values ->", f"{self.logp:.6f}")
        for param in self.params:
            value, scale = self.params[param], self.scales[param]
            print(param, end=": ")
            for delta in [-scale, scale]:
                print(f"{value:.6f}" + ("-" if delta < 0 else "+") + f"{scale:.6f}",
                      f"{self.try_value(param, value + delta):.6f}",
                      sep=" -> ", end=", " if delta < 0 else "\n")
            self.params[param] = value


class CoordDesc:

    def __init__(self, filename, shrink=0.5, shrink_init=0,
                 rtol=1e-6, drthresh=1e-3, verbose=True):
        self.filename = filename
        self.params = Params(DVCalc(), self.filename)
        self.current_logp = self.params.logp
        self.scales = self.params.scales.copy()

        for param in Params.COSMOLOGY:
            self.scales[param] *= shrink ** shrink_init
        self.shrink = shrink  # Factor to shrink step sizes.
        self.shrink_count = shrink_init  # Count of shrink iterations.
        self.shrink_max = np.ceil(np.log(rtol) / np.log(shrink)).astype(int)

        self.rtol = rtol  # Relative tolerance.
        self.drthresh = drthresh  # Threshold for dynamic range.
        self.verbose = verbose  # Whether to print progress.

    def try_value(self, param, new_value, verb="Accepted"):
        new_logp = self.params.try_value(param, new_value)
        if new_logp > self.current_logp:
            if self.verbose:
                print(f"{verb} {param}: {new_value:.6f} -> logp: {new_logp:.6f}")
            self.params.set_value(param, new_value)
            self.current_logp = new_logp
            return None
        else:
            return new_logp

    def optimize_param(self, param):
        value, scale = self.params.params[param], self.scales[param]

        for symbol in [-1, 1]:
            while True:
                new_value = value + symbol * scale
                new_logp = self.try_value(param, new_value)
                if new_logp is None:
                    self.improved = True
                    value = new_value
                else:
                    break

            if symbol < 0: logp_minus = new_logp
            elif symbol > 0: logp_plus = new_logp

        # Center of the parabola fit to the three points.
        if min(logp_minus, logp_plus, self.current_logp) > -np.inf:
            center = (logp_minus - logp_plus) / 2 \
                / (logp_minus + logp_plus - self.current_logp * 2)
            new_value = value + center * scale
            self.try_value(param, new_value, verb="Refined")
            self.dynranges[param] = min(logp_minus, logp_plus, self.current_logp) \
                / max(logp_minus, logp_plus, self.current_logp) - 1

    def optimize(self, maxiter=10, oversamp=2, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        param_list = list(self.scales.keys())
        self.dynranges = {param: np.inf for param in param_list}

        for it in range(maxiter):
            print(f"Iteration {it+1}/{maxiter}, current logp: {self.current_logp:.6f}")

            self.improved = False
            for ov in range(oversamp):
                for param in param_list:
                    if param in Params.COSMOLOGY and ov > 0:
                        continue
                    self.optimize_param(param)

            if not self.improved:
                self.shrink_count += 1
                if self.shrink_count > self.shrink_max:
                    print(f"Stopping optimization: reached relative tolerance\n", flush=True)
                    break
                if max(self.dynranges.values()) < self.drthresh:
                    print(f"Stopping optimization: reached small dynamic ranges\n", flush=True)
                    break
                if self.verbose:
                    print(f"No improvement in iteration {it+1}, shrinking scales by",
                          f"{self.shrink:.2f} ({self.shrink_count}/{self.shrink_max})")
                for param in param_list:
                    if self.dynranges[param] < self.drthresh:
                        continue
                    self.scales[param] *= self.shrink

            print(flush=True)
        self.params.save_params(self.filename)

    def driver(self, n_meta=1, **kwargs):
        with redirect_stdout(open(self.filename.replace("json", "log"), "w")):
            for i_meta in range(n_meta):
                self.optimize(**kwargs)


class CoordDescPCA(CoordDesc):
    PROPOSAL = "./projects/roman_cpip_data_challenge/data_challenge1_real_medium/DC1_PROPOSAL.covmat"
    RESCALE = True

    def __init__(self, filename, shrink=0.5, shrink_init=0,
                 rtol=1e-6, drthresh=1e-3, verbose=True):
        super().__init__(filename, shrink=shrink, shrink_init=shrink_init,
                         rtol=rtol, drthresh=drthresh, verbose=verbose)

        self.proposal = np.genfromtxt(self.PROPOSAL)
        self.eigenvalues, self.eigenvectors = np.linalg.eig(
            get_corr(self.proposal) if self.RESCALE else self.proposal)
        self.pc_sigmas = np.sqrt(self.eigenvalues)
        for i_pc in range(len(Params.ALLPARAMS)):
            self.pc_sigmas[i_pc] *= shrink ** shrink_init
        self.shrink_count_pca = shrink_init  # Count of shrink iterations.

        self.delta_vecs = {}
        if self.RESCALE: scalings = np.sqrt(np.diagonal(self.proposal))
        for i_pc in range(self.eigenvectors.shape[1]):
            self.eigenvectors[:, i_pc] *= np.sign(
                self.eigenvectors[np.argmax(np.abs(self.eigenvectors[:, i_pc])), i_pc])
            self.delta_vecs[i_pc] = self.eigenvectors[:, i_pc]
            if self.RESCALE: self.delta_vecs[i_pc] *= scalings

    def try_pc(self, i_pc, new_value, new_vec, verb="Accepted"):
        new_logp = self.params.try_values(Params.ALLPARAMS, new_vec)
        if new_logp > self.current_logp:
            if self.verbose:
                print(f"{verb} PC{i_pc}: {new_value:+.6f} -> logp: {new_logp:.6f}")
            self.params.set_values(Params.ALLPARAMS, new_vec)
            self.current_logp = new_logp
            return None
        else:
            return new_logp

    def optimize_pc(self, i_pc):
        param_vec = np.array(list(self.params.params.values()))
        delta_vec = self.delta_vecs[i_pc]
        value, scale = 0, self.pc_sigmas[i_pc]

        for symbol in [-1, 1]:
            while True:
                new_value = value + symbol * scale
                new_vec = param_vec + new_value * delta_vec
                new_logp = self.try_pc(i_pc, new_value, new_vec)
                if new_logp is None:
                    self.improved = True
                    value = new_value
                else:
                    break

            if symbol < 0: logp_minus = new_logp
            elif symbol > 0: logp_plus = new_logp

        # Center of the parabola fit to the three points.
        if min(logp_minus, logp_plus, self.current_logp) > -np.inf:
            center = (logp_minus - logp_plus) / 2 \
                / (logp_minus + logp_plus - self.current_logp * 2)
            new_value = value + center * scale
            new_vec = param_vec + new_value * delta_vec
            self.try_pc(i_pc, new_value, new_vec, verb="Refined")
            self.dynranges[i_pc] = min(logp_minus, logp_plus, self.current_logp) \
                / max(logp_minus, logp_plus, self.current_logp) - 1

    def optimize(self, maxiter=10, oversamp=2, pca_mode=False, **kwargs):
        if not pca_mode: 
            super().optimize(maxiter=maxiter, oversamp=oversamp, **kwargs)
            return

        for key, value in kwargs.items():
            setattr(self, key, value)
        pc_list = list(range(len(Params.ALLPARAMS)))
        self.dynranges = {i_pc: np.inf for i_pc in pc_list}

        for it in range(maxiter):
            print(f"Iteration {it+1}/{maxiter}, current logp: {self.current_logp:.6f}")

            self.improved = False
            for i_pc in pc_list:
                self.optimize_pc(i_pc)

            if not self.improved:
                self.shrink_count_pca += 1
                if self.shrink_count_pca > self.shrink_max:
                    print(f"Stopping optimization: reached relative tolerance\n", flush=True)
                    break
                if max(self.dynranges.values()) < self.drthresh:
                    print(f"Stopping optimization: reached small dynamic ranges\n", flush=True)
                    break
                if self.verbose:
                    print(f"No improvement in iteration {it+1}, shrinking scales by",
                          f"{self.shrink:.2f} ({self.shrink_count_pca}/{self.shrink_max})")
                for i_pc in pc_list:
                    if self.dynranges[i_pc] < self.drthresh:
                        continue
                    self.pc_sigmas[i_pc] *= self.shrink

            print(flush=True)
        self.params.save_params(self.filename)

    def driver(self, n_meta=1, pca_mode=False, **kwargs):
        if not pca_mode: 
            super().driver(n_meta=n_meta, **kwargs)
            return

        with redirect_stdout(open(self.filename.replace("json", "log"), "w")):
            for i_meta in range(n_meta):
                self.optimize(pca_mode=False, **kwargs)
                self.optimize(pca_mode=True, **kwargs)


def show_matrix(fig, ax, mat, symlog=False):
    vmin, vmax = mat.min(), mat.max()
    gain = max(abs(vmin), abs(vmax))
    if symlog:
        norm = SymLogNorm(vmin=-gain, vmax=gain,
                          linthresh=max(10**(np.rint(np.log10(gain))-8), 1e-16))
    else:
        norm = Normalize(vmin=-gain, vmax=gain)

    im = ax.imshow(mat, cmap='bwr', norm=norm, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylim((vmin, vmax))


def get_corr(cov):
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return corr


def generate_dataset_and_yaml(space=SPACE):
    cwd = os.getcwd()
    os.chdir(f"./projects/roman_cpip_data_challenge/data_challenge1_{space}_medium")

    for mv in glob.glob("./alternates/*.modelvector"):
        name = Path(mv).stem

        os.system(f"cp dc1.dataset {name}.dataset")
        with open(f"{name}.dataset", "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "data_file" in line:
                lines[i] = lines[i].replace("challenge1.modelvector", mv[2:])
                break
        with open(f"{name}.dataset", "w") as f:
            f.writelines(lines)

        os.system(f"cp EXAMPLE_EVALUATE.yaml EXAMPLE_EVALUATE{name[3:]}.yaml")
        with open(f"EXAMPLE_EVALUATE{name[3:]}.yaml", "r") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "data_file" in line:
                lines[i] = lines[i].replace("dc1.dataset", f"{name}.dataset")
                break
        with open(f"EXAMPLE_EVALUATE{name[3:]}.yaml", "w") as f:
            f.writelines(lines)

    os.chdir(cwd)

