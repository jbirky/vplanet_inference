import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from functools import partial
import multiprocessing as mp
import os
import tqdm
import copy
import yaml
from yaml.loader import SafeLoader
from collections import OrderedDict
from astropy import units as u

from matplotlib import rc
try:
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
except:
    rc('text', usetex=False)


# import vplanet_inference as vpi
from .parameters import VplanetParameters
from .model import VplanetModel


__all__ = ["AnalyzeVplanetModel", "sort_yaml_key"]


def sort_yaml_key(base, sub_key):
    """Extract a sub-key value from each entry in a nested YAML dict.

    Iterates over the top-level keys of *base* and returns the value of
    ``base[key][sub_key]`` for each, attempting ``eval()`` then ``float()``
    conversion before falling back to ``None``.

    Parameters
    ----------
    base : dict
        Outer YAML dictionary (e.g. the ``"input_var"`` block).
    sub_key : str
        Name of the nested key to extract (e.g. ``"units"``, ``"bounds"``,
        ``"true_value"``).

    Returns
    -------
    list
        One entry per top-level key in *base*, converted to a Python object
        where possible, ``None`` otherwise.
    """
    sub_list = []
    for key in base.keys():
        try:
            sub_val = eval(base[key][sub_key])
        except:
            try:
                sub_val = float(base[key][sub_key])
            except:
                sub_val = None
        sub_list.append(sub_val)
        
    return sub_list


class AnalyzeVplanetModel(object):
    """High-level driver for VPLanet parameter studies loaded from a YAML config.

    Reads a YAML configuration file that specifies fixed input parameters,
    variable input parameters (with prior bounds), and output parameters
    (with optional observational data).  Wraps a :class:`VplanetModel`
    instance and exposes methods for running forward models, computing
    log-likelihoods, running MCMC, and performing global sensitivity analysis.

    Parameters
    ----------
    cfile : str
        Path to the YAML configuration file.  Must contain ``"input_fix"``,
        ``"input_var"``, ``"output"``, and ``"inpath"`` blocks.
    inpath : str, optional
        Override the VPLanet infile path from the YAML ``"inpath"`` key.
    outpath : str, optional
        Directory for all output files.  Defaults to ``"."``.
    verbose : bool, optional
        Passed through to :class:`VplanetModel`.  Defaults to ``True``.
    compute_true : bool, optional
        If ``True`` (default) run the model at the ``"true_value"``
        parameters to populate ``outparams.data`` (synthetic test mode).
    ncore : int, optional
        Number of CPU cores for parallel model evaluation.
        Defaults to ``multiprocessing.cpu_count()``.
    vpm_kwargs : dict, optional
        Additional keyword arguments forwarded to :class:`VplanetModel`.

    Attributes
    ----------
    vpm : VplanetModel
        The underlying VPLanet model instance.
    inparams_fix : VplanetParameters
        Fixed input parameters.
    inparams_var : VplanetParameters
        Variable input parameters (those varied during inference).
    inparams_all : VplanetParameters
        Union of fixed and variable input parameters.
    outparams : VplanetParameters
        Output parameters with observational data and uncertainties.
    like_data : np.ndarray
        Observational data array used by :meth:`lnlike`, shape
        ``(n_outparams, 2)`` with columns ``[mean, std]``.
    """

    def __init__(self,
                 cfile,
                 inpath=None,
                 outpath=".",
                 verbose=True,
                 compute_true=True,
                 ncore=mp.cpu_count(),
                 vpm_kwargs={},
                 **kwargs):

        # number of CPU cores for parallelization
        self.ncore = ncore
        self.cfile = cfile
        self.config_id = self.cfile.split("/")[-1].split(".yaml")[0].strip("/")

        # directory to save results (defaults to local)
        self.outpath = outpath
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        # load YAML config file
        with open(self.cfile) as f:
            data = yaml.load(f, Loader=SafeLoader)

        inparams_fix = VplanetParameters(names=list(data["input_fix"].keys()),
                                             units=sort_yaml_key(data["input_fix"], "units"),
                                             true=sort_yaml_key(data["input_fix"], "true_value"),
                                             labels=sort_yaml_key(data["input_fix"], "label"))

        inparams_var = VplanetParameters(names=list(data["input_var"].keys()),
                                             units=sort_yaml_key(data["input_var"], "units"),
                                             true=sort_yaml_key(data["input_var"], "true_value"),
                                             bounds=sort_yaml_key(data["input_var"], "bounds"),
                                             data=sort_yaml_key(data["input_var"], "data"),
                                             labels=sort_yaml_key(data["input_var"], "label"))

        inparams_all = VplanetParameters(names=inparams_fix.names + inparams_var.names,
                                             units=inparams_fix.units + inparams_var.units,
                                             true=inparams_fix.true + inparams_var.true)

        outparams = VplanetParameters(names=list(data["output"].keys()),
                                          units=sort_yaml_key(data["output"], "units"),
                                          data=sort_yaml_key(data["output"], "data"),
                                          uncertainty=sort_yaml_key(data["output"], "uncertainty"),
                                          labels=sort_yaml_key(data["output"], "label"))

        # initialize vplanet model
        if inpath is None:
            self.inpath = data["inpath"]
        else:
            self.inpath = inpath

        self.vpm = VplanetModel(inparams_all.dict_units, 
                                inpath=self.inpath, 
                                outparams=outparams.dict_units, 
                                verbose=verbose,
                                **vpm_kwargs)

        # if this is a synthetic model test, run vplanet model on true parameters
        if compute_true == True:
            output_true = self.vpm.run_model(inparams_all.true)
            outparams.set_data(output_true)
            self.like_data = outparams.data

        self.inparams_fix = inparams_fix
        self.inparams_var = inparams_var
        self.inparams_all = inparams_all
        self.outparams = outparams


    def format_theta(self, theta_var):
        """Merge variable parameters into the full parameter vector.

        Starts from the true/fiducial values of all parameters and
        substitutes the provided variable values.

        Parameters
        ----------
        theta_var : array-like of float
            Values for the variable parameters only, in the order of
            ``self.inparams_var.names``.

        Returns
        -------
        np.ndarray
            Full parameter vector of length ``len(self.inparams_all.names)``.
        """
        # defaults to running true values
        theta_run_dict = copy.copy(self.inparams_all.dict_true)
        
        # substitute variable parameters
        theta_var_dict = dict(zip(self.inparams_var.names, theta_var))

        for key in self.inparams_var.names:
            theta_run_dict[key] = theta_var_dict[key]

        theta_run = np.array(list(theta_run_dict.values()))

        return theta_run


    def run_model_format(self, theta_var, **kwargs):
        """Run the model for a vector of variable parameters.

        Internally calls :meth:`format_theta` to build the full parameter
        vector and then delegates to :meth:`VplanetModel.run_model`.

        Parameters
        ----------
        theta_var : array-like of float
            Variable parameter values.

        Returns
        -------
        np.ndarray
            Model output values; see :meth:`VplanetModel.run_model`.
        """
        # format fixed theta + variable theta
        theta_run = self.format_theta(theta_var)

        # run model
        output = self.vpm.run_model(theta_run, remove=True)

        return output


    def run_models(self, theta_var_array, **kwargs):
        """Run the model for an array of variable parameter vectors.

        Uses ``multiprocessing.Pool`` when ``self.ncore > 1``.

        Parameters
        ----------
        theta_var_array : np.ndarray, shape (n_samples, n_var_params)
            Each row is one set of variable parameter values.

        Returns
        -------
        np.ndarray, shape (n_samples, n_outparams)
            Model outputs for every sample.
        """
        if self.ncore <= 1:
            outputs = np.zeros(theta_var_array.shape[0])
            for ii, tt in tqdm.tqdm(enumerate(theta_var_array)):
                outputs[ii] = self.run_model_format(tt, **kwargs)
        else:
            with mp.Pool(self.ncore) as p:
                outputs = []
                for result in tqdm.tqdm(p.imap(func=self.run_model_format, iterable=theta_var_array), total=len(theta_var_array)):
                    outputs.append(result)
                outputs = np.array(outputs)

        return outputs


    def lnlike(self, theta_var):
        """Gaussian log-likelihood for the variable parameters.

        Computes :math:`\\ln \\mathcal{L} = -\\tfrac{1}{2}\\sum_i
        \\left(\\tfrac{m_i - d_i}{\\sigma_i}\\right)^2` where :math:`m_i` are
        the model outputs and :math:`d_i, \\sigma_i` come from
        ``self.outparams.data``.

        Parameters
        ----------
        theta_var : array-like of float
            Variable parameter values.

        Returns
        -------
        float
            Log-likelihood value.
        """
        output = self.run_model_format(theta_var)

        # compute log likelihood
        lnl = -0.5 * np.sum(((output - self.outparams.data.T[0])/self.outparams.data.T[1])**2)

        return lnl


    def run_mcmc(self, method="dynesty",
                       reload=False,
                       kernel="ExpSquaredKernel",
                       ntrain=1000,
                       ntest=100,
                       niter=500):
        """Run Bayesian inference using the specified MCMC method.

        Parameters
        ----------
        method : {"dynesty", "alabi", "alabi_dynesty"}, optional
            Inference backend.

            ``"dynesty"``
                Direct nested sampling with ``dynesty`` on the true forward
                model (no surrogate).
            ``"alabi"``
                Train a Gaussian Process surrogate with ``alabi``'s BAPE
                active-learning algorithm, then stop (no MCMC run yet).
            ``"alabi_dynesty"``
                Train the GP surrogate, then run ``dynesty`` on it.

        reload : bool, optional
            If ``True``, load a previously saved ``alabi`` cache from the
            results directory instead of retraining.  Defaults to ``False``.
        kernel : str, optional
            GP kernel name passed to ``alabi.SurrogateModel.init_gp``.
            Defaults to ``"ExpSquaredKernel"``.
        ntrain : int, optional
            Initial training-set size for the GP surrogate.  Defaults to
            ``1000``.
        ntest : int, optional
            Test-set size for GP evaluation.  Defaults to ``100``.
        niter : int, optional
            Number of active-learning iterations.  Defaults to ``500``.

        Raises
        ------
        Exception
            If ``self.like_data`` is ``None`` or ``alabi`` is not installed.
        """

        if self.like_data is None:
            raise Exception("No likelihood data specified.")

        try:
            import alabi
        except:
            raise Exception("Dependency 'alabi' not installed. To install alabi run: \n\n" + 
                            "git clone https://github.com/jbirky/alabi \n" +
                            "cd alabi \n" +
                            "python setup.py install")

        if method == "dynesty":
            savedir = os.path.join(self.outpath, "results_dynesty/", self.config_id)

        elif method == "alabi":
            savedir = os.path.join(self.outpath, "results_alabi/", self.config_id)

        elif method == "alabi_dynesty":
            savedir = os.path.join(self.outpath, "results_alabi/", self.config_id)

        # set up prior for dynesty
        self.ptform = partial(alabi.utility.prior_transform_normal, 
                              bounds=self.inparams_var.bounds, 
                              data=self.inparams_var.data)
        
        # Configure MCMC
        if reload == True:
            sm = alabi.load_model_cache(savedir)
        else:
            prior_sampler = partial(alabi.utility.prior_sampler, bounds=self.inparams_var.bounds, sampler='uniform')
            sm = alabi.SurrogateModel(fn=self.lnlike, bounds=self.inparams_var.bounds, prior_sampler=prior_sampler,
                                    savedir=savedir, labels=self.inparams_var.labels)
            sm.init_samples(ntrain=ntrain, ntest=ntest, reload=False)
            sm.init_gp(kernel=kernel, fit_amp=False, fit_mean=True, white_noise=-15)
        
        # Run MCMC
        if method == "dynesty":
            sm.run_dynesty(like_fn="true", ptform=self.ptform, mode="static", multi_proc=True, save_iter=100)
            sm.plot(plots=["dynesty_all"])

        elif method == "alabi":
            sm.active_train(niter=niter, algorithm="bape", gp_opt_freq=10, save_progress=True)
            sm.plot(plots=["gp_all"])

        elif method == "alabi_dynesty":
            sm.active_train(niter=niter, algorithm="bape", gp_opt_freq=10, save_progress=True)
            sm.plot(plots=["gp_all"])

            sm.run_dynesty(ptform=self.ptform, mode="static", multi_proc=True, save_iter=100)
            sm.plot(plots=["dynesty_all"])

        return None


    def plot_sensitivity_table(self, table):
        """Plot a Sobol sensitivity index as a labelled heatmap.

        Parameters
        ----------
        table : pandas.DataFrame
            Sensitivity table with input parameters as the index and output
            parameters as columns (values in [0, 1]).

        Returns
        -------
        matplotlib.figure.Figure
        """
        # plot sensitivity tables
        fig = plt.figure(figsize=[12,8])
        sn.heatmap(table, yticklabels=self.inparams_var.labels, xticklabels=self.outparams.labels, 
                   annot=True, annot_kws={"size": 18}, vmin=0, vmax=1, cmap="bone") 
        plt.title("First order sensitivity (S1) index", fontsize=25)
        plt.xticks(rotation=45, fontsize=18, ha='right')
        plt.yticks(rotation=0)
        plt.yticks(fontsize=18)
        plt.xlabel("Final Conditions", fontsize=22)
        plt.ylabel("Initial Conditions", fontsize=22)
        plt.close()

        return fig


    def variance_global_sensitivity(self, param_values=None, Y=None, nsample=1024, save=False, subpath="results_sensitivity"):
        """Run Sobol variance-based global sensitivity analysis.

        Uses the SALib library to sample the parameter space (Saltelli scheme)
        and compute first-order (S1) and total-effect (ST) Sobol indices for
        every output parameter.

        Parameters
        ----------
        param_values : np.ndarray, optional
            Pre-computed Saltelli sample array of shape
            ``(nsample * (n_var_params + 2), n_var_params)``.  If ``None``
            (default) new samples are drawn.
        Y : np.ndarray, optional
            Pre-computed model outputs of shape
            ``(n_samples, n_outparams)``.  If ``None`` (default) the model is
            run for all ``param_values``.
        nsample : int, optional
            Base sample size *N* for the Saltelli sampler.  The total number of
            model runs is ``N * (n_var_params + 2)``.  Defaults to ``1024``.
        save : bool, optional
            If ``True``, save samples, sensitivity tables, and figures to
            ``self.outpath / subpath / config_id /``.  Defaults to ``False``.
        subpath : str, optional
            Subdirectory within ``self.outpath`` for saved results.
            Defaults to ``"results_sensitivity"``.

        Returns
        -------
        None
            Results are stored as ``self.table_s1``, ``self.table_sT``,
            ``self.fig_s1``, and ``self.fig_sT``.
        """

        from SALib.sample import saltelli
        from SALib.analyze import sobol

        problem = {
            "num_vars": self.inparams_var.num,
            "names": self.inparams_var.names,
            "bounds": self.inparams_var.bounds
        }

        if param_values is None:
            param_values = saltelli.sample(problem, nsample)

        if Y is None:
            Y = self.run_models(param_values)

        # save samples to npz file
        savedir = os.path.join(self.outpath, subpath, self.config_id)
        if not os.path.exists(savedir):
            os.makedirs(savedir)    
        if save == True:       
            np.savez(f"{savedir}/var_global_sensitivity_sample.npz", param_values=param_values, Y=Y)

        dict_s1 = {"input": self.inparams_var.names}
        dict_sT = {"input": self.inparams_var.names}
                        
        for ii in range(Y.shape[1]):
            res = sobol.analyze(problem, Y.T[ii])
            dict_s1[self.outparams.names[ii]] = res['S1']
            dict_sT[self.outparams.names[ii]] = res['ST']
            
        table_s1 = pd.DataFrame(data=dict_s1).round(2)
        table_s1 = table_s1.set_index("input").rename_axis(None, axis=0)
        table_s1[table_s1.values <= 0] = 0
        table_s1[table_s1.values > 1] = 1
        self.table_s1 = table_s1
        if save == True:
            table_s1.to_csv(f"{savedir}/sensitivity_table_s1.csv")

        table_sT = pd.DataFrame(data=dict_sT).round(2)
        table_sT = table_sT.set_index("input").rename_axis(None, axis=0)
        table_sT[table_sT.values <= 0] = 0
        table_sT[table_sT.values > 1] = 1
        self.table_sT = table_sT
        if save == True:
            table_sT.to_csv(f"{savedir}/sensitivity_table_sT.csv")

        self.fig_s1 = self.plot_sensitivity_table(self.table_s1)
        self.fig_sT = self.plot_sensitivity_table(self.table_sT)
        if save == True:
            self.fig_s1.savefig(f"{savedir}/sensitivity_table_s1.png", bbox_inches="tight")
            self.fig_sT.savefig(f"{savedir}/sensitivity_table_sT.png", bbox_inches="tight")