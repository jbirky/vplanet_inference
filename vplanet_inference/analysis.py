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
except:
    rc('text', usetex=False)


import vplanet_inference as vpi


__all__ = ["AnalyzeVplanetModel"]


def sort_yaml_key(base, sub_key):
    
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
    
    def __init__(self, 
                 cfile, 
                 verbose=True, 
                 compute_true=True,
                 ncore=mp.cpu_count(),
                 **kwargs):

        """
        Class for analyzing VPLANET models (run parameter sweeps, MCMC, sensitivity analysis, etc.) 
        """

        # number of CPU cores for parallelization
        self.ncore = ncore
        self.cfile = cfile
        self.config_id = self.cfile.split("/")[-1].split(".yaml")[0].strip("/")

        # directory to save results (defaults to local)
        self.outpath = kwargs.get("outpath", ".")
        if not os.path.exists(self.outpath):
            os.makedirs(self.outpath)

        # load YAML config file
        with open(self.cfile) as f:
            data = yaml.load(f, Loader=SafeLoader)

        inparams_fix = vpi.VplanetParameters(names=list(data["input_fix"].keys()),
                                             units=sort_yaml_key(data["input_fix"], "units"),
                                             true=sort_yaml_key(data["input_fix"], "true_value"),
                                             labels=sort_yaml_key(data["input_fix"], "label"))

        inparams_var = vpi.VplanetParameters(names=list(data["input_var"].keys()),
                                             units=sort_yaml_key(data["input_var"], "units"),
                                             true=sort_yaml_key(data["input_var"], "true_value"),
                                             bounds=sort_yaml_key(data["input_var"], "bounds"),
                                             data=sort_yaml_key(data["input_var"], "data"),
                                             labels=sort_yaml_key(data["input_var"], "label"))

        inparams_all = vpi.VplanetParameters(names=inparams_fix.names + inparams_var.names,
                                             units=inparams_fix.units + inparams_var.units,
                                             true=inparams_fix.true + inparams_var.true)

        outparams = vpi.VplanetParameters(names=list(data["output"].keys()),
                                          units=sort_yaml_key(data["output"], "units"),
                                          data=sort_yaml_key(data["output"], "data"),
                                          uncertainty=sort_yaml_key(data["output"], "uncertainty"))

        # initialize vplanet model
        self.inpath = kwargs.get("inpath", data["inpath"])
        self.vpm = vpi.VplanetModel(inparams_all.dict_units, inpath=self.inpath, outparams=outparams.dict_units, verbose=verbose)

        # if this is a synthetic model test, run vplanet model on true parameters
        if (outparams.data is None) & (compute_true == True):
            outparams.data = self.vpm.run_model(inparams_all.true)

        self.inparams_fix = inparams_fix
        self.inparams_var = inparams_var
        self.inparams_all = inparams_all
        self.outparams = outparams


    def format_theta(self, theta_var):

        theta_run_dict = copy.copy(self.inparams_all.dict_true)
        theta_var_dict = dict(zip(self.inparams_var.names, theta_var))

        for key in self.inparams_var.names:
            theta_run_dict[key] = theta_var_dict[key]

        theta_run = np.array(list(theta_run_dict.values()))

        return theta_run


    def run_model_format(self, theta_var):

        # format fixed theta + variable theta
        theta_run = self.format_theta(theta_var)

        # run model
        output = self.vpm.run_model(theta_run, remove=True)

        return output


    def run_models(self, theta_var_array):

        if self.ncore <= 1:
            outputs = np.zeros(theta_var_array.shape[0])
            for ii, tt in tqdm.tqdm(enumerate(theta_var_array)):
                outputs[ii] = self.run_model_format(tt)
        else:
            with mp.Pool(self.ncore) as p:
                outputs = []
                for result in tqdm.tqdm(p.imap(func=self.run_model_format, iterable=theta_var_array), total=len(theta_var_array)):
                    outputs.append(result)
                outputs = np.array(outputs)

        return outputs


    def lnlike(self, theta_var):

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

        # set up prior for dynesty
        self.ptform = partial(alabi.utility.prior_transform_normal, bounds=self.bounds, data=self.prior_data)

        # Configure MCMC
        if reload == True:
            sm = alabi.load_model_cache(savedir)
        else:
            prior_sampler = partial(alabi.utility.prior_sampler, bounds=self.bounds, sampler='uniform')
            sm = alabi.SurrogateModel(fn=self.lnlike, bounds=self.bounds, prior_sampler=prior_sampler,
                                    savedir=savedir, labels=self.labels)
            sm.init_samples(ntrain=ntrain, ntest=ntest, reload=False)
            sm.init_gp(kernel=kernel, fit_amp=False, fit_mean=True, white_noise=-15)

        # Run MCMC
        if method == "dynesty":
            sm.run_dynesty(like_fn="true", ptform=self.ptform, mode="static", multi_proc=True, save_iter=100)
            sm.plot(plots=["dynesty_all"])

        elif method == "alabi":
            sm.active_train(niter=niter, algorithm="bape", gp_opt_freq=10, save_progress=True)
            sm.plot(plots=["gp_all"])
            sm.run_dynesty(ptform=self.ptform, mode="static", multi_proc=True, save_iter=100)
            sm.plot(plots=["dynesty_all"])

        return None


    def plot_sensitivity_table(self, table):

        # plot sensitivity tables
        fig = plt.figure(figsize=[12,8])
        sn.heatmap(table, annot=True, annot_kws={"size": 18}, vmin=0, vmax=1, cmap="bone") 
        plt.title("First order sensitivity (S1) index", fontsize=25)
        plt.xticks(rotation=45, fontsize=18, ha='right')
        plt.yticks(rotation=0)
        plt.yticks(fontsize=18)
        plt.xlabel("Final Conditions", fontsize=22)
        plt.ylabel("Initial Conditions", fontsize=22)
        plt.close()

        return fig


    def variance_global_sensitivity(self, param_values=None, Y=None, nsample=1024):

        from SALib.sample import saltelli
        from SALib.analyze import sobol

        problem = {
            'num_vars': self.inparams_var.num,
            'names': self.inparams_var.labels,
            'bounds': self.inparams_var.bounds
        }

        if param_values is None:
            param_values = saltelli.sample(problem, nsample)

        if Y is None:
            Y = self.run_models(param_values)

        # save samples to npz file
        savedir = os.path.join(self.outpath, "results_sensitivity", self.config_id)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.savez(f"{savedir}/var_global_sensitivity_sample.npz", param_values=param_values, Y=Y)

        dict_s1 = {'input': self.inparams_var.labels}
        dict_sT = {'input': self.inparams_var.labels}
                        
        for ii in range(Y.shape[1]):
            res = sobol.analyze(problem, Y.T[ii])
            dict_s1[self.outparams.labels[ii]] = res['S1']
            dict_sT[self.outparams.labels[ii]] = res['ST']
            
        table_s1 = pd.DataFrame(data=dict_s1)
        table_s1 = table_s1.set_index("input").rename_axis(None, axis=0).round(2)
        table_s1[table_s1.values <= 0] = 0
        table_s1[table_s1.values > 1] = 1
        self.table_s1 = table_s1
        table_s1.to_csv(f"{savedir}/sensitivity_table_s1.csv")

        table_sT = pd.DataFrame(data=dict_sT)
        table_sT = table_sT.set_index("input").rename_axis(None, axis=0).round(2)
        table_sT[table_sT.values <= 0] = 0
        table_sT[table_sT.values > 1] = 1
        self.table_sT = table_sT
        table_sT.to_csv(f"{savedir}/sensitivity_table_sT.csv")

        self.fig_s1 = self.plot_sensitivity_table(table_s1)
        self.fig_s1.savefig(f"{savedir}/sensitivity_table_s1.png", bbox_inches="tight")

        self.fig_sT = self.plot_sensitivity_table(table_sT)
        self.fig_sT.savefig(f"{savedir}/sensitivity_table_sT.png", bbox_inches="tight")