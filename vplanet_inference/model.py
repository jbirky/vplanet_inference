import vplot as vpl 
import vplanet
import numpy as np 
import os
import re
import subprocess
import shutil
import time
import random

__all__ = ["VplanetModel"]


class VplanetModel(object):

    def __init__(self, params, inpath=".", vplfile="vpl.in", sys_name="system", 
                 factor=None, infile_list=None, verbose=True):
        """
        params  : (str, list) variable parameter names
                  ['vpl.dStopTime', 'star.dRotPeriod', 'star.dMass', 'planet.dEcc', 'planet.dOrbPeriod']
                
        inpath  : (str) path to template infiles
                  'infiles/'

        factor  : (float, list) theta conversion factor
        """

        self.params = params
        self.inpath = inpath
        self.vplfile = vplfile
        self.sys_name = sys_name

        self.nparam = len(params)

        if factor is None:
            self.factor = np.ones(self.nparam)
        else:
            self.factor = factor
        
        if infile_list is None:
            self.infile_list = os.listdir(self.inpath)
        else:
            self.infile_list = infile_list

        self.verbose = verbose


    def initialize_model(self, theta, outpath=None):
        """
        theta   : (float, list) parameter values, corresponding to self.param

        outpath : (str) path to where model infiles should be written
            'output/'
        """
        
        # Apply unit conversions to theta
        theta = np.array(theta) * self.factor 

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        param_file_all = np.array([x.split('.')[0] for x in self.params])  # e.g. ['vpl', 'primary', 'primary', 'secondary', 'secondary']
        param_name_all = np.array([x.split('.')[1] for x in self.params])  # e.g. ['dStopTime', 'dRotPeriod', 'dMass', 'dRotPeriod', 'dMass']

        for file in self.infile_list: # vpl.in, primary.in, secondary.in
            with open(os.path.join(self.inpath, file), 'r') as f:
                file_in = f.read()
                
            ind = np.where(param_file_all == file.strip('.in'))[0]
            theta_file = theta[ind]
            param_name_file = param_name_all[ind]
            
            if file == 'vpl.in':
                file_in = re.sub("%s(.*?)#" % "sSystemName", "%s %s #" % ("sSystemName", self.sys_name), file_in)
                
            for i in range(len(theta_file)):
                file_in = re.sub("%s(.*?)#" % param_name_file[i], "%s %.6e #" % (param_name_file[i], theta_file[i]), file_in)

            write_file = os.path.join(outpath, file)
            with open(write_file, 'w') as f:
                print(file_in, file = f)

            if self.verbose:
                print(f"Created file {write_file}")


    def get_outparam(self, output, outparams, **kwargs):
        """
        output    : (vplot object) results form a model run obtained using vplot.GetOutput()

        outparams : (str, list) return specified list of parameters from log file
                    ['initial.primary.Luminosity', 'final.primary.Radius', 'initial.secondary.Luminosity', 'final.secondary.Radius']
        """
        nout = len(outparams)
        outvalues = np.zeros(nout)

        for i in range(nout):
            base = kwargs.get('base', output.log)
            for attr in outparams[i].split('.'):
                base = getattr(base, attr)
            outvalues[i] = float(base.value)

        return outvalues


    def run_model(self, theta, remove=True, outparams=None, outpath=None):
        """
        theta     : (float, list) parameter values, corresponding to self.params

        remove    : (bool) True will erase input/output files after model is run

        outparams : (str, list) return specified list of parameters from log file
                    ['initial.primary.Luminosity', 'final.primary.Radius', 'initial.secondary.Luminosity', 'final.secondary.Radius']
        """

        # randomize output directory
        if outpath is None:
            while True:
                # create a unique randomized path to execute the model files
                outpath = f"output/{random.randrange(16**12)}"
                if not os.path.exists(outpath):
                    break

        self.initialize_model(theta, outpath=outpath)
        
        t0 = time.time()

        # Execute the model!
        subprocess.call(["vplanet vpl.in"], cwd=outpath, shell=True)

        # if no logfile is found, it's probably because there was something wrong with the infile formatting
        output = vplanet.get_output(outpath)

        if self.verbose == True:
            print('Executed model %svpl.in %.3f s'%(outpath, time.time() - t0))

        if outparams is not None:
            self.outparams = outparams 
            outvalues = self.get_outparam(output, self.outparams)
            model_out = outvalues
        else:
            model_out = output

        if self.verbose:
            print(f"theta : {theta} \t y : {model_out}")

        if remove == True:
            shutil.rmtree(outpath)

        return model_out


    def initialize_bayes(self, data=None, outparams=None, bounds=None):
        """
        data      : (float, matrix)
                    [(rad, radSig), 
                     ...
                     (lum, lumSig)]

        outparams : (str, list) return specified list of parameters from log file
                    ['final.primary.Radius', ..., 'final.primary.Luminosity']
        """
        if data is not None:
            self.data = data 
        else:
            print('Must input data!')
            raise 
        
        if outparams is not None:
            self.outparams = outparams
            self.nout = len(outparams)
        else:
            print('Must specify outparams!')
            raise 

        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = np.empty(shape=(self.nparams, 2), dtype='object') 

    
    def lnlike(self, theta, outpath=None):
        """
        Gaussian likelihood function comparing vplanet model and given observational values/uncertainties
        """

        ymodel = self.run_model(theta, outparams=self.outparams, outpath=outpath)

        # Gaussian likelihood 
        lnlike = -0.5 * np.sum(((ymodel - self.data.T[0])/self.data.T[1])**2)

        return lnlike