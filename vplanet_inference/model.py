import vplot as vpl 
import vplanet
import numpy as np 
import os
import re
import subprocess
import shutil
import time
import random
import astropy.units as u

__all__ = ["VplanetModel"]


class VplanetModel(object):

    def __init__(self, inparams, inpath=".", outparams=None, outpath="output/", fixsub=None,
                 vplfile="vpl.in", sys_name="system", verbose=True):
        """
        params  : (str, list) variable parameter names
                  ['vpl.dStopTime', 'star.dRotPeriod', 'star.dMass', 'planet.dEcc', 'planet.dOrbPeriod']
                
        inpath  : (str) path to template infiles
                  'infiles/'

        outparams : (str, list) return specified list of parameters from log file
                    ['final.primary.Radius', 'final.secondary.Radius']
        """

        # Input parameters
        self.inparams = list(inparams.keys())
        self.in_units = list(inparams.values())
        self.inpath = inpath
        self.vplfile = vplfile
        self.sys_name = sys_name
        self.ninparam = len(inparams)
        self.verbose = verbose

        # Output parameters
        if outparams is not None:
            self.outparams = list(outparams.keys())
            self.out_units = list(outparams.values())
            self.noutparam = len(self.outparams)
        self.outpath_base = outpath

        # List of infiles (vpl.in + body files)
        self.infile_list = [vplfile]
        with open(os.path.join(inpath, vplfile), 'r') as vplf:
            for readline in vplf.readlines():
                line = readline.strip().split()
                if len(line) > 1:
                    if line[0] == "saBodyFiles":
                        for ll in line:
                            if ".in" in ll:
                                self.infile_list.append(ll)
        
        # Fixed parameter substitutions - to be implemented!
        if fixsub is not None:
            self.fixparam = list(fixsub.keys())
            self.fixvalue = list(fixsub.values())


    def initialize_model(self, theta, outpath=None):
        """
        theta   : (float, list) parameter values, corresponding to self.param

        outpath : (str) path to where model infiles should be written
            'output/'
        """
        
        # Convert units of theta to SI
        theta_conv = np.zeros(self.ninparam)
        theta_new_unit = []

        for ii in range(self.ninparam):
            if isinstance(self.in_units[ii], u.function.logarithmic.DexUnit):
                # un-log units
                new_theta = (theta[ii] * self.in_units[ii]).physical.si
            else:
                new_theta = (theta[ii] * self.in_units[ii]).si
            theta_conv[ii] = new_theta.value
            theta_new_unit.append(new_theta.unit)

        if self.verbose:
            print("\nInput:")
            print("-----------------")
            for ii in range(self.ninparam):
                print("%s : %s %s (user)   --->   %s %s (vpl file)"%(self.inparams[ii],
                    theta[ii], self.in_units[ii], theta_conv[ii], theta_new_unit[ii]))
            print("")

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        param_file_all = np.array([x.split('.')[0] for x in self.inparams])  # e.g. ['vpl', 'primary', 'primary', 'secondary', 'secondary']
        param_name_all = np.array([x.split('.')[1] for x in self.inparams])  # e.g. ['dStopTime', 'dRotPeriod', 'dMass', 'dRotPeriod', 'dMass']

        for file in self.infile_list: # vpl.in, primary.in, secondary.in
            with open(os.path.join(self.inpath, file), 'r') as f:
                file_in = f.read()
                
            ind = np.where(param_file_all == file.strip('.in'))[0]
            theta_file = theta_conv[ind]
            param_name_file = param_name_all[ind]
            
            # if VPL file
            if file == 'vpl.in':
                file_in = re.sub("%s(.*?)#" % "sSystemName", "%s %s #" % ("sSystemName", self.sys_name), file_in)

                # Set units to SI
                file_in = re.sub("%s(.*?)#" % "sUnitMass", "%s %s #" % ("sUnitMass", "kg"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitLength", "%s %s #" % ("sUnitLength", "m"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitTime", "%s %s #" % ("sUnitTime", "sec"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitAngle", "%s %s #" % ("sUnitAngle", "rad"), file_in)
                file_in = re.sub("%s(.*?)#" % "sUnitTemp", "%s %s #" % ("sUnitTemp", "K"), file_in)
                
            # iterate over all input parameters, and substitute parameters in appropriate files
            for i in range(len(theta_file)):
                file_in = re.sub("%s(.*?)#" % param_name_file[i], "%s %.10e #" % (param_name_file[i], theta_file[i]), file_in)

                # Set output timesteps to simulation stop time
                if param_name_file[i] == 'dStopTime':
                    file_in = re.sub("%s(.*?)#" % "dOutputTime", "%s %.10e #" % ("dOutputTime", theta_file[i]), file_in)

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
        outvalues = np.zeros(self.noutparam)

        for i in range(self.noutparam):
            base = kwargs.get('base', output.log)
            for attr in outparams[i].split('.'):
                base = getattr(base, attr)
            outvalues[i] = base.to(self.out_units[i]).value

        return outvalues


    def run_model(self, theta, remove=True, outsubpath=None):
        """
        theta     : (float, list) parameter values, corresponding to self.inparams

        remove    : (bool) True will erase input/output files after model is run
        """

        # randomize output directory
        if outsubpath is None:
            # create a unique randomized path to execute the model files
            outsubpath = random.randrange(16**12)
        outpath = f"{self.outpath_base}/{outsubpath}"

        self.initialize_model(theta, outpath=outpath)
        
        t0 = time.time()

        # Execute the model!
        subprocess.call(["vplanet vpl.in"], cwd=outpath, shell=True)

        # if no logfile is found, it's probably because there was something wrong with the infile formatting
        output = vplanet.get_output(outpath)

        if self.verbose == True:
            print('Executed model %svpl.in %.3f s'%(outpath, time.time() - t0))

        if self.outparams is None:
            model_out = output
        else:
            outvalues = self.get_outparam(output, self.outparams)
            model_out = outvalues

            if self.verbose:
                print("\nOutput:")
                print("-----------------")
                for ii in range(self.noutparam):
                    print("%s : %s %s"%(self.outparams[ii], model_out[ii], self.out_units[ii]))
                print("")

        if remove == True:
            shutil.rmtree(outpath)

        return model_out


    def initialize_bayes(self, data=None, bounds=None):
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

        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = np.empty(shape=(self.ninparams, 2), dtype='object') 

    
    def lnlike(self, theta, outsubpath=None):
        """
        Gaussian likelihood function comparing vplanet model and given observational values/uncertainties
        """

        ymodel = self.run_model(theta, outsubpath=outsubpath)

        # Gaussian likelihood 
        lnlike = -0.5 * np.sum(((ymodel - self.data.T[0])/self.data.T[1])**2)

        return lnlike