import numpy as np 
import vplot as vpl 
import os
import re
import subprocess
import time

__all__ = ["VplanetModel"]


class VplanetModel(object):

    def __init__(self, params, **kwargs):
        """
        param   : (str, list) variable parameter names
                  ['vpl.dStopTime', 'star.dRotPeriod', 'star.dMass', 'planet.dEcc', 'planet.dOrbPeriod']
                
        inpath  : (str) path to template infiles
                  'infiles/'

        outpath : (str) path to where model infiles should be written
                  'output/'

        factor  : (float, list) theta conversion factor
        """

        self.params = params
        self.inpath = kwargs.get('inpath', '.')
        self.vplfile = kwargs.get('vplfile', 'vpl.in')
        self.sys_name = kwargs.get('sys_name', 'system')

        self.nparam = len(params)
        self.factor = np.array(kwargs.get('factor', np.ones(nparam)))
        
        self.infile_list = kwargs.get('infile_list', os.listdir(self.inpath))


    def initialize_model(self, theta, **kwargs):
        """
        theta   : (float, list) parameter values, corresponding to self.param
        """
        
        # Apply unit conversions to theta
        theta = np.array(theta) * self.factor 

        self.outpath = kwargs.get('outpath', '.')

        if not os.path.exists(self.outpath):
            os.mkdir(self.outpath)
        
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

            with open(os.path.join(self.outpath, file), 'w') as f:
                print(file_in, file = f)


    def get_outparam(self, output, outparams):

        outparam_file_all = np.array([x.split('.')[0] for x in outparams]) 
        outparam_name_all = np.array([x.split('.')[1] for x in outparams])

        num_out = len(outparams)
        outvalues = np.zeros(num_out)

        for i in range(num_out):
            outvalues[i] = float(getattr(getattr(output.log.final, outparam_file_all[i]), outparam_name_all[i]))

        return outvalues


    def run_model(self, theta, remove=False, verbose=True, **kwargs):
        """
        theta   : (float, list) parameter values, corresponding to self.param
        remove  : (bool) True will erase input/output files after model is run
        """

        self.outpath = kwargs.get('outpath', '.')
        self.initialize_model(theta, **kwargs)
        
        t0 = time.time()
        subprocess.call(["vplanet vpl.in"], cwd=self.outpath, shell=True)

        if verbose == True:
            print('Executed model %svpl.in %.3f s'%(self.outpath, time.time() - t0))

        # if no logfile is found, it's probably because there was something wrong with the infile formatting
        output = vpl.GetOutput(self.outpath, logfile=self.sys_name+'.log')

        if 'outparams' in kwargs:
            outparams = kwargs['outparams']  # e.g. ['primary.Luminosity', 'primary.Radius', 'secondary.Luminosity', 'secondary.Radius']
            outvalues = self.get_outparam(output, outparams)
            return outvalues

        else:
            return output


    def run_model_batch(self, theta_list, remove=False, verbose=True, **kwargs):

        for i, tt in enumerate(theta_list):
            op = os.path.join(outpath, 'model%s/'%(i))
            vpm.run_model(tt, outpath=op)

        return None