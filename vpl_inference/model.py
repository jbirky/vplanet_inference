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

        factor  : (float, list) theta conversion factor
        """

        self.params = params
        self.inpath = kwargs.get('inpath', '.')
        self.vplfile = kwargs.get('vplfile', 'vpl.in')
        self.sys_name = kwargs.get('sys_name', 'system')

        self.nparam = len(params)
        self.factor = np.array(kwargs.get('factor', np.ones(self.nparam)))
        
        self.infile_list = kwargs.get('infile_list', os.listdir(self.inpath))


    def InitializeModel(self, theta, **kwargs):
        """
        theta   : (float, list) parameter values, corresponding to self.param

        outpath : (str) path to where model infiles should be written
            'output/'
        """
        
        # Apply unit conversions to theta
        self.factor = np.array(kwargs.get('factor', self.factor))
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


    def GetOutparam(self, output, outparams, **kwargs):
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
            outvalues[i] = float(base)

        return outvalues


    def RunModel(self, theta, remove=False, verbose=True, **kwargs):
        """
        theta     : (float, list) parameter values, corresponding to self.param

        remove    : (bool) True will erase input/output files after model is run

        outparams : (str, list) return specified list of parameters from log file
                    ['initial.primary.Luminosity', 'final.primary.Radius', 'initial.secondary.Luminosity', 'final.secondary.Radius']
        """

        self.InitializeModel(theta, **kwargs)
        
        t0 = time.time()
        subprocess.call(["vplanet vpl.in"], cwd=self.outpath, shell=True)

        if verbose == True:
            print('Executed model %svpl.in %.3f s'%(self.outpath, time.time() - t0))

        # if no logfile is found, it's probably because there was something wrong with the infile formatting
        output = vpl.GetOutput(self.outpath, logfile=self.sys_name+'.log')

        if 'outparams' in kwargs:
            self.outparams = kwargs['outparams']  
            outvalues = self.GetOutparam(output, self.outparams)
            return outvalues

        else:
            return output


    def RunModelBatch(self, theta_list, remove=False, verbose=True, **kwargs):

        # run models in parallel
        if 'ncore' in kwargs:
            ncore = kwargs['ncore']

            """
            to be implemented
            """

        # run models sequentially
        else:  
            for i, tt in enumerate(theta_list):
                op = os.path.join(outpath, 'model%s/'%(i))
                vpm.RunModel(tt, outpath=op)

        return None

    
    def LnLike(self, data, theta, outparams):
        """
        Gaussian likelihood function comparing vplanet model and given observational values/uncertainties

        data      : (float, matrix)
                    [(rad, radSig), 
                     ...
                     (lum, lumSig)]

        outparams : (str, list) return specified list of parameters from log file
                    ['final.primary.Radius', ..., 'final.primary.Luminosity']
        """

        ymodel = self.RunModel(theta, outparams=outparams)

        # Gaussian likelihood 
        lnlike = -0.5 * np.sum(((ymodel - data.T[0])/data.T[1])**2)

        return lnlike


    def LnPriorFlat(self, theta, bounds):

        theta = np.array(theta)
        nparam = theta.shape[0]

        if nparam != bounds.shape[0]:
            print('dim theta != dim bounds')
            raise

        for i in range(nparam):
            if (theta[i] < bounds[i][0]) or (theta[i] > bounds[i][1]):
                return -np.inf

        return 0


    def LnPriorSample(self):
        """
        prior format for approxposterior
        """

        return None


    def LnPriorTransform(self):
        """
        prior format for nested sampling
        """

        return None


    def PosteriorSweep(self):

        return None