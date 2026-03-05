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
    """Interface for running VPLanet forward models with unit-aware parameter substitution.

    Reads a set of VPLanet template input files, substitutes parameter values
    (converting from user-specified astropy units to SI), executes VPLanet, and
    returns the requested output quantities with optional unit conversion.

    Parameters
    ----------
    inparams : dict
        Ordered mapping of ``"body.dParamName"`` keys to astropy units.
        The units define how the corresponding element of ``theta`` is
        interpreted when passed to :meth:`run_model`.  Use
        ``u.dex(u.dimensionless_unscaled)`` for log-scale inputs and
        ``-u.dimensionless_unscaled`` for parameters stored with a sign
        flip in the VPLanet infile.
        Example::

            {"star.dMass": u.Msun, "star.dRotPeriod": u.day, "vpl.dStopTime": u.Gyr}

    inpath : str, optional
        Path to the directory containing template ``.in`` files.
        Defaults to the current working directory ``"."``.
    outparams : dict, optional
        Ordered mapping of ``"final.body.ParamName"`` (or
        ``"initial.body.ParamName"``) keys to astropy units.  Each key
        selects one scalar value from the VPLanet log file; each unit
        controls the conversion applied before returning.
        Example::

            {"final.star.Luminosity": u.Lsun, "final.star.RotPer": u.day}

    outpath : str, optional
        Base directory for per-run output subdirectories.
        Defaults to ``"output/"``.
    fixsub : dict, optional
        Fixed parameter substitutions applied to every run (not yet fully
        implemented).
    executable : str, optional
        Name or path of the VPLanet binary.  Defaults to ``"vplanet"``.
    vplfile : str, optional
        Name of the primary VPLanet input file.  Defaults to ``"vpl.in"``.
    sys_name : str, optional
        Value substituted for ``sSystemName`` in the primary input file.
        Defaults to ``"system"``.
    timesteps : astropy.units.Quantity, optional
        If given, ``dOutputTime`` is set to this value and
        :meth:`run_model` returns a time-series dictionary instead of a
        flat array.  Must carry astropy time units (e.g.
        ``1e6 * u.yr``).
    time_init : astropy.units.Quantity, optional
        Initial simulation age written to ``dAge`` in every body file.
        Defaults to ``5e6 * u.yr``.
    forward : bool, optional
        If ``True`` (default) run a forward evolution (``bDoForward 1``).
        Set to ``False`` for backward integration.
    verbose : bool, optional
        Print substituted parameter values and model outputs after each
        run.  Defaults to ``True``.
    quiet : bool, optional
        Suppress VPLanet Python-binding log output.  Defaults to ``True``.
    debug : bool, optional
        Print per-parameter substitution details during
        :meth:`initialize_model`.  Defaults to ``False``.

    Examples
    --------
    >>> import vplanet_inference as vpi
    >>> import astropy.units as u
    >>> vpm = vpi.VplanetModel(
    ...     inparams={"star.dMass": u.Msun, "vpl.dStopTime": u.Gyr},
    ...     outparams={"final.star.Luminosity": u.Lsun},
    ...     inpath="infiles/stellar/",
    ...     outpath="output/",
    ...     verbose=False,
    ... )
    >>> result = vpm.run_model([0.09, 7.6])
    """

    def __init__(self,
                 inparams,
                 inpath=".",
                 outparams=None,
                 outpath="output/",
                 fixsub=None,
                 executable="vplanet",
                 vplfile="vpl.in",
                 sys_name="system",
                 timesteps=None,
                 time_init=5e6*u.yr,
                 forward=True,
                 verbose=True,
                 quiet=True,
                 debug=False):

        # Input parameters
        self.inparams = list(inparams.keys())
        self.in_units = list(inparams.values())
        self.inpath = inpath
        self.executable = executable
        self.vplfile = vplfile
        self.sys_name = sys_name
        self.ninparam = len(inparams)
        self.verbose = verbose
        self.quiet = quiet
        self.debug = debug
        
        if self.quiet:
            import logging
            logging.getLogger("vplanet").setLevel(logging.CRITICAL)

        # Output parameters - preserve user-specified order; keys() and values()
        # iterate in the same insertion order, so self.outparams[i] always
        # corresponds to self.out_units[i].
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

        # Set output timesteps (if specified, otherwise will default to same as dStopTime)
        if timesteps is not None:
            try:
                self.timesteps = timesteps.si.value
            except:
                raise ValueError("Units for timestep not valid.")
        else:
            self.timesteps = None

        # Run model foward (true) or backwards (false)?
        self.forward = forward

        # Set initial simulation time (dAge)
        try:
            self.time_init = time_init.si.value
        except:
            raise ValueError("Units for time_init not valid.")


    def initialize_model(self, theta, outpath=None):
        """Write VPLanet input files for a single parameter vector.

        Converts ``theta`` from user units to SI, substitutes values into the
        template ``.in`` files, forces SI unit declarations and ``sSystemName``,
        and writes the result to ``outpath``.

        Parameters
        ----------
        theta : array-like of float
            Parameter values in the units defined by ``inparams``, in the same
            order as ``self.inparams``.
        outpath : str
            Directory in which to write the substituted input files.  Created
            automatically if it does not exist.

        Raises
        ------
        ValueError
            If a parameter name listed in ``inparams`` is not found in the
            corresponding template file, or if the ``timesteps`` / ``time_init``
            units are invalid.
        """
        
        # Convert units of theta to SI
        theta_conv = np.zeros(self.ninparam)
        theta_new_unit = []

        for ii in range(self.ninparam):
            if self.in_units[ii] is None:
                theta_conv[ii] = theta[ii]
                theta_new_unit.append(None)
            elif isinstance(self.in_units[ii], u.function.logarithmic.DexUnit):
                # un-log units
                new_theta = (theta[ii] * self.in_units[ii]).physical.si
                theta_conv[ii] = new_theta.value
                theta_new_unit.append(new_theta.unit)
            else:
                new_theta = (theta[ii] * self.in_units[ii]).si
                theta_conv[ii] = new_theta.value
                theta_new_unit.append(new_theta.unit)

        if self.verbose:
            print("\nInput:")
            print("-----------------")
            for ii in range(self.ninparam):
                print("%s : %s [%s] (user)   --->   %s [%s] (vpl file)"%(self.inparams[ii],
                    theta[ii], self.in_units[ii], theta_conv[ii], theta_new_unit[ii]))
            print("")

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        
        # format list of input parameters
        param_file_all = np.array([x.split('.')[0] for x in self.inparams])  # e.g. ['vpl', 'primary', 'primary', 'secondary', 'secondary']
        param_name_all = np.array([x.split('.')[1] for x in self.inparams])  # e.g. ['dStopTime', 'dRotPeriod', 'dMass', 'dRotPeriod', 'dMass']

        # format list of output parameters
        key_split  = []
        unit_split = []
        for ii, key in enumerate(self.outparams):
            if "final" in key.split("."):
                key_split.append(key.split(".")[1:])
                unit_split.append([key.split(".")[1], self.out_units[ii]])
        out_name_split = np.array(key_split).T  # e.g. [['primary', 'secondary', 'secondary'], ['RotPer', 'RotPer', 'OrbPeriod']]
        out_unit_split = np.array(unit_split).T # e.g. [['primary', 'secondary', 'secondary'], [Unit("d"), Unit("d"), Unit("d")]]

        # save dictionary for retrieving output arrays
        out_body_name_dict = {key: [] for key in set(out_name_split[0])}
        out_body_unit_dict = {key: [] for key in set(out_unit_split[0])}
        for ii in range(out_name_split.shape[1]):
            out_body_name_dict[out_name_split[0][ii]].append(out_name_split[1][ii])
            out_body_unit_dict[out_unit_split[0][ii]].append(out_unit_split[1][ii])
        self.out_body_name_dict = out_body_name_dict  # e.g. {'secondary': ['RotPer', 'OrbPeriod'], 'primary': ['RotPer']}
        self.out_body_unit_dict = out_body_unit_dict  # e.g. {'secondary': [Unit("d"), Unit("d")], 'primary': [Unit("d")]}

        for file in self.infile_list: # vpl.in, primary.in, secondary.in
            with open(os.path.join(self.inpath, file), 'r') as f:
                file_in = f.read()

            # Strip inline comments (everything from # to end of line) so that
            # substitutions work regardless of whether template lines have # comments
            file_in = re.sub(r'[ \t]*#[^\n]*', '', file_in)

            ind = np.where(param_file_all == file.strip('.in'))[0]
            theta_file = theta_conv[ind]
            param_name_file = param_name_all[ind]

            # get saOutputOrder for evolution
            if file.strip('.in') in out_name_split[0]:
                output_order_vars = out_name_split[1][np.where(out_name_split[0] == file.strip('.in'))[0]]
                output_order_str = "Time " + " ".join(output_order_vars)

                # Set variables for tracking evolution (replace entire saOutputOrder line)
                file_in = re.sub(r"saOutputOrder[^\n]*", "saOutputOrder %s #" % output_order_str, file_in)

            # iterate over all input parameters, and substitute parameters in appropriate files
            for i in range(len(theta_file)):
                if self.debug:
                    print(f"Setting {param_name_file[i]} to {theta_file[i]} in file {file}")

                # Store original file content to check if substitution occurred
                original_file_in = file_in

                # Match "parameter value" with optional trailing comment (comments stripped above)
                pattern = r"%s\s+[^\s\n]+" % re.escape(param_name_file[i])
                replacement = "%s %.10e" % (param_name_file[i], theta_file[i])
                file_in = re.sub(pattern, replacement, file_in)

                # Check if the parameter was actually found and substituted
                if file_in == original_file_in:
                    raise ValueError(f"Parameter '{param_name_file[i]}' not found in file '{file}'. "
                                   f"Make sure the parameter exists in the template file.")

                # Set output timesteps to simulation stop time
                if param_name_file[i] == 'dStopTime':
                    file_in = re.sub(r"dOutputTime\s+\S+", "dOutputTime %.10e" % theta_file[i], file_in)

            # if VPL file
            if file == 'vpl.in':
                file_in = re.sub(r"sSystemName\s+\S+", "sSystemName %s #" % self.sys_name, file_in)

                # Set units to SI
                file_in = re.sub(r"sUnitMass\s+\S+", "sUnitMass kg #", file_in)
                file_in = re.sub(r"sUnitLength\s+\S+", "sUnitLength m #", file_in)
                file_in = re.sub(r"sUnitTime\s+\S+", "sUnitTime sec #", file_in)
                file_in = re.sub(r"sUnitAngle\s+\S+", "sUnitAngle rad #", file_in)
                file_in = re.sub(r"sUnitTemp\s+\S+", "sUnitTemp K #", file_in)

                # Set output timesteps (if specified, otherwise will default to same as dStopTime)
                if self.timesteps is not None:
                    file_in = re.sub(r"dOutputTime\s+\S+", "dOutputTime %.10e #" % self.timesteps, file_in)

                # Run evolution forward or backward
                if self.forward == True:
                    file_in = re.sub(r"bDoForward\s+\S+", "bDoForward 1 #", file_in)
                    file_in = re.sub(r"bDoBackward\s+\S+", "bDoBackward 0 #", file_in)
                else:
                    file_in = re.sub(r"bDoForward\s+\S+", "bDoForward 0 #", file_in)
                    file_in = re.sub(r"bDoBackward\s+\S+", "bDoBackward 1 #", file_in)

            else: # (not VPL file)
                # Set output timesteps (if specified, otherwise will default to same as dStopTime)
                if self.time_init is not None:
                    file_in = re.sub(r"dAge\s+\S+", "dAge %.10e #" % self.time_init, file_in)

            write_file = os.path.join(outpath, file)
            with open(write_file, 'w') as f:
                print(file_in, file = f)

            if self.verbose:
                print(f"Created file {write_file}")


    def get_outparam(self, output, **kwargs):
        """Extract scalar final-state output values from a VPLanet output object.

        Walks the attribute path of each key in ``self.outparams`` (e.g.
        ``"final.star.Luminosity"`` → ``output.log.final.star.Luminosity``),
        converts the result to the requested unit, and returns all values as a
        flat array.  Handles VPLanet modules that emit ``"(null)"`` as a unit
        string by falling back to the raw SI value.

        Parameters
        ----------
        output : vplanet output object
            Result returned by ``vplanet.get_output(outpath)``.

        Returns
        -------
        outvalues : np.ndarray, shape (n_outparams,)
            Output values in the units specified by ``outparams``, in the same
            order as ``self.outparams``.
        """
        outvalues = np.zeros(self.noutparam)

        for i in range(self.noutparam):
            base = kwargs.get('base', output.log)
            for attr in self.outparams[i].split('.'):
                base = getattr(base, attr)

            # Apply unit conversions to SI
            if self.out_units[i] is None:
                outvalues[i] = base
            else:
                try:
                    outvalues[i] = base.to(self.out_units[i]).value
                except:
                    # Some VPLanet modules (e.g. thermint) emit "(null)" as the unit
                    # for certain outputs in the log file. vplot cannot parse "(null)",
                    # so it assigns an empty unit string. Since vpl.in forces SI units
                    # (sUnitMass kg, sUnitLength m, sUnitTime sec, sUnitTemp K), the
                    # raw numerical value is already in SI — just take it directly.
                    try:
                        outvalues[i] = (base.value * self.out_units[i].si).to(self.out_units[i]).value
                    except:
                        outvalues[i] = np.nan

        return outvalues

    
    def get_evol(self, output):
        """Extract time-series evolution arrays from a VPLanet output object.

        Reads the ``.forward`` file data (loaded by ``vplanet.get_output``),
        converts each tracked quantity to its requested unit, and returns the
        time array and a list of converted output arrays.

        Parameters
        ----------
        output : vplanet output object
            Result returned by ``vplanet.get_output(outpath)``.

        Returns
        -------
        time_out : astropy.units.Quantity
            Time array from the VPLanet ``.forward`` file, in the native
            simulation time unit (seconds in SI mode).
        evol_out : list of astropy.units.Quantity
            One entry per output parameter in ``self.outparams``, each a
            1-D array with the requested unit applied.

        Notes
        -----
        This method assumes all ``outparams`` keys follow the
        ``"final.body.ParamName"`` naming convention.
        """

        evol_out = []

        for bf in sorted(self.out_body_name_dict.keys()):

            body_outputs = getattr(output, bf)[:]

            # arr[0] is time, create separate array
            time_out = body_outputs[0]
            body_out_array = body_outputs[1:]

            body_out_units = self.out_body_unit_dict[bf]
            body_out_names = self.out_body_name_dict[bf]
            body_nparam = len(self.out_body_unit_dict[bf])

            for ii in range(body_nparam):
                if body_out_units[ii] is None:
                        evol_out.append(body_out_array[ii].value)
                else:
                    try:
                        try:
                            output_converted = body_out_array[ii].to(body_out_units[ii])
                        except:
                            # Some VPLanet modules (e.g. thermint) emit "(null)" as the unit
                            # for certain outputs in the log file. vplot cannot parse "(null)",
                            # so it assigns an empty unit string. Since vpl.in forces SI units
                            # (sUnitMass kg, sUnitLength m, sUnitTime sec, sUnitTemp K), the
                            # raw numerical value is already in SI — just take it directly.
                            output_converted = (body_out_array[ii].value * body_out_units[ii].si).to(body_out_units[ii])
                        evol_out.append(output_converted)
                    except:
                        print(f"Failed to convert parameter {bf} {body_out_names[ii]} to unit {body_out_units[ii]}")
                        print(f"{bf} {body_out_names[ii]} array: {body_out_array[ii]}")

        return time_out, evol_out


    def run_model(self, theta, remove=True, outsubpath=None, return_output=False):
        """Run VPLanet for a single parameter vector and return the results.

        Writes input files to a unique randomized subdirectory of
        ``self.outpath_base``, executes VPLanet, parses the output, and
        optionally removes the run directory afterwards.

        Parameters
        ----------
        theta : array-like of float
            Parameter values in the units defined by ``inparams``, ordered to
            match ``self.inparams``.
        remove : bool, optional
            If ``True`` (default) delete the run directory after the model
            completes.  Set to ``False`` to inspect the raw VPLanet output.
        outsubpath : str or int, optional
            Subdirectory name appended to ``self.outpath_base``.  If ``None``
            (default) a random 12-digit hex string is used to avoid collisions
            in parallel runs.
        return_output : bool, optional
            If ``True`` return the raw ``vplanet.get_output`` object instead of
            the processed array.  Useful for debugging.  Defaults to ``False``.

        Returns
        -------
        np.ndarray or dict
            * If ``timesteps`` was **not** set at construction: a 1-D
              ``np.ndarray`` of shape ``(n_outparams,)`` with the final-state
              scalar values in the units specified by ``outparams``.
            * If ``timesteps`` **was** set: a ``dict`` mapping each
              ``outparams`` key to its time-series ``astropy.units.Quantity``
              array, plus a ``"Time"`` key containing the time axis in years.

        Examples
        --------
        >>> result = vpm.run_model([0.09, 7.6], remove=True)
        >>> print(result)   # array([5.22e-4])  — final Luminosity in Lsun
        """

        # randomize output directory
        if outsubpath is None:
            # create a unique randomized path to execute the model files
            outsubpath = random.randrange(16**12)
        outpath = f"{self.outpath_base}/{outsubpath}"

        self.initialize_model(theta, outpath=outpath)
        
        t0 = time.time()

        # Execute the model!
        subprocess.call([f"{self.executable} {self.vplfile}"], cwd=outpath, shell=True)

        try:
            output = vplanet.get_output(outpath)
        except:
            print("If no logfile is found, it's probably because there was something wrong with the infile formatting.")
            print(f"Try executing 'vplanet vpl.in' in directory {outpath} to diagnose the error in vplanet.")

        if return_output == True:
            return output

        if self.verbose == True:
            print('Executed model %s/vpl.in %.3f s'%(outpath, time.time() - t0))

        # return final values
        outvalues = self.get_outparam(output)
        model_final = outvalues

        # if timesteps are specified, return evolution
        if self.timesteps is not None:
            model_time, evol = self.get_evol(output)
            model_evol = dict(zip(self.outparams, evol))
            model_evol['Time'] = model_time.to(u.yr)

        if self.verbose:
            print("\nOutput:")
            print("-----------------")
            for ii in range(self.noutparam):
                print("%s : %s [%s]"%(self.outparams[ii], model_final[ii], self.out_units[ii]))
            print("")

        if remove == True:
            shutil.rmtree(outpath)

        if self.timesteps is None:
            return model_final
        else:
            return model_evol


    def quickplot_evol(self, time, evol, ind=None):
        """Quick diagnostic plot of evolution time series.

        Parameters
        ----------
        time : array-like
            Time array (any units; used as the x-axis).
        evol : array-like, shape (n_outparams, n_timesteps)
            Evolution arrays, one row per output parameter.
        ind : array-like of int, optional
            Indices of rows in ``evol`` to plot.  Defaults to all rows.

        Returns
        -------
        matplotlib.figure.Figure
        """

        import matplotlib.pyplot as plt
        from matplotlib import rc
        try:
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
            rc('text', usetex=True)
            rc('xtick', labelsize=20)
            rc('ytick', labelsize=20)
        except:
            rc('text', usetex=False)

        time = np.array(time)
        evol = np.array(evol)

        if ind is None:
            ind = np.arange(len(evol))
        evol = evol[ind]
        nplots = len(evol)

        fig, axs = plt.subplots(nplots, 1, figsize=[4*nplots, 8], sharex=True)
        for ii in range(nplots):
            axs[ii].plot(time, evol[ii])
        plt.close()

        return fig