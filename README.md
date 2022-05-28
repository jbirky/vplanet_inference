## vplanet_inference

Python tools for doing inference with vplanet

### Installation

```
git clone https://github.com/jbirky/vplanet_inference
cd vplanet_inference
python setup.py install
```

Dependencies (optional):
- [alabi](https://github.com/jbirky/alabi)

```
git clone https://github.com/jbirky/alabi
cd alabi
python setup.py install
```

### Basic Example

Compute radius and luminosity evolution of a solar mass star over 2.5 Gyr:

```
# Specify infile directory, and output file directory
inpath = os.path.join(vpi.INFILE_DIR, "stellar/")
outpath = "output/"

# Fixed parameter substitutions
fixparams = {"vpl.dOutputTime": u.Gyr, 
             "star.dAge": u.Gyr}
             
fixvalues = np.array([1e7, 1e6])

# Variable parameter substitutions
inparams = {"star.dMass": u.Msun, 
            "vpl.dStopTime": u.Gyr}

outparams = {"final.star.Radius": u.Rsun, 
             "final.star.Luminosity": u.Lsun}

# Initialize the vplanet model
vpm = vpi.VplanetModel(inparams=inparams, outparams=outparams, fixparams=fixparams,
                       inpath=inpath, outpath=outpath)

# Run the vplanet model
theta = np.array([1.0, 2.5])
output = vpm.run_model(theta)
```
