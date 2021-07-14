## vplanet_inference

Python tools for doing inference with vplanet

### Basic Setup

Dependencies:
- vplanet
- vplot

Add environmental variable:
```
export PYTHONPATH=$PYTHONPATH:/path_to/vplanet_inference
```

#### Setting up the infiles
Example input directory structure:
```
infiles/
    vpl.in
    primary.in
    secondary.in
```

#### Initialize the model:
```
from vplanet_inference import VplanetModel

inparams = ['vpl.dStopTime', 'primary.dMass', 'secondary.dMass']
infile_list = ['vpl.in', 'primary.in', 'secondary.in']

vpm = VplanetModel(inparams, inpath='infiles/', infile_list=infile_list)
```

#### Execute a single model
Return all output (vplot object):
```
theta = np.array([5e8, 1.0, 0.9])

output = vpm.run_model(theta, outpath='output/')
```

Or return specified output parameters:
```
theta = np.array([5e8, 1.0, 0.9])
outparams = ['initial.primary.Luminosity', 'final.primary.Radius', 'initial.secondary.Luminosity', 'final.secondary.Radius']

output = vpm.run_model(theta, outparams=outparams)
```

#### Create infiles for many models
```
theta_list = np.array([[5e8, 1.0, 0.9],
                        ...
                       [5e8, 0.9, 1.0]])

for i, tt in enumerate(theta_list):
    vpm.initialize_model(tt, outpath='output/model%s'%(i)) 
```
This should write the infiles
```
output/
    model0/
        vpl.in
        primary.in
        secondary.in
    ...
    model10/
        vpl.in
        primary.in
        secondary.in
```