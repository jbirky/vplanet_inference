## vplanet_inference

Python tools for doing inference with vplanet

### Setup

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

params = ['vpl.dStopTime', 'primary.dMass', 'secondary.dMass']
infile_list = ['vpl.in', 'primary.in', 'secondary.in']

vpm = VplanetModel(params, inpath='infiles/', outpath='output/', infile_list=infile_list)
```

#### Execute a single model
Return all output (vplot object):
```
theta = np.array([5e8, 1.0, 0.9])

output = vpm.run_model(theta)
```

Return specified output parameters:
```
theta = np.array([5e8, 1.0, 0.9])
outparams = ['primary.Radius', 'secondary.Radius']

output = vpm.run_model(theta, outparams=outparams)
```