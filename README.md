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

#### Configure vplanet forward model

```python
import vplanet_inference as vpi
import numpy as np
import os

inpath = os.path.join(vpi.INFILE_DIR, "stellar")
infile_list = ["vpl.in", "star.in"]

inparams  = ["star.dMass", 
             "star.dSatXUVFrac",
             "star.dSatXUVTime",
             "vpl.dStopTime",
             "star.dXUVBeta"]

outparams = ["final.star.Luminosity",
             "final.star.LXUVStellar"]

factor = np.array([1, -1, -1, 1e9, -1])

vpm = vpi.VplanetModel(inparams, inpath=inpath, infile_list=infile_list, factor=factor, verbose=False)
```

#### Configure prior and likelihood

```python
# Prior bounds
bounds = [(0.07, 0.11),        
          (-5.0, -1.0),
          (0.1, 12.0),
          (0.1, 12.0),
          (-2.0, 0.0)]

data = np.array([[5.22e-4, 0.19e-4],
                 [7.5e-4, 1.5e-4]])

vpm.initialize_bayes(data=data, bounds=bounds, outparams=outparams)
```