import numpy as np 
import pandas as pd
import multiprocessing as mp
import scipy
from functools import partial
from tqdm import tqdm

from alabi import utility as ut
import vplanet_inference as vpi

from earth_model import lnlike, like_data, inlabels, bounds, prior_sampler
from fischer import compute_fisher_information, analyze_fisher_information


neg_lnlike = lambda x: -lnlike(x, like_data)
def opt_parallel(x0):
    mll = scipy.optimize.minimize(neg_lnlike, x0=x0, bounds=bounds, method="nelder-mead", 
                                  options={"maxiter":1000, "adaptive": True})
    return mll.x


if __name__ == "__main__":

    # find maximum likelihood estimates in parallel
    nopts = 100
    ncore = 32
    thetas_random = prior_sampler(nsample=nopts)
    
    print(f"Running {nopts} optimization runs with {ncore} cores...")
    with mp.Pool(ncore) as pool:
        results = list(tqdm(pool.imap(opt_parallel, thetas_random), 
                           total=len(thetas_random), 
                           desc="Optimizing"))
    
    # Create DataFrame with parameter values
    df = pd.DataFrame({
        "key": inlabels + ["lnlike"], 
    })
    for ii, res in enumerate(results):
        df[f"mll_{ii}"] = list(res) + [lnlike(res, like_data)]
    df.to_csv(f"mll_rad_heat_{len(inlabels)}_parameters.csv", index=False)
        
    # Compute Fisher Information at true parameters
    # df = pd.read_csv("mll_rad_heat_parameters_earth.csv")
    tb = np.array(df[["key"] + [f"mll_{i}" for i in range(30)]])
    best_theta = tb.T[np.argmin(tb.T[-1])][:-1]
    FI = compute_fisher_information(lnlike, best_theta, like_data, method='hessian')

    print("Fisher Information Matrix:")
    print(FI)
    print("\nAnalysis:")
    results = analyze_fisher_information(FI, param_names=inlabels)
    np.savez("fisher_information_earth.npz", FI=FI, results=results)
    
    print(f"Standard Errors: {results['standard_errors']}")
    print(f"Condition Number: {results['condition_number']:.2f}")
    print(f"\nCorrelation Matrix:")
    print(results['correlation_matrix'])