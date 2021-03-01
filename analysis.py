import numpy as np
import emcee

__all__ = ["extractMCMCResults"]


def extractMCMCResults(filename, verbose=True, applyBurnin=True, thinChains=True,
                       blobsExist=True, burn=None, removeRogueChains=False):
    """
    Extract and process MCMC results

    Parameters
    ----------
    filename : str
        path to emcee MCMC h5 file
    verbose : bool (optional)
        Output convergence diagnostics? Defaults to True.
    applyBurnin : bool (optional)
        Apply a burnin to reduce size of chains? Defaults to True.
    thinChains : bool (optional)
        Thin chains to reduce their size? Defaults to True.
    blobsExist : bool (optional)
        Whether or not blobs exist.  If True, return them! Defaults to True.
    burn : int (optional)
        User-specified burn-in. Defaults to None
    removeRogueChains : bool (optional)
        Whether or not to remove rogue chains, that is, chains with acceptance
        fractions < 0.01.

    Returns
    -------
    chain : numpy array
        MCMC chain
    blobs : numpy array
        MCMC ancillary and derived quantities. Only returned if blobsExist is True
    """

    # Open file
    reader = emcee.backends.HDFBackend(filename)

    if verbose:
        # Compute acceptance fraction for each walker
        print("Number of iterations: %d" % reader.iteration)
        print("Acceptance fraction for each walker:")
        print(reader.accepted / reader.iteration)
        print("Mean acceptance fraction:", np.mean(reader.accepted / reader.iteration))

    # Compute convergence diagnostics

    # Compute burnin?
    tau = reader.get_autocorr_time(tol=0)
    if applyBurnin:
        burnin = int(2*np.max(tau))
    else:
        burnin = 0
    if thinChains:
        thin = int(0.5*np.min(tau))
    else:
        thin = 1

    # Use user-specified burn-in?
    if burn is not None:
        burnin = burn

    # Output convergence diagnostics?
    if verbose:
        print("Burnin, thin:", burnin, thin)

        # Is the length of the chain at least 50 tau?
        print("Likely converged if iterations > 50 * tau, where tau is the integrated autocorrelation time.")
        print("Number of iterations / tau:", reader.iteration / tau)
        print("Mean Number of iterations / tau:", np.mean(reader.iteration / tau))

    # Load data
    if removeRogueChains:
        # Read in chain and remove errant walkers
        chain = reader.get_chain(discard=burnin, flat=False, thin=thin)

        # Find chains to keep
        mask = reader.accepted / reader.iteration > 0.15
        chain = chain[:,mask,:]

        # Flatten chain
        chain = chain.reshape((chain.shape[0]*chain.shape[1], chain.shape[-1]))
    else:
        chain = reader.get_chain(discard=burnin, flat=True, thin=thin)

    # Properly shape blobs
    tmp = reader.get_blobs(discard=burnin, flat=True, thin=thin)
    if blobsExist:
        blobs = []
        for bl in tmp:
            blobs.append([bl[ii] for ii in range(len(bl))])
        blobs = np.array(blobs)

        return chain, blobs
    else:
        return chain