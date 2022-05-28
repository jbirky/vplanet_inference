import multiprocessing as mp
from vplanet_inference import analysis


synth = analysis.AnalyzeVplanetModel("config.yaml", verbose=False, ncore=mp.cpu_count())

# nsample should be an integer 2^n (change to something like 1024 for more robust sample)
synth.variance_global_sensitivity(nsample=8)