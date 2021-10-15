import numpy as np

def corr_metric(predicted, measured):
    corr = np.corrcoef(predicted, measured)

    return corr[0][-1]
