import numpy as np
import pandas as pd

## Indices for hypo frames
HYP_FRAMES_IDX = 0
## Indices for hypo vehicle ids
HYP_VID_IDX = 1
## Indices for hypo top left coords
HYP_TL_X = 2
HYP_TL_Y = 3

## Same as above but for the formatted matrix
FMT_DSET_IDX = 0
FMT_VID_IDX = 1
FMT_FRAMES_IDX = 2
FMT_TL_X = 3
FMT_TL_Y = 4

def getDsetID(fname):
    ## filenames follow 'folder/noisy_hypotheses_xxx.txt'. Interested in the xx but do not
    ## want to make assumptions on xx length
    return fname.split('/')[1]
        .split('_')[2]
        .split('.')[0]

# returns matrix with columns
# Dset id,vehicle id,frame number,tl x,tl y
def formatHypo(dsetID, hypo_mtrx):

    formatted_mtrx = np.zeros((hypo_mtrx.shape[0], 5))

    formatted_mtrx[:, FMT_DSET_IDX] = dsetID
    formatted_mtrx[:, FMT_VID_IDX] = hypo_mtrx[:,HYP_VID_IDX]
    formatted_mtrx[:, FMT_FRAMES_IDX] = hypo_mtrx[:,HYP_FRAMES_IDX]
    formatted_mtrx[:, FMT_TL_X] = hypo_mtrx[:,HYP_TL_X]
    formatted_mtrx[:, FMT_TL_Y] = hypo_mtrx[:,HYP_TL_Y]

    return formatted_mtrx


def getFormattedInput(raw_fname):

    dsetID = getDsetID(raw_fname)

    with open(raw_fname) as f:
        hypo_mtrx = np.loadtxt(f, delimiter=',')

    return formatHypo(dsetID, hypo_mtrx)

def 
