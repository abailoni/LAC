from utils.data_utils import loadConfigFile, savedata
import h5py as h
import numpy as np

confFile = loadConfigFile("./config/paths.yaml")
googleData = confFile['CREMI_googleAffinities']

# Affinities:
f1 = h.File(googleData+"sampleA_orig.h5")
f2 = h.File(googleData+"sampleB_corrected_orig.h5")
f3 = h.File(googleData+"sampleC_corrected_orig.h5")

# Apply same shape of raw CREMI data:
affA = np.transpose(f1['data'][:], (3,2,1,0))
affB = np.transpose(f2['data'][:], (3,2,1,0))
affC = np.transpose(f3['data'][:], (3,2,1,0))

# Adjust affinities so that:
#   0 --> dim. 1
#   1 --> dim. 2
#   2 --> dim. 3
# Note that in CREMI data the "anisotropic slices" are along x (not z).

permutation = [2,1,0]
i = np.argsort(permutation)
affA = affA[i,...]
affB = affB[i,...]
affC = affC[i,...]

savedata(affA, googleData+"sampleA.h5")
savedata(affB, googleData+"sampleB.h5")
savedata(affC, googleData+"sampleC.h5")

# # Raw data:
# g1 = h.File(googleData+"../sample_A_20160501.hdf")
# g2 =h.File(googleData+"../sample_B_20160501.hdf")
# g3 =h.File(googleData+"../sample_C_20160501.hdf")
#
# print f1['data']
# print g1['volumes/raw']
#
# rawA = g1['volumes/raw'][:].astype(np.float32)
# labA = g1['volumes/batchLabels/neuron_ids'][:].astype(np.uint32)
#
# from utils.voluminaView import volumina_n_layer
# volumina_n_layer([rawA, affA[0], affA[1], affA[2], labA], ["raw", "affx", "affy", "affz", "labs"])
#
#
#
#
#
#
#
