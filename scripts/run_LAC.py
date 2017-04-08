from utils.data_utils import loadConfigFile
import sys
import LAC.dataProvider as dtProv


import logging
logging.basicConfig(level=logging.INFO)



confFile = loadConfigFile("./config/paths.yaml")
CREMI_data_path = confFile['CREMI_data_path']

netFov = 5
n_batches = 10


# Load dataset:
dataProvider = dtProv.AffinityDataProvider(
    CREMI_data_path+"sample_A_20160501.h5",
    affinity_data_path = CREMI_data_path+'affin_google/sampleA.h5',
    affinity_data_h5path = 'data',
    mirrowBorders=True,
    netFov=netFov)

# dataProvider.show_dataset_volumina()

# # OLD STUFF:
#
# # Create a batch:
# batch = dtProv.LAC_BatchProvider(dataProvider,n_batches,dimXYpred=500,netFov=netFov)
#
# batch.init_batch(range(n_batches))
#
# print batch.get_GTlabelBatch_shape()
#
# batch.show_batch_volumina(prob_map=True)
#
# # cropped_batch = batch.crop_batch_XY((100,200), slice(0,5))
# # dtProv.show_cropped_batch_volumina(cropped_batch)


batchProv = dtProv.StaticBatchProvider2D(dataProvider,n_batches,sizeXYpred=(10,10),netFov=netFov)

from LAC.env import LACstate

options = []
prova = LACstate(batchProv, options)
