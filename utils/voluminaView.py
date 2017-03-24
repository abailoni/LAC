"""
author: Constantin Pape
"""

import sys
from PyQt4.QtCore import QTimer
from PyQt4.QtGui import QApplication, QColor

import vigra
import numpy as np
import argparse
import h5py

from types import *



# plot single data layer
def volumina_single_layer(data):
	app = QApplication (sys.argv)
	from volumina.api import Viewer

	v = Viewer ()
	v.title = " Volumina Demo "
	v.showMaximized ()
	v.addGrayscaleLayer (data , name =" raw data ")

	app . exec_ ()


# plot 2 data layers
def volumina_double_layer(data, overlay):
	# get data type of the elements of overlay, to determine
	# if we use a grayscale overlay (float32) or a randomcolors overlay (uint) for batchLabels
	mask = []
	for i in range( len(overlay.shape) ):
		mask.append(0)
	mask = tuple(mask)
	data_type = type(overlay[mask])

	app = QApplication (sys.argv)
	from volumina.api import Viewer

	v = Viewer ()
	v.title = " Volumina Demo "
	v.showMaximized ()
	v.addGrayscaleLayer(data , name = " raw data ")

	if data_type == np.float32:
		v.addGrayscaleLayer(overlay , name = " overlay ")
	else:
		v.addRandomColorsLayer(overlay, name = " overlay ")

	app . exec_ ()

# plot n data layers
def volumina_n_layer(data, labels = None):
    """
    BASIC FUNCTION

    :param data:
    :param labels:
    :return:
    """

    app = QApplication (sys.argv)

    import volumina
    from volumina.api import Viewer

    v = Viewer ()
    v.title = " Volumina Demo "
    v.showNormal ()

    for ind, d in enumerate(data):
    	layer_name = "layer_" + str(ind)
        if labels is not None:
            layer_name = labels[ind]
    	# get data type of the elements d, to determine
    	# if we use a grayscale overlay (float32) or a randomcolors overlay (uint) for batchLabels
    	data_type = d.dtype

    	if data_type is FloatType or data_type == np.float32 or data_type == np.float64:
    	    v.addGrayscaleLayer(d , name = layer_name)
    	else:
    	    v.addRandomColorsLayer(d.astype(np.uint32), name = layer_name)

    app.exec_()


# plot n data layers
def volumina_flexible_layer(data, layer_types, labels = None):

    assert len(layer_types) == len(data)

    app = QApplication (sys.argv)
    from volumina.api import Viewer

    v = Viewer ()
    v.title = " Volumina Demo "
    v.showMaximized ()

    for i, d in enumerate(data):
    	layer_name = "layer_" + str(i)
        if labels is not None:
            layer_name = labels[i]
    	# get data type of the elements d, to determine
    	# if we use a grayscale overlay (float32) or a randomcolors overlay (uint) for batchLabels
    	data_type = d.dtype

        if layer_types[i] == 'Grayscale':
    	    v.addGrayscaleLayer(d , name = layer_name)
        elif layer_types[i] == 'RandomColors':
    	    v.addRandomColorsLayer(d.astype(np.uint32), name = layer_name)
        elif layer_types[i] == 'Red':
    	    v.addAlphaModulatedLayer(d , name = layer_name, tintColor=QColor(255,0,0))
        elif layer_types[i] == 'Green':
    	    v.addAlphaModulatedLayer(d , name = layer_name, tintColor=QColor(0,255,0))
        elif layer_types[i] == 'Blue':
    	    v.addAlphaModulatedLayer(d , name = layer_name, tintColor=QColor(0,0,255))
        else:
            raise KeyError("Invalid Layer Type!")

    app.exec_()


# adapted from https://github.com/ilastik/volumina/blob/master/examples/streamingViewer.py
def streaming_n_layer(files, keys, labels = None, block_shape = [100,100,100]):
    from volumina.api import Viewer
    from volumina.pixelpipeline.datasources import LazyflowSource

    from lazyflow.graph import Graph
    from lazyflow.operators.ioOperators.opStreamingHdf5Reader import OpStreamingHdf5Reader
    from lazyflow.operators import OpCompressedCache

    app = QApplication(sys.argv)

    v = Viewer()

    graph = Graph()

    def mkH5source(fname, gname):
        h5file = h5py.File(fname)
        dtype = h5file[gname].dtype

        source = OpStreamingHdf5Reader(graph=graph)
        source.Hdf5File.setValue(h5file)
        source.InternalPath.setValue(gname)

        op = OpCompressedCache( parent=None, graph=graph )
        op.BlockShape.setValue(block_shape )
        op.Input.connect( source.OutputImage )

        return op.Output, dtype

    #rawSource = mkH5source(data[0], keys[0])
    #v.addGrayscaleLayer(rawSource, name = 'raw')

    for i, f in enumerate(files):

        if labels is not None:
            layer_name = labels[i]
        else:
            layer_name = "layer_%i" % (i)

        source, dtype = mkH5source(f, keys[i])

        if np.dtype(dtype) in (np.dtype('uint8'), np.dtype('float32'), np.dtype('float64')):
            v.addGrayscaleLayer(source, name = layer_name)
        else:
            v.addRandomColorsLayer(source, name = layer_name)

    v.setWindowTitle("Streaming Viewer")
    v.showNormal()
    app.exec_()



if __name__ == '__main__':
    raw = vigra.readHDF5('/mnt/localdata0/abailoni/datasets/CREMI/sample_A_20160501.hdf','volumes/raw').astype('float32')
    labels = vigra.readHDF5('/mnt/localdata0/abailoni/datasets/CREMI/sample_A_20160501.hdf','volumes/batchLabels/neuron_ids').astype('uint32')
    volumina_n_layer([raw,labels],['raw','batchLabels'])
