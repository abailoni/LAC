import yaml
import numpy as np

"""
LOADING FILES AND DATA
"""

def loadConfigFile(configpath):
    with open(configpath, mode='r') as configfile:
        try:
            config = yaml.load(configfile)
        except Exception as e:
            print("Could not parse YAML.")
            raise e
    return config


def loaddata(path, **kwargs):
    """
    Function to load in a TIFF or HDF5 volume from file in `path`.
    :type path: str
    :param path: Path to file (must end with .tiff or .h5).
    :param kwargs: Possible options for opening the .h5 file
    """
    if path.endswith(".tiff") or path.endswith(".tif"):
        try:
            from vigra.impex import readVolume
        except ImportError:
            raise ImportError("Vigra is needed to read/write TIFF volumes, but could not be imported.")

        volume = readVolume(path)
        return volume

    elif path.endswith(".h5"):
        volume = load_h5(path, **kwargs)
        return volume
    else:
        raise NotImplementedError("Can't load: unsupported format. Supported formats are .tiff and .h5")


def load_h5(path, h5_key=None, group=None, subGroup=None, slices=None):
    """
    The function returns a list of numpy arrays

    :param path:

    :param h5_key:      names of the datasets in the h5 file. By default all the datasets will be extracted.
    :type h5_key:       str or list (default None)

    :param group:       optional group in the h5 file
    :param subGroup:    optional subgroup in the h5 file

    :param slices:      deprecated, needs improvements (see Antipasti.netdatautils.fromh5)
    :type slices:       slice indicating which part of data to keep, e.g. (slice(0,5), :, :)

    :return:            list of numpy arrays with the extracted datasets
    """
    try:
        import h5py as h
    except ImportError:
        raise ImportError("h5py is needed to read/write HDF5 volumes, but could not be imported.")

    f = h.File(path, 'r')
    if group is not None:
        g = f[group]
        if subGroup is not None:
            g = g[subGroup]
    else:   # no groups in file structure
        g = f
    if h5_key is None:     # no h5 key specified
        output = list()
        for key in g.keys():
            output.append(np.array(g[key], dtype='float32'))
    elif isinstance(h5_key, str):   # string
        output = [np.array(g[h5_key], dtype='float32')]
    elif isinstance(h5_key, list):          # list
        output = list()
        for key in h5_key:
            output.append(np.array(g[key], dtype='float32'))
    else:
        raise Exception('h5 key type is not supported')
    if slices is not None:
        import warnings
        raise warnings.warn("The slice function is no longer available", DeprecationWarning)
        output = [output[0][slices]]
    f.close()
    return output


def savedata(data, path):
    """
    Saves volume as a .tiff or .h5 file in path (Using Vigra).
    :type data: numpy.ndarray
    :param data: Volume to be saved.
    :type path: str
    :param path: Path to the file where the volume is to be saved. Must end with .tiff or .h5.
    Author: nasimrahaman
    """
    if path.endswith(".tiff") or path.endswith('.tif'):
        try:
            from vigra.impex import writeVolume
        except ImportError:
            raise ImportError("Vigra is needed to read/write TIFF volumes, but could not be imported.")

        writeVolume(data, path, '', dtype='UINT8')

    elif path.endswith(".h5"):
        try:
            from vigra.impex import writeHDF5
            vigra_available = True
        except ImportError:
            vigra_available = False
            import h5py

        if vigra_available:
            writeHDF5(data, path, "/data")
        else:
            with h5py.File(path, mode='w') as hf:
                hf.create_dataset(name='data', data=data)

    else:
        raise NotImplementedError("Can't save: unsupported format. Supported formats are .tiff and .h5")



def save_h5(path, h5_key, data, overwrite='w-', compression=None):
    try:
        import h5py as h
    except ImportError:
        raise ImportError("h5py is needed to read/write HDF5 volumes, but could not be imported.")
    f = h.File(path, overwrite)
    if isinstance(h5_key, str):
        f.create_dataset(h5_key, data=data, compression=compression)
    if isinstance(h5_key, list):
        for key, values in zip(h5_key, data):
            f.create_dataset(key, data=values, compression=compression)
    f.close()


"""
TRANSFORMATIONS IMAGES:
"""

def mirror_cube(array, pad_length):
    """
    The function extends the last two dimensions (x and y) of an array by mirroring the image
       with a certain padding.

    :param array: the last two dimensions are the ones that are mirrored at the border
    :param pad_length: how much I want to paddle at the borders

    :return: extended array
    """
    pad_info = tuple((array.ndim-2)*[(0,0)]+ [(pad_length, pad_length), (pad_length, pad_length)])
    mirrored_array = np.pad(array, pad_info, mode='reflect')
    return mirrored_array

def transpose_last(arr):
    order = np.arange(len(arr.shape))
    order[-2:] = order[-2:][::-1]
    return np.transpose(arr, tuple(order))