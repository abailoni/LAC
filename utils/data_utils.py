import yaml

def loadConfigFile(configpath):
    with open(configpath, mode='r') as configfile:
        try:
            config = yaml.load(configfile)
        except Exception as e:
            print("Could not parse YAML.")
            raise e
    return config


def loaddata(path):
    """
    Function to load in a TIFF or HDF5 volume from file in `path`.
    :type path: str
    :param path: Path to file (must end with .tiff or .h5).
    Author: nasimrahaman
    """
    if path.endswith(".tiff") or path.endswith(".tif"):
        try:
            from vigra.impex import readVolume
        except ImportError:
            raise ImportError("Vigra is needed to read/write TIFF volumes, but could not be imported.")

        volume = readVolume(path)
        return volume

    elif path.endswith(".h5"):
        try:
            from Antipasti.netdatautils import fromh5
        except ImportError:
            raise ImportError("h5py is needed to read/write HDF5 volumes, but could not be imported.")

        volume = fromh5(path)
        return volume

    else:
        raise NotImplementedError("Can't load: unsupported format. Supported formats are .tiff and .h5")



def savedata(data, path):
    """
    Saves volume as a .tiff or .h5 file in path.
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