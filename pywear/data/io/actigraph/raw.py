import h5py
import json
import numpy as np


def read_raw_h5(fname):
    """Reader function for Raw h5 actigraph formatted file

    Args:
        fname (str): The raw filename to load

    Returns:
        raw (dict): A dictionary object containing h5 actigraph data
    """

    with h5py.File(fname, "r") as f:
        rawAccel = np.array(f.get("accelerometer1Raw"))
        fsamp = np.array(f.get("sampleRate"))
        info_str = f.get("info_str")
        info = json.loads(info_str[:, 0].tostring())

    raw = dict()
    raw['acc'] = rawAccel
    raw['fsamp'] = fsamp
    raw['info'] = info

    return raw
