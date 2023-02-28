from mmengine.device import get_device
from typing import Mapping, Sequence
import torch
import numpy as np


def cast_data(data):
    if isinstance(data, Mapping):
        return {key: cast_data(data[key]) for key in data}
    elif isinstance(data, (str, bytes)) or data is None:
        return data
    elif isinstance(data, Sequence):
        return type(data)(cast_data(sample) for sample in data)
    elif isinstance(data, np.float64):
        data = torch.FloatTensor(np.array(data))
        return data.to(get_device())
    elif isinstance(data, np.ndarray):
        data = torch.FloatTensor(np.array(data))
        return data.to(get_device())
    elif isinstance(data, np.int64):
        data = torch.FloatTensor(np.array(float(data)))
        return data.to(get_device())
    elif isinstance(data, torch.Tensor):
        return data.to(get_device())
    else:
        raise TypeError(
            'batch data must contain tensors, numpy arrays, numbers, '
            f'dicts or lists, but found {type(data)}')
