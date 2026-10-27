import copy
from typing import List, Optional, Sequence, TypeVar

import gymnasium as gym

import numpy as np
import torch


T = TypeVar("T")


def all_equal(array: List):
    return len(set(array)) == 1


def all_same_type(array: List):
    return len(set(map(type, array))) == 1


def recursive_fill(data, fill_value, inplace=False):
    if isinstance(data, dict):
        if inplace:
            for k, v in data.items():
                data[k] = recursive_fill(v, fill_value)
            return data
        return dict((k, recursive_fill(v, fill_value)) for k, v in data.items())
    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        if inplace:
            data[:] = fill_value
            return data
        if isinstance(data, np.ndarray):
            return np.full_like(data, fill_value)
        if isinstance(data, torch.Tensor):
            return torch.full_like(data, fill_value)
    # NOTE (arth): "zeroing out" a set doesn't necessarily make sense
    if isinstance(data, set):
        raise NotImplementedError(
            f"Got set {data} with type {type(data)}. Zeroing out sets doens't really make sense. wrap in try/except if you want to ignore these sorts of errors"
        )
    if np.iterable(data):
        if inplace:
            for i in range(len(data)):
                data[i] = fill_value
        return type(data)(fill_value for _ in data)
    return 0


def recursive_to_python(data, inplace=False):
    if isinstance(data, dict):
        if inplace:
            for k, v in data.items():
                data[k] = recursive_to_python(v)
            return data
        return dict((k, recursive_to_python(v)) for k, v in data.items())
    if isinstance(data, list):
        if inplace:
            for i, x in enumerate(data):
                data[i] = recursive_to_python(x)
            return data
        return [recursive_to_python(x) for x in data]
    if isinstance(data, np.ndarray) or isinstance(data, torch.Tensor):
        return data.tolist()
    if isinstance(data, np.number):
        return data.item()
    return data


def recursive_deepcopy(data: T) -> T:
    if isinstance(data, dict):
        return dict((k, recursive_deepcopy(v)) for k, v in data.items())
    if isinstance(data, list):
        return [recursive_deepcopy(x) for x in data]
    if isinstance(data, set):
        return set(recursive_deepcopy(list(data)))
    if isinstance(data, torch.Tensor):
        return data.clone()
    if isinstance(data, np.ndarray):
        return data.copy()
    return copy.deepcopy(data)


def recursive_shallow_copy(data: T) -> T:
    if isinstance(data, dict):
        return dict((k, recursive_shallow_copy(v)) for k, v in data.items())
    if isinstance(data, list):
        return [recursive_shallow_copy(x) for x in data]
    if isinstance(data, tuple):
        return tuple(recursive_shallow_copy(x) for x in data)
    return data


def recursive_slice(obs, slice, inplace=False):
    if isinstance(obs, dict):
        if inplace:
            for k, v in obs.items():
                obs[k] = recursive_slice(v, slice)
            return obs
        return dict((k, recursive_slice(v, slice)) for k, v in obs.items())
    return obs[slice]


def recursive_cat(a, b):
    if isinstance(a, dict):
        return dict((k, recursive_cat(a[k], b[k])) for k in a)
    return torch.cat([a, b], dim=0)


def to_tensor(
    array,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: Optional[bool] = None,
):
    """
    Maps any given sequence to a torch tensor on the CPU/GPU. If physx gpu is not enabled then we use CPU, otherwise GPU, unless specified
    by the device argument

    Args:
        array: The data to map to a tensor
        device: The device to put the tensor on. By default this is None and to_tensor will put the device on the GPU if physx is enabled
            and CPU otherwise

    """
    if isinstance(array, (dict)):
        return {k: to_tensor(v, device=device) for k, v in array.items()}
    elif isinstance(array, torch.Tensor):
        ret = array.to(device).to(dtype)
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint16:
            array = array.astype(np.int32)
        elif array.dtype == np.uint32:
            array = array.astype(np.int64)
        ret = torch.tensor(array, device=device, dtype=dtype)
    else:
        if isinstance(array, list) and isinstance(array[0], np.ndarray):
            array = np.array(array)
        ret = torch.tensor(array, device=device, dtype=dtype)
    if requires_grad is not None:
        ret.requires_grad_(requires_grad)
    return ret


def to_numpy(array: Sequence, dtype=None) -> np.ndarray:
    if isinstance(array, (dict)):
        return {k: to_numpy(v, dtype=dtype) for k, v in array.items()}
    if isinstance(array, torch.Tensor):
        if dtype is not None:
            return array.cpu().numpy().astype(dtype)
        return array.cpu().numpy()
    if isinstance(array, np.ndarray):
        if dtype is not None:
            return array.astype(dtype)
        return array
    if (
        isinstance(array, bool)
        or isinstance(array, str)
        or isinstance(array, float)
        or isinstance(array, int)
    ):
        return array

    if dtype is not None:
        return np.array(array, dtype=dtype)
    return np.array(array)


def _batch(array):
    if isinstance(array, (dict)):
        return {k: _batch(v) for k, v in array.items()}
    if isinstance(array, str):
        return array
    if isinstance(array, torch.Tensor):
        return array[None, :]
    if isinstance(array, np.ndarray):
        if array.shape == ():
            return array.reshape(1, 1)
        return array[None, :]
    if isinstance(array, list):
        if len(array) == 1:
            return [array]
    if (
        isinstance(array, float)
        or isinstance(array, int)
        or isinstance(array, bool)
        or isinstance(array, np.bool_)
    ):
        return np.array([[array]])
    return array


def batch(*args):
    """Adds one dimension in front of everything. If given a dictionary, every leaf in the dictionary
    has a new dimension. If given a tuple, returns the same tuple with each element batched
    """
    x = [_batch(x) for x in args]
    if len(args) == 1:
        return x[0]
    return tuple(x)


class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (
                        torch.float32
                        if v.dtype in (np.float32, np.float64)
                        else (
                            torch.uint8
                            if v.dtype == np.uint8
                            else (
                                torch.int16
                                if v.dtype == np.int16
                                else torch.int32 if v.dtype == np.int32 else v.dtype
                            )
                        )
                    )
                    self.data[k] = torch.zeros(
                        buffer_shape + v.shape, dtype=dtype, device=device
                    )

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[: len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)
