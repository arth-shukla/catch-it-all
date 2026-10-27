from typing import Dict, TypeVar

import dacite

import numpy as np

from humanoid.utils.dataclasses import recursive_asdict


T = TypeVar("T")


def dict_merge(dct: dict, merge_dct: dict):
    """In place recursive merge of `merge_dct` into `dct`"""
    for k, v in merge_dct.items():
        if (
            k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)
        ):  # noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


def dict_dataclass_merge(base, additional, dataclass: T = None) -> T:
    additional = recursive_asdict(additional)
    merged_dict = recursive_asdict(base)
    dict_merge(merged_dict, additional)
    if dataclass is not None:
        return dacite.from_dict(
            data_class=dataclass,
            data=merged_dict,
        )
    return merged_dict


def extract_scalars_from_info(info: dict, blacklist=()) -> Dict[str, float]:
    """Recursively extract scalar metrics from an info dict returned by env.step.

    Args:
        info (dict): info dict
        blacklist (tuple, optional): keys to exclude.

    Returns:
        Dict[str, float]: scalar metrics
    """
    ret = {}
    for k, v in info.items():
        if k in blacklist:
            continue

        # Ignore placeholder
        if v is None:
            continue

        # Recursively extract scalars
        elif isinstance(v, dict):
            ret2 = extract_scalars_from_info(v, blacklist=blacklist)
            ret2 = {f"{k}.{k2}": v2 for k2, v2 in ret2.items()}
            ret2 = {k2: v2 for k2, v2 in ret2.items() if k2 not in blacklist}

        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            try:
                ret[k] = float(v)
            except:
                pass
    return ret
