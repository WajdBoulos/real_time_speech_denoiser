#!/usr/bin/env python3

from __future__ import absolute_import

import struct

def _get_unpack_string(sample_size):
    if sample_size == 4:
        return "f"
    elif sample_size == 2:
        return "h"
    else:
        raise ValueError(f"unsupported sample size {sample_size}")

def _should_use_int(sample_size):
    if sample_size == 4:
        return False
    elif sample_size == 2:
        return True
    else:
        raise ValueError(f"unsupported sample size {sample_size}")

def raw_samples_to_array(data, sample_size):
    return [struct.unpack(_get_unpack_string(sample_size), data[i : i + sample_size])[0] for i in range(0, len(data), sample_size)]

def array_to_raw_samples(samples, data, sample_size):
    for i, sample in zip(range(0, len(data), sample_size), samples):
        if _should_use_int(sample_size):
            sample = int(sample)
        sample_bytes = struct.pack(_get_unpack_string(sample_size), sample)
        for j, value in enumerate(sample_bytes):
            data[i + j:i + j + 1] = bytes([value])
