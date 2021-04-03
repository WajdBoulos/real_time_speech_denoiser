#!/usr/bin/env python3

"""Utility module to help convert samples from/to a python bytearray of the raw bytes of the stream to/from a list of samples.
The samples in the code are passed between the objects as a raw stream of bytes, where every few bytes represent some value of a sample.
As some objects (processors, writers) want to read or make changes to the actual sample values, this module help convert between this
representation and a list of samples.

Samples can  be represented in two ways - a 32 bit float, or 16 bit integer. All the functions in this module can accept and convert
both types, and know which type they need to deal with by the size of the samples.
"""

from __future__ import absolute_import

import struct

def _get_unpack_string(sample_size):
    """Get the string needed to pack/unpack a bytearray of sample_size bytes into a sample of this type.

    Args:
        sample_size (integer):  Size of a sample in bytes. Can be 2 or 4.
        
    Returns:
        The string needed to pass to struct.pack or struct.unpack.
    """
    if sample_size == 4:
        # The samples are represented by a 32 bit float
        return "f"
    elif sample_size == 2:
        # The samples are represented by a 16 bit integer
        return "h"
    else:
        # Unknown sample size
        raise ValueError(f"unsupported sample size {sample_size}")

def _should_use_int(sample_size):
    """Decide whether or not to use an int for the samples. The other option is a float.

    Args:
        sample_size (integer):  Size of a sample in bytes. Can be 2 or 4.
        
    Returns:
        Whether the sample needs to be converted to an int before packing it.
    """
    if sample_size == 4:
        # The samples are represented by a 32 bit float
        return False
    elif sample_size == 2:
        # The samples are represented by a 16 bit integer
        return True
    else:
        # Unknown sample size
        raise ValueError(f"unsupported sample size {sample_size}")

def raw_samples_to_array(data, sample_size):
    """Convert a buffer of raw bytes into an array of samples.

    Args:
        data (buffer):          Buffer of bytes that represents samples, either in a 32 bit float or 16 bit integer formatting.
        sample_size (integer):  Size of a sample in bytes. Can be 2 or 4.
        
    Returns:
        A list of all the samples that where in the buffer.
    """
    # List comprehension to unpack each sample_size bytes, according to the format decided by sample_size.
    return [struct.unpack(_get_unpack_string(sample_size), data[i : i + sample_size])[0] for i in range(0, len(data), sample_size)]

def array_to_raw_samples(samples, data, sample_size):
    """Convert a list of samples to a raw bytes buffer. The raw bytes are written directly into the supplied buffer.

    Args:
        samples (list):         List of samples to convert to the raw bytes.
        data (buffer):          Buffer of bytes to fill with the conversion of the samples.
        sample_size (integer):  Size of a sample in bytes. Can be 2 or 4.
        
    Returns:
        This function does not return anything.
    """
    # Go over each of the samples, and hold an index to point to where the raw bytes of the sample should start to be
    # inserted into in the data buffer.
    for i, sample in zip(range(0, len(data), sample_size), samples):
        # Check if this formatting requires that the samples be int the int format
        if _should_use_int(sample_size):
            # Convert the samples from (potentially) a float to an int
            sample = int(sample)
        # Pack the samples into the raw bytes that represent them
        sample_bytes = struct.pack(_get_unpack_string(sample_size), sample)
        # Go over the raw bytes, and fill each of them to where it needs to be in the data buffer.
        for j, value in enumerate(sample_bytes):
            data[i + j:i + j + 1] = bytes([value])
