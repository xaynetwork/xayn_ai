#!/usr/bin/python3

"""
Extracts ListNet weights from a onnx file and writes them into a bincode compatible file.

We write a bincode compatible format. Even through there is no official bincode specifications
we still can do this as bincode guarantees a stable data format (except on new major releses).

Extracted Tensor Initial Values
===============================

- 'ltr/v1/dense_1/weights'         from 'StatefulPartitionedCall/functional_1/dense_1/Tensordot/ReadVariableOp:0'
- 'ltr/v1/dense_1/bias'            from 'StatefulPartitionedCall/functional_1/dense_1/BiasAdd/ReadVariableOp:0'
- 'ltr/v1/dense_2/weights'         from 'StatefulPartitionedCall/functional_1/dense_2/Tensordot/ReadVariableOp:0'
- 'ltr/v1/dense_2/bias'            from 'StatefulPartitionedCall/functional_1/dense_2/BiasAdd/ReadVariableOp:0'
- 'ltr/v1/scores/weights'          from 'StatefulPartitionedCall/functional_1/scores/Tensordot/ReadVariableOp:0'
- 'ltr/v1/scores/bias'             from 'StatefulPartitionedCall/functional_1/scores/BiasAdd/ReadVariableOp:0'
- 'ltr/v1/scores_prop_dist/weights' from 'StatefulPartitionedCall/functional_1/scores_prob_dist/MatMul/ReadVariableOp:0'
- 'ltr/v1/scores_prop_dist/bias'    from 'StatefulPartitionedCall/functional_1/scores_prob_dist/BiasAdd/ReadVariableOp:0'

All other parameters are only used to configure the pipeline like,
making sure we gather into the right matrix after doing a tensor
dot multiplication, which axis to run the softmax on, into which
form to flatten etc.

Pre-Amble
==========

A file/stream is prefixed with a single version *byte*,
this must be read/skipped over before passing the rest
to bincode.

The current version is 0x01.


Required Bincode Option
========================

This serializes using `Fixed` integer encoding,
as such the deserializer must be setup to use
exactly that. (We would safe 3+9*nr_arrays bytes
only which isn't worth it.)

LittleEndianness is used for all integer.


What is Serialized
===================

Should be equivalent to bincode
serializing following `HashMap<String, Array>`
with:

```
struct Array {
    shape: &[usize], //or Vec<usize>
    data: &[f32], //or Vec<usize>
}
```

Be aware that the order of fields matters
for serde, and especially bincode.

Data should be reinterpreted as a nested
array based on 'C" format (i.e. row based,
i.e. an array of rows). Be aware that this
is independent of the byte-order. 'C' format
just mens row based instead of column based
(which is known as 'F', i.e. fortran format).

Bincode Serialization Format (Unofficial Spec)
==============================================

Integer/Float
--------------

As we use fixed encoding all integers
are the raw byte representation in
little endianness.

Same is true for floats.

usize/iszie
------------

Are always encoded as u64,i64.

Maps
-----

```ascii
[len:u64]([field][value])*len
```

String
-------
[len:u64][byte:u8]*len

Vec<T>
--------
[len:u64][item]*len


Final Binary Format
====================

```ascii
[version=0x1:u8][nr_arrays:u64](
    [name_len:u64][name_utf8:u8]*name_len
    [ndim:u64][dim:u64]*ndim
    [n_params:u64][param:f32]*n_params
)*nr_arrays
```

There is a additional invariant that
`n_params == 1*...*dim[ndim-1]`, if that
is not meet conversion to `ndarray::Array`
and similar will fail.

If the first byte is == 0x1 and excluded
the rest is bincode decodeable as mentioned
above. This is also why all length are in
u64.
"""

#TODO refactor(a lot), and add tests

import onnx, onnx.numpy_helper
from typing import Tuple, Dict

LIST_NET_NAME_MAPPING = {
    'StatefulPartitionedCall/functional_1/dense_1/Tensordot/ReadVariableOp:0'           : 'ltr/v1/dense_1/weights'        ,
    'StatefulPartitionedCall/functional_1/dense_1/BiasAdd/ReadVariableOp:0'             : 'ltr/v1/dense_1/bias'           ,
    'StatefulPartitionedCall/functional_1/dense_2/Tensordot/ReadVariableOp:0'           : 'ltr/v1/dense_2/weights'        ,
    'StatefulPartitionedCall/functional_1/dense_2/BiasAdd/ReadVariableOp:0'             : 'ltr/v1/dense_2/bias'           ,
    'StatefulPartitionedCall/functional_1/scores/Tensordot/ReadVariableOp:0'            : 'ltr/v1/scores/weights'         ,
    'StatefulPartitionedCall/functional_1/scores/BiasAdd/ReadVariableOp:0'              : 'ltr/v1/scores/bias'            ,
    'StatefulPartitionedCall/functional_1/scores_prob_dist/MatMul/ReadVariableOp:0'     : 'ltr/v1/scores_prop_dist/weights',
    'StatefulPartitionedCall/functional_1/scores_prob_dist/BiasAdd/ReadVariableOp:0'    : 'ltr/v1/scores_prop_dist/bias'   ,
}

def load_list_net_parameters(path):
    model = onnx.load(path)

    result = { name: None for name in LIST_NET_NAME_MAPPING.values() }

    for tensor in model.graph.initializer:
        if tensor.name in LIST_NET_NAME_MAPPING:
            result[LIST_NET_NAME_MAPPING[tensor.name]] = onnx.numpy_helper.to_array(tensor)

    for (name, val) in result.items():
        if val is None:
            raise RuntimeError(f"Missing tensor initializers for {LIST_NET_NAME_MAPPING[name]}")

    return result


VERSION=1
#u64
LEN_SIZE=8
ENDIAN='little'

def write_list_net_parameters(path, matrices):
    with open(path, 'wb') as out:
        write_list_net_parameters_to_sink(matrices, out)

def write_list_net_parameters_to_sink(matrices, out):
    version = VERSION.to_bytes(1, ENDIAN)
    out.write(version)

    write_map(matrices, write_string, write_array, out)

def write_map(input_map, write_key, write_value, out):
    length = len(input_map).to_bytes(LEN_SIZE, ENDIAN)
    out.write(length)

    for (key, value) in input_map.items():
        write_key(key, out)
        write_value(value, out)


def write_byte_slice(data, out):
    length = len(data).to_bytes(LEN_SIZE, ENDIAN)
    out.write(length)
    out.write(data)

def write_string(string, out):
    encoded = string.encode('utf-8')
    write_byte_slice(encoded, out)

def write_shape(shape, out):
    length = len(shape)
    assert length > 0, "can only serialize arrays not scalar values"
    length = length.to_bytes(LEN_SIZE, ENDIAN)
    out.write(length)
    for dim in shape:
        dim = dim.to_bytes(LEN_SIZE, ENDIAN)
        out.write(dim)

def write_array(array, out):
    write_shape(array.shape, out)
    # row based, byte-order was already managed
    # '<f4' (dtype) => little endian 4byte float
    # oder='C' => row based array
    # casting='equiv' => only allow byte-order changes
    # copy=False => don't copy the array if not necessary
    array_bytes = array.astype('<f4', order='C', casting='equiv', copy=False).tobytes(order='C')
    n_elements = array.size.to_bytes(LEN_SIZE, ENDIAN)
    out.write(n_elements)
    out.write(array_bytes)


# given from numpy import array, float32
#input:  {'a': array([[1., 2.], [3., 4.]], dtype=float32), 'b': array([3., 2., 1., 4.], dtype=float32)}
#output: b'\x01\x02\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00a\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80?\x00\x00\x00@\x00\x00@@\x00\x00\x80@\x01\x00\x00\x00\x00\x00\x00\x00b\x01\x00\x00\x00\x00\x00\x00\x00\x04\x00\x00\x00\x00\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00@@\x00\x00\x00@\x00\x00\x80?\x00\x00\x80@'


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5 or sys.argv[1] != '--from' or sys.argv[3] != '--to':
        raise RuntimeError("wrong arguments expect --from <ltr.onnx> --to ltr.params")
    in_path = sys.argv[2]
    out_path = sys.argv[4]

    matrices = load_list_net_parameters(in_path)
    write_list_net_parameters(out_path, matrices)

