"""
Authors(s):
Abdelrahman Abdelhamed (a.abdelhamed@samsung.com)

Utility functions for handling DNG opcode lists.
"""
import struct
import numpy as np
from .exif_utils import get_tag_values_from_ifds


class Opcode:
    def __init__(self, id_, dng_spec_ver, option_bits, size_bytes, data):
        self.id = id_
        self.dng_spec_ver = dng_spec_ver
        self.size_bytes = size_bytes
        self.option_bits = option_bits
        self.data = data


def parse_opcode_lists(ifds):
    # OpcodeList1, 51008, 0xC740
    # Applied to raw image as read directly form file

    # OpcodeList2, 51009, 0xC741
    # Applied to raw image after being mapped to linear reference values
    # That is, after linearization, black level subtraction, normalization, and clipping

    # OpcodeList3, 51022, 0xC74E
    # Applied to raw image after being demosaiced

    opcode_list_tag_nums = [51008, 51009, 51022]
    opcode_lists = {}
    for i, tag_num in enumerate(opcode_list_tag_nums):
        opcode_list_ = get_tag_values_from_ifds(tag_num, ifds)
        if opcode_list_ is not None:
            opcode_list_ = bytearray(opcode_list_)
            opcodes = parse_opcodes(opcode_list_)
            opcode_lists.update({tag_num: opcodes})
        else:
            pass

    return opcode_lists


def parse_opcodes(opcode_list):
    """
    Parse a byte array representing an opcode list.
    :param opcode_list: An opcode list as a byte array.
    :return: Opcode lists as a dictionary.
    """
    # opcode lists are always stored in big endian
    endian_sign = ">"

    # opcode IDs
    # 9: GainMap
    # 1: Rectilinear Warp

    # clip to
    # [0, 2^32 - 1] for OpcodeList1
    # [0, 2^16 - 1] for OpcodeList2
    # [0, 1] for OpcodeList3

    i = 0
    num_opcodes = struct.unpack(endian_sign + "I", opcode_list[i:i + 4])[0]
    i += 4

    opcodes = {}
    for j in range(num_opcodes):
        opcode_id_ = struct.unpack(endian_sign + "I", opcode_list[i:i + 4])[0]
        i += 4
        dng_spec_ver = [struct.unpack(endian_sign + "B", opcode_list[i + k:i + k + 1])[0] for k in range(4)]
        i += 4
        option_bits = struct.unpack(endian_sign + "I", opcode_list[i:i + 4])[0]
        i += 4

        # option bits
        if option_bits & 1 == 1:  # optional/unknown
            pass
        elif option_bits & 2 == 2:  # can be skipped for "preview quality", needed for "full quality"
            pass
        else:
            pass

        opcode_size_bytes = struct.unpack(endian_sign + "I", opcode_list[i:i + 4])[0]
        i += 4

        opcode_data = opcode_list[i:i + 4 * opcode_size_bytes]
        i += 4 * opcode_size_bytes

        # GainMap (lens shading correction map)
        if opcode_id_ == 9:
            opcode_gain_map_data = parse_opcode_gain_map(opcode_data)
            opcode_data = opcode_gain_map_data

        # set opcode object
        opcode = Opcode(id_=opcode_id_, dng_spec_ver=dng_spec_ver, option_bits=option_bits,
                        size_bytes=opcode_size_bytes,
                        data=opcode_data)
        opcodes.update({opcode_id_: opcode})

        return opcodes


def parse_opcode_gain_map(opcode_data):
    endian_sign = ">"  # big
    opcode_dict = {}
    keys = ['top', 'left', 'bottom', 'right', 'plane', 'planes', 'row_pitch', 'col_pitch', 'map_points_v',
            'map_points_h', 'map_spacing_v', 'map_spacing_h', 'map_origin_v', 'map_origin_h', 'map_planes', 'map_gain']
    dtypes = ['L'] * 10 + ['d'] * 4 + ['L'] + ['f']
    dtype_sizes = [4] * 10 + [8] * 4 + [4] * 2  # data type size in bytes
    counts = [1] * 15 + [0]  # 0 count means variable count, depending on map_points_v and map_points_h
    # values = []

    i = 0
    for k in range(len(keys)):
        if counts[k] == 0:  # map_gain
            counts[k] = opcode_dict['map_points_v'] * opcode_dict['map_points_h']

        if counts[k] == 1:
            vals = struct.unpack(endian_sign + dtypes[k], opcode_data[i:i + dtype_sizes[k]])[0]
            i += dtype_sizes[k]
        else:
            vals = []
            for j in range(counts[k]):
                vals.append(struct.unpack(endian_sign + dtypes[k], opcode_data[i:i + dtype_sizes[k]])[0])
                i += dtype_sizes[k]

        opcode_dict[keys[k]] = vals

    opcode_dict['map_gain_2d'] = np.reshape(opcode_dict['map_gain'],
                                            (opcode_dict['map_points_v'], opcode_dict['map_points_h']))

    return opcode_dict
