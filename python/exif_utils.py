"""
Author(s):
Abdelrahman Abdelhamed

Manual parsing of image file directories (IFDs).
"""


import struct
from fractions import Fraction
from .exif_data_formats import exif_formats


class Ifd:
    def __init__(self):
        self.offset = -1
        self.tags = {}  # <key, tag> dict; tag number will be key.


class Tag:
    def __init__(self):
        self.offset = -1
        self.tag_num = -1
        self.data_format = -1
        self.num_values = -1
        self.values = []


def parse_exif(image_path, verbose=True):
    """
    Parse EXIF tags from a binary file and return IFDs.
    Returned IFDs include EXIF SubIFDs, if any.
    """

    def print_(str_):
        if verbose:
            print(str_)

    ifds = {}  # dict of <offset, Ifd> pairs; using offset to IFD as key.

    with open(image_path, 'rb') as fid:
        fid.seek(0)
        b0 = fid.read(1)
        _ = fid.read(1)
        # byte storage direction (endian):
        # +1: b'M' (big-endian/Motorola)
        # -1: b'I' (little-endian/Intel)
        endian = 1 if b0 == b'M' else -1
        print_("Endian = {}".format(b0))
        endian_sign = "<" if endian == -1 else ">"  # used in struct.unpack
        print_("Endian sign = {}".format(endian_sign))
        _ = fid.read(2)  # 0x002A
        b4_7 = fid.read(4)  # offset to first IFD
        offset_ = struct.unpack(endian_sign + "I", b4_7)[0]
        i = 0
        ifd_offsets = [offset_]
        while len(ifd_offsets) > 0:
            offset_ = ifd_offsets.pop(0)
            # check if IFD at this offset was already parsed before
            if offset_ in ifds:
                continue
            print_("=========== Parsing IFD # {} ===========".format(i))
            ifd_ = parse_exif_ifd(fid, offset_, endian_sign, verbose)
            ifds.update({ifd_.offset: ifd_})
            print_("=========== Finished parsing IFD # {} ===========".format(i))
            i += 1
            # check SubIFDs; zero or more offsets at tag 0x014a
            sub_idfs_tag_num = int('0x014a', 16)
            if sub_idfs_tag_num in ifd_.tags:
                ifd_offsets.extend(ifd_.tags[sub_idfs_tag_num].values)
            # check Exif SUbIDF; usually one offset at tag 0x8769
            exif_sub_idf_tag_num = int('0x8769', 16)
            if exif_sub_idf_tag_num in ifd_.tags:
                ifd_offsets.extend(ifd_.tags[exif_sub_idf_tag_num].values)
    return ifds


def parse_exif_ifd(binary_file, offset_, endian_sign, verbose=True):
    """
    Parse an EXIF IFD.
    """

    def print_(str_):
        if verbose:
            print(str_)

    ifd = Ifd()
    ifd.offset = offset_
    print_("IFD offset = {}".format(ifd.offset))
    binary_file.seek(offset_)
    num_entries = struct.unpack(endian_sign + "H", binary_file.read(2))[0]  # format H = unsigned short
    print_("Number of entries = {}".format(num_entries))
    for t in range(num_entries):
        print_("---------- Tag {} / {} ----------".format(t + 1, num_entries))
        if t == 22:
            ttt = 1
        tag_ = parse_exif_tag(binary_file, endian_sign, verbose)
        ifd.tags.update({tag_.tag_num: tag_})  # supposedly, EXIF tag numbers won't repeat in the same IFD
    # TODO: check for subsequent IFDs by parsing the next 4 bytes immediately after the IFD
    return ifd


def parse_exif_tag(binary_file, endian_sign, verbose=True):
    """
    Parse EXIF tag from a binary file starting from the current file pointer and returns the tag values.
    """

    def print_(str_):
        if verbose:
            print(str_)

    tag = Tag()

    # tag offset
    tag.offset = binary_file.tell()
    print_("Tag offset = {}".format(tag.offset))

    # tag number
    bytes_ = binary_file.read(2)
    tag.tag_num = struct.unpack(endian_sign + "H", bytes_)[0]  # H: unsigned 2-byte short
    print_("Tag number = {} = 0x{:04x}".format(tag.tag_num, tag.tag_num))

    # data format (some value between [1, 12])
    tag.data_format = struct.unpack(endian_sign + "H", binary_file.read(2))[0]  # H: unsigned 2-byte short
    exif_format = exif_formats[tag.data_format]
    print_("Data format = {} = {}".format(tag.data_format, exif_format.name))

    # number of components/values
    tag.num_values = struct.unpack(endian_sign + "I", binary_file.read(4))[0]  # I: unsigned 4-byte integer
    print_("Number of values = {}".format(tag.num_values))

    # total number of data bytes
    total_bytes = tag.num_values * exif_format.size
    print_("Total bytes = {}".format(total_bytes))

    # seek to data offset (if needed)
    data_is_offset = False
    current_offset = binary_file.tell()
    if total_bytes > 4:
        print_("Total bytes > 4; The next 4 bytes are an offset.")
        data_is_offset = True
        data_offset = struct.unpack(endian_sign + "I", binary_file.read(4))[0]
        current_offset = binary_file.tell()
        print_("Current offset = {}".format(current_offset))
        print_("Seeking to data offset = {}".format(data_offset))
        binary_file.seek(data_offset)

    # read values
    # TODO: need to distinguish between numeric and text values?
    if tag.num_values == 1 and total_bytes < 4:
        # special case: data is a single value that is less than 4 bytes inside 4 bytes, take care of endian
        val_bytes = binary_file.read(4)
        # if endian_sign == ">":
        # val_bytes = val_bytes[4 - total_bytes:]
        # else:
        # val_bytes = val_bytes[:total_bytes][::-1]
        val_bytes = val_bytes[:total_bytes]
        tag.values.append(struct.unpack(endian_sign + exif_format.short_name, val_bytes)[0])
    else:
        # read data values one by one
        for k in range(tag.num_values):
            val_bytes = binary_file.read(exif_format.size)
            if exif_format.name == 'unsigned rational':
                tag.values.append(eight_bytes_to_fraction(val_bytes, endian_sign, signed=False))
            elif exif_format.name == 'signed rational':
                tag.values.append(eight_bytes_to_fraction(val_bytes, endian_sign, signed=True))
            else:
                tag.values.append(struct.unpack(endian_sign + exif_format.short_name, val_bytes)[0])
        if total_bytes < 4:
            # special case: multiple values less than 4 bytes in total, inside the 4 bytes; skip the extra bytes
            binary_file.seek(4 - total_bytes, 1)

    if verbose:
        if len(tag.values) > 100:
            print_("Got more than 100 values; printing first 100 only:")
            print_("Tag values = {}".format(tag.values[:100]))
        else:
            print_("Tag values = {}".format(tag.values))
    if tag.data_format == 2:
        print_("Tag values (string) = {}".format(b''.join(tag.values).decode()))

    if data_is_offset:
        # seek back to current position to read the next tag
        print_("Seeking back to current offset = {}".format(current_offset))
        binary_file.seek(current_offset)

    return tag


def get_tag_values_from_ifds(tag_num, ifds):
    """
    Return values of a tag, if found in ifds. Return None otherwise.
    Assuming any tag exists only once in all ifds.
    """
    for key, ifd in ifds.items():
        if tag_num in ifd.tags:
            return ifd.tags[tag_num].values
    return None


def eight_bytes_to_fraction(eight_bytes, endian_sign, signed):
    """
    Convert 8-byte array into a Fraction. Take care of endian and sign.
    """
    if signed:
        num = struct.unpack(endian_sign + "l", eight_bytes[:4])[0]
        den = struct.unpack(endian_sign + "l", eight_bytes[4:])[0]
    else:
        num = struct.unpack(endian_sign + "L", eight_bytes[:4])[0]
        den = struct.unpack(endian_sign + "L", eight_bytes[4:])[0]
    den = den if den != 0 else 1
    return Fraction(num, den)
