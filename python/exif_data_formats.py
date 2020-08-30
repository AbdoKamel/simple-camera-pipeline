class ExifFormat:
    def __init__(self, id, name, size, short_name):
        self.id = id
        self.name = name
        self.size = size
        self.short_name = short_name  # used with struct.unpack()


exif_formats = {
    1: ExifFormat(1, 'unsigned byte', 1, 'B'),
    2: ExifFormat(2, 'ascii string', 1, 's'),
    3: ExifFormat(3, 'unsigned short', 2, 'H'),
    4: ExifFormat(4, 'unsigned long', 4, 'L'),
    5: ExifFormat(5, 'unsigned rational', 8, ''),
    6: ExifFormat(6, 'signed byte', 1, 'b'),
    7: ExifFormat(7, 'undefined', 1, 'B'),  # consider `undefined` as `unsigned byte`
    8: ExifFormat(8, 'signed short', 2, 'h'),
    9: ExifFormat(9, 'signed long', 4, 'l'),
    10: ExifFormat(10, 'signed rational', 8, ''),
    11: ExifFormat(11, 'single float', 4, 'f'),
    12: ExifFormat(12, 'double float', 8, 'd'),
}
