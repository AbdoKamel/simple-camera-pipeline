"""
Author(s):
Abdelrahman Abdelhamed

Camera pipeline utilities.
"""

import os
from fractions import Fraction

import cv2
import numpy as np
import exifread
# from exifread import Ratio
from exifread.utils import Ratio
import rawpy
from scipy.io import loadmat
from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
import struct

from .dng_opcode import parse_opcode_lists
from .exif_data_formats import exif_formats
from .exif_utils import parse_exif_tag, parse_exif, get_tag_values_from_ifds


def get_visible_raw_image(image_path):
    raw_image = rawpy.imread(image_path).raw_image_visible.copy()
    # raw_image = rawpy.imread(image_path).raw_image.copy()
    return raw_image


def get_image_tags(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)
    return tags


def get_image_ifds(image_path):
    ifds = parse_exif(image_path, verbose=False)
    return ifds


def get_metadata(image_path):
    metadata = {}
    tags = get_image_tags(image_path)
    ifds = get_image_ifds(image_path)
    metadata['linearization_table'] = get_linearization_table(tags, ifds)
    metadata['black_level'] = get_black_level(tags, ifds)
    metadata['white_level'] = get_white_level(tags, ifds)
    metadata['cfa_pattern'] = get_cfa_pattern(tags, ifds)
    metadata['as_shot_neutral'] = get_as_shot_neutral(tags, ifds)
    color_matrix_1, color_matrix_2 = get_color_matrices(tags, ifds)
    metadata['color_matrix_1'] = color_matrix_1
    metadata['color_matrix_2'] = color_matrix_2
    metadata['orientation'] = get_orientation(tags, ifds)
    metadata['noise_profile'] = get_noise_profile(tags, ifds)
    # ...

    # opcode lists
    metadata['opcode_lists'] = parse_opcode_lists(ifds)

    # fall back to default values, if necessary
    if metadata['black_level'] is None:
        metadata['black_level'] = 0
        print("Black level is None; using 0.")
    if metadata['white_level'] is None:
        metadata['white_level'] = 2 ** 16
        print("White level is None; using 2 ** 16.")
    if metadata['cfa_pattern'] is None:
        metadata['cfa_pattern'] = [0, 1, 1, 2]
        print("CFAPattern is None; using [0, 1, 1, 2] (RGGB)")
    if metadata['as_shot_neutral'] is None:
        metadata['as_shot_neutral'] = [1, 1, 1]
        print("AsShotNeutral is None; using [1, 1, 1]")
    if metadata['color_matrix_1'] is None:
        metadata['color_matrix_1'] = np.eye(3)
        print("ColorMatrix1 is None; using\n" + str(metadata['color_matrix_1']))
    if metadata['color_matrix_2'] is None:
        metadata['color_matrix_2'] = np.eye(3)
        print("ColorMatrix2 is None; using\n" + str(metadata['color_matrix_2']))
    if metadata['orientation'] is None:
        metadata['orientation'] = 0
        print("Orientation is None; using 0.")
    # ...
    return metadata


def get_linearization_table(tags, ifds):
    possible_keys = ['Image Tag 0xC618', 'Image Tag 50712', 'LinearizationTable', 'Image LinearizationTable']
    return get_values(tags, possible_keys)


def get_black_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61A', 'Image Tag 50714', 'BlackLevel', 'Image BlackLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Black level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50714, ifds)
    return vals


def get_white_level(tags, ifds):
    possible_keys = ['Image Tag 0xC61D', 'Image Tag 50717', 'WhiteLevel', 'Image WhiteLevel']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("White level not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(50717, ifds)
    return vals


def get_cfa_pattern(tags, ifds):
    possible_keys = ['CFAPattern', 'Image CFAPattern']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("CFAPattern not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(33422, ifds)
    return vals


def get_as_shot_neutral(tags, ifds):
    possible_keys = ['Image Tag 0xC628', 'Image Tag 50728', 'AsShotNeutral', 'Image AsShotNeutral']
    return get_values(tags, possible_keys)


def get_color_matrices(tags, ifds):
    possible_keys_1 = ['Image Tag 0xC621', 'Image Tag 50721', 'ColorMatrix1', 'Image ColorMatrix1']
    color_matrix_1 = get_values(tags, possible_keys_1)
    possible_keys_2 = ['Image Tag 0xC622', 'Image Tag 50722', 'ColorMatrix2', 'Image ColorMatrix2']
    color_matrix_2 = get_values(tags, possible_keys_2)
    return color_matrix_1, color_matrix_2


def get_orientation(tags, ifds):
    possible_tags = ['Orientation', 'Image Orientation']
    return get_values(tags, possible_tags)


def get_noise_profile(tags, ifds):
    possible_keys = ['Image Tag 0xC761', 'Image Tag 51041', 'NoiseProfile', 'Image NoiseProfile']
    vals = get_values(tags, possible_keys)
    if vals is None:
        # print("Noise profile not found in exifread tags. Searching IFDs.")
        vals = get_tag_values_from_ifds(51041, ifds)
    return vals


def get_values(tags, possible_keys):
    values = None
    for key in possible_keys:
        if key in tags.keys():
            values = tags[key].values
    return values


def normalize(raw_image, black_level, white_level):
    if type(black_level) is list and len(black_level) == 1:
        black_level = float(black_level[0])
    if type(white_level) is list and len(white_level) == 1:
        white_level = float(white_level[0])
    black_level_mask = black_level
    if type(black_level) is list and len(black_level) == 4:
        if type(black_level[0]) is Ratio:
            black_level = ratios2floats(black_level)
        black_level_mask = np.zeros(raw_image.shape)
        idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
        step2 = 2
        for i, idx in enumerate(idx2by2):
            black_level_mask[idx[0]::step2, idx[1]::step2] = black_level[i]
    normalized_image = raw_image.astype(np.float32) - black_level_mask
    # if some values were smaller than black level
    normalized_image[normalized_image < 0] = 0
    normalized_image = normalized_image / (white_level - black_level_mask)
    return normalized_image


def ratios2floats(ratios):
    floats = []
    for ratio in ratios:
        floats.append(float(ratio.num) / ratio.den)
    return floats


def lens_shading_correction(raw_image, gain_map_opcode, bayer_pattern, gain_map=None, clip=True):
    """
    Apply lens shading correction map.
    :param raw_image: Input normalized (in [0, 1]) raw image.
    :param gain_map_opcode: Gain map opcode.
    :param bayer_pattern: Bayer pattern (RGGB, GRBG, ...).
    :param gain_map: Optional gain map to replace gain_map_opcode. 1 or 4 channels in order: R, Gr, Gb, and B.
    :param clip: Whether to clip result image to [0, 1].
    :return: Image with gain map applied; lens shading corrected.
    """

    if gain_map is None and gain_map_opcode:
        gain_map = gain_map_opcode.data['map_gain_2d']

    # resize gain map, make it 4 channels, if needed
    gain_map = cv2.resize(gain_map, dsize=(raw_image.shape[1] // 2, raw_image.shape[0] // 2),
                          interpolation=cv2.INTER_LINEAR)
    if len(gain_map.shape) == 2:
        gain_map = np.tile(gain_map[..., np.newaxis], [1, 1, 4])

    if gain_map_opcode:
        # TODO: consider other parameters

        top = gain_map_opcode.data['top']
        left = gain_map_opcode.data['left']
        bottom = gain_map_opcode.data['bottom']
        right = gain_map_opcode.data['right']
        rp = gain_map_opcode.data['row_pitch']
        cp = gain_map_opcode.data['col_pitch']

        gm_w = right - left
        gm_h = bottom - top

        # gain_map = cv2.resize(gain_map, dsize=(gm_w, gm_h), interpolation=cv2.INTER_LINEAR)

        # TODO
        # if top > 0:
        #     pass
        # elif left > 0:
        #     left_col = gain_map[:, 0:1]
        #     rep_left_col = np.tile(left_col, [1, left])
        #     gain_map = np.concatenate([rep_left_col, gain_map], axis=1)
        # elif bottom < raw_image.shape[0]:
        #     pass
        # elif right < raw_image.shape[1]:
        #     pass

    result_image = raw_image.copy()

    # one channel
    # result_image[::rp, ::cp] *= gain_map[::rp, ::cp]

    # per bayer channel
    upper_left_idx = [[0, 0], [0, 1], [1, 0], [1, 1]]
    bayer_pattern_idx = np.array(bayer_pattern)
    # blue channel index --> 3
    bayer_pattern_idx[bayer_pattern_idx == 2] = 3
    # second green channel index --> 2
    if bayer_pattern_idx[3] == 1:
        bayer_pattern_idx[3] = 2
    else:
        bayer_pattern_idx[2] = 2
    for c in range(4):
        i0 = upper_left_idx[c][0]
        j0 = upper_left_idx[c][1]
        result_image[i0::2, j0::2] *= gain_map[:, :, bayer_pattern_idx[c]]

    if clip:
        result_image = np.clip(result_image, 0.0, 1.0)

    return result_image


def white_balance(normalized_image, as_shot_neutral, cfa_pattern):
    if type(as_shot_neutral[0]) is Ratio:
        as_shot_neutral = ratios2floats(as_shot_neutral)
    idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
    step2 = 2
    white_balanced_image = np.zeros(normalized_image.shape)
    for i, idx in enumerate(idx2by2):
        idx_y = idx[0]
        idx_x = idx[1]
        white_balanced_image[idx_y::step2, idx_x::step2] = \
            normalized_image[idx_y::step2, idx_x::step2] / as_shot_neutral[cfa_pattern[i]]
    white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
    return white_balanced_image


def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
    # using opencv edge-aware demosaicing
    if alg_type != '':
        alg_type = '_' + alg_type
    if output_channel_order == 'BGR':
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
            print("CFA pattern not identified.")
    else:  # RGB
        if cfa_pattern == [0, 1, 1, 2]:  # RGGB
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
        elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
        elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
        elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
        else:
            opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
            print("CFA pattern not identified.")
    return opencv_demosaic_flag


def demosaic(white_balanced_image, cfa_pattern, output_channel_order='BGR', alg_type='VNG'):
    """
    Demosaic a Bayer image.
    :param white_balanced_image:
    :param cfa_pattern:
    :param output_channel_order:
    :param alg_type: algorithm type. options: '', 'EA' for edge-aware, 'VNG' for variable number of gradients
    :return: Demosaiced image
    """
    if alg_type == 'VNG':
        max_val = 255
        wb_image = (white_balanced_image * max_val).astype(dtype=np.uint8)
    else:
        max_val = 16383
        wb_image = (white_balanced_image * max_val).astype(dtype=np.uint16)

    if alg_type in ['', 'EA', 'VNG']:
        opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
        demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
    elif alg_type == 'menon2007':
        cfa_pattern_str = "".join(["RGB"[i] for i in cfa_pattern])
        demosaiced_image = demosaicing_CFA_Bayer_Menon2007(wb_image, pattern=cfa_pattern_str)

    demosaiced_image = demosaiced_image.astype(dtype=np.float32) / max_val

    return demosaiced_image


def apply_color_space_transform(demosaiced_image, color_matrix_1, color_matrix_2):
    if type(color_matrix_1[0]) is Ratio:
        color_matrix_1 = ratios2floats(color_matrix_1)
    if type(color_matrix_2[0]) is Ratio:
        color_matrix_2 = ratios2floats(color_matrix_2)
    xyz2cam1 = np.reshape(np.asarray(color_matrix_1), (3, 3))
    xyz2cam2 = np.reshape(np.asarray(color_matrix_2), (3, 3))
    # normalize rows (needed?)
    xyz2cam1 = xyz2cam1 / np.sum(xyz2cam1, axis=1, keepdims=True)
    xyz2cam2 = xyz2cam2 / np.sum(xyz2cam1, axis=1, keepdims=True)
    # inverse
    cam2xyz1 = np.linalg.inv(xyz2cam1)
    cam2xyz2 = np.linalg.inv(xyz2cam2)
    # for now, use one matrix  # TODO: interpolate btween both
    # simplified matrix multiplication
    xyz_image = cam2xyz1[np.newaxis, np.newaxis, :, :] * demosaiced_image[:, :, np.newaxis, :]
    xyz_image = np.sum(xyz_image, axis=-1)
    xyz_image = np.clip(xyz_image, 0.0, 1.0)
    return xyz_image


def transform_xyz_to_srgb(xyz_image):
    # srgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
    #                      [0.2126729, 0.7151522, 0.0721750],
    #                      [0.0193339, 0.1191920, 0.9503041]])

    # xyz2srgb = np.linalg.inv(srgb2xyz)

    xyz2srgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                         [-0.9692660, 1.8760108, 0.0415560],
                         [0.0556434, -0.2040259, 1.0572252]])

    # normalize rows (needed?)
    xyz2srgb = xyz2srgb / np.sum(xyz2srgb, axis=-1, keepdims=True)

    srgb_image = xyz2srgb[np.newaxis, np.newaxis, :, :] * xyz_image[:, :, np.newaxis, :]
    srgb_image = np.sum(srgb_image, axis=-1)
    srgb_image = np.clip(srgb_image, 0.0, 1.0)
    return srgb_image


def fix_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW

    if type(orientation) is list:
        orientation = orientation[0]

    if orientation == 1:
        pass
    elif orientation == 2:
        image = cv2.flip(image, 0)
    elif orientation == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 4:
        image = cv2.flip(image, 1)
    elif orientation == 5:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orientation == 6:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 7:
        image = cv2.flip(image, 0)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 8:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return image


def reverse_orientation(image, orientation):
    # 1 = Horizontal(normal)
    # 2 = Mirror horizontal
    # 3 = Rotate 180
    # 4 = Mirror vertical
    # 5 = Mirror horizontal and rotate 270 CW
    # 6 = Rotate 90 CW
    # 7 = Mirror horizontal and rotate 90 CW
    # 8 = Rotate 270 CW
    rev_orientations = np.array([1, 2, 3, 4, 5, 8, 7, 6])
    return fix_orientation(image, rev_orientations[orientation - 1])


def apply_gamma(x):
    return x ** (1.0 / 2.2)


def apply_tone_map(x):
    # simple tone curve
    # return 3 * x ** 2 - 2 * x ** 3

    # tone_curve = loadmat('tone_curve.mat')
    tone_curve = loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tone_curve.mat'))
    tone_curve = tone_curve['tc']
    x = np.round(x * (len(tone_curve) - 1)).astype(int)
    tone_mapped_image = np.squeeze(tone_curve[x])
    return tone_mapped_image


def raw_rgb_to_cct(rawRgb, xyz2cam1, xyz2cam2):
    """Convert raw-RGB triplet to corresponding correlated color temperature (CCT)"""
    pass
    # pxyz = [.5, 1, .5]
    # loss = 1e10
    # k = 1
    # while loss > 1e-4:
    #     cct = XyzToCct(pxyz)
    #     xyz = RawRgbToXyz(rawRgb, cct, xyz2cam1, xyz2cam2)
    #     loss = norm(xyz - pxyz)
    #     pxyz = xyz
    #     fprintf('k = %d, loss = %f\n', [k, loss])
    #     k = k + 1
    # end
    # temp = cct
