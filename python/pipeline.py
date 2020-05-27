import numpy as np
from python.pipeline_utils import get_visible_raw_image, get_metadata, normalize, white_balance, demosaic, \
    apply_color_space_transform, transform_xyz_to_srgb, apply_gamma, apply_tone_map, fix_orientation


def run_pipeline(image_path, params):
    # raw image data
    raw_image = get_visible_raw_image(image_path)

    # metadata
    metadata = get_metadata(image_path)

    # linearization
    linearization_table = metadata['linearization_table']
    if linearization_table is not None:
        print('Linearization table found. Not handled.')
        # TODO

    normalized_image = normalize(raw_image, metadata['black_level'], metadata['white_level'])

    if params['output_stage'] == 'normal':
        return normalized_image

    white_balanced_image = white_balance(normalized_image, metadata['as_shot_neutral'], metadata['cfa_pattern'])

    if params['output_stage'] == 'white_balance':
        return white_balanced_image

    demosaiced_image = demosaic(white_balanced_image, metadata['cfa_pattern'], output_channel_order='BGR',
                                alg_type=params['demosaic_type'])

    # fix image orientation, if needed
    demosaiced_image = fix_orientation(demosaiced_image, metadata['orientation'])

    if params['output_stage'] == 'demosaic':
        return demosaiced_image

    xyz_image = apply_color_space_transform(demosaiced_image, metadata['color_matrix_1'], metadata['color_matrix_2'])

    if params['output_stage'] == 'xyz':
        return xyz_image

    srgb_image = transform_xyz_to_srgb(xyz_image)

    if params['output_stage'] == 'srgb':
        return srgb_image

    gamma_corrected_image = apply_gamma(srgb_image)

    if params['output_stage'] == 'gamma':
        return gamma_corrected_image

    tone_mapped_image = apply_tone_map(gamma_corrected_image)
    if params['output_stage'] == 'tone':
        return tone_mapped_image

    output_image = None
    return output_image
