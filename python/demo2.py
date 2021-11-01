import glob
import os
import cv2
import numpy as np

from python.pipeline import run_pipeline_v2
from python.pipeline_utils import get_visible_raw_image, get_metadata

params = {
    'input_stage': 'raw',  # options: 'raw', 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
    'output_stage': 'tone',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
    'save_as': 'png',  # options: 'jpg', 'png', 'tif', etc.
    'demosaic_type': 'EA',
    'save_dtype': np.uint8
}

# processing a directory
images_dir = '../data/'
image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
for image_path in image_paths:

    # raw image data
    raw_image = get_visible_raw_image(image_path)

    # metadata
    metadata = get_metadata(image_path)

    # modify WB here
    metadata['as_shot_neutral'] = [1., 1., 1.]

    # render
    output_image = run_pipeline_v2(image_path, params)

    # save
    output_image_path = image_path.replace('.dng', '_{}.'.format(params['output_stage']) + params['save_as'])
    max_val = 2 ** 16 if params['save_dtype'] == np.uint16 else 255
    output_image = (output_image[..., ::-1] * max_val).astype(params['save_dtype'])
    if params['save_as'] == 'jpg':
        cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(output_image_path, output_image)
