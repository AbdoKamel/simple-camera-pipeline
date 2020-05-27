import glob
import os
import cv2
import numpy as np

from python.pipeline import run_pipeline

params = {
    'output_stage': 'tone',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
    'save_as': 'jpg',  # options: 'jpg', 'png', 'tif', etc.
    'demosaic_type': 'menon2007'  # options: '' for simple interpolation, 'EA' for edge-aware,
    # 'VNG' for variable number of gradients, 'menon2007' for Menon's algorithm
}

# processing a directory
images_dir = '../data/'
image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
for image_path in image_paths:
    print("processing {}".format(image_path))
    for dem_type in ['EA', 'VNG', 'menon2007']:  # '',
        print("dem_type = {}".format(dem_type))
        output_image = run_pipeline(image_path, params)
        output_image_path = image_path.replace('.dng', '_' + dem_type + '.' + params['save_as'])
        output_image = (output_image * 255).astype(np.uint8)
        if params['save_as'] == 'jpg':
            cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(output_image_path, output_image)
