import glob
import os
import cv2
import numpy as np

from python.pipeline import run_pipeline


params = {
    'output_stage': 'srgb',  # options: 'normal', 'white_balance', 'demosaic', 'xyz', 'srgb', 'gamma', 'tone'
}

# processing a directory
images_dir = '../data/'
image_paths = glob.glob(os.path.join(images_dir, '*.dng'))
for image_path in image_paths:
    output_image = run_pipeline(image_path, params)
    output_image_path = image_path.replace('.dng', '_processed.jpg')
    output_image = (output_image * 255).astype(np.uint8)
    cv2.imwrite(output_image_path, output_image, [cv2.IMWRITE_JPEG_QUALITY, 100])
