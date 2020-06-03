# simple-camera-pipeline
A simple and light-weight camera image processing pipeline implemented in MATLAB and Python.

Start by running:

Matlab: [matlab/demo.m](matlab/demo.m)

Python: [python/demo.py](python/demo.py)

Python requirements ([python/requirements.txt](python/requirements.txt)) -- _other versions may work_
```
numpy==1.17.2
scipy==1.3.1
opencv-python==4.1.1.26
rawpy==0.14.0
exifread==2.1.2
colour-demosaicing==0.1.5
```

This is the code used to render the sRGB images from the Raw-RGB images of the [Smartphone Image Denoising Dataset (SIDD)](https://www.eecs.yorku.ca/~kamel/sidd/).

This code is helpful for participants of the real image denoising challenges on CodaLab:

[NTIRE 2020 Real Image Denoising Challenge - Track 1: rawRGB](https://competitions.codalab.org/competitions/22230)

[NTIRE 2020 Real Image Denoising Challenge - Track 2: sRGB](https://competitions.codalab.org/competitions/22231)

[NTIRE 2019 Real Image Denoising Challenge - Track 1: Raw-RGB](https://competitions.codalab.org/competitions/21258)

[NTIRE 2019 Real Image Denoising Challenge - Track 2: sRGB](https://competitions.codalab.org/competitions/21266)

### Paper
Abdelhamed, A., Lin, S., & Brown, M. S. (2018). A High-Quality Denoising Dataset for Smartphone Cameras. In 2018 {IEEE}/{CVF} Conference on Computer Vision and Pattern Recognition. {IEEE}. Retrieved from https://doi.org/10.1109%2Fcvpr.2018.00182

Enjoy!
