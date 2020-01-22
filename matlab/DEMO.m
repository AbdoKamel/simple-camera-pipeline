% render DNG image file into sRGB image
dngFilename = 'colorchart.dng';
[imRaw, metadata] = Load_Data_and_Metadata_from_DNG(...
    fullfile('data', dngFilename));
imSrgb1 = run_pipeline(imRaw, metadata, 'raw', 'srgb');
imwrite(imSrgb1, fullfile('data', [dngFilename, '_sRGB.png']));

% render .MAT normalized raw image into sRGB image (metadata file required)
normalizedRawFilename = '0001_GT_RAW_010.MAT';
metadataFilename = '0001_METADATA_RAW_010.MAT';
imNormRaw = load(fullfile('data', normalizedRawFilename));
imNormRaw = imNormRaw.x;
metadata = load(fullfile('data', metadataFilename));
metadata = metadata.metadata;
imSrgb2 = run_pipeline(imNormRaw, metadata, 'normal', 'srgb');
imwrite(imSrgb2, fullfile('data', [normalizedRawFilename, '_sRGB.png']));
