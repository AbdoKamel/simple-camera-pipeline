function [ raw_data, meta_data ] = Load_Data_and_Metadata_from_DNG( image_name )
%Load_Data_and_Metadata_from_DNG Reads and returns raw image data and 
% metadata from DNG file "image_name"

    t = Tiff(char(image_name), 'r');
    if t.getTag('BitsPerSample') <= 8 % raw should be at least 10 bps
        try
            offsets = getTag(t, 'SubIFD');
            setSubDirectory(t, offsets(1));
        catch 
        end
    end
    raw_data = read(t);
    close(t);
    meta_data = imfinfo(char(image_name));
    
    
end

