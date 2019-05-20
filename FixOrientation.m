function [fixedImage] = FixOrientation(image, metadata)
%FIXORIENTATION Fix the image's orientation

    %     0x0112	Orientation	int16u	IFD0	
    %     1 = Horizontal (normal) 
    %     2 = Mirror horizontal 
    %     3 = Rotate 180 CW
    %     4 = Mirror vertical 
    %     5 = Mirror horizontal and rotate 270 CW 
    %     6 = Rotate 90 CW 
    %     7 = Mirror horizontal and rotate 90 CW 
    %     8 = Rotate 270 CW
    
    fixedImage = image;
    if isfield(metadata, 'Orientation')
        switch metadata.Orientation
            case 2
                fixedImage = flip(image, 2);
            case 3
                fixedImage = imrotate(image, -180);
            case 4
                fixedImage = flip(image, 1);
            case 5
                fixedImage = flip(image, 2);
                fixedImage = imrotate(fixedImage, -270);
            case 6
                fixedImage = imrotate(image, -90);
            case 7
                fixedImage = flip(image, 2);
                fixedImage = imrotate(fixedImage, -90);
            case 8
                fixedImage = imrotate(image, -270);
        end
    end
    
end

