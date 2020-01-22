function [ output_image, lin_xyz ] = run_pipeline( input_image, metadata, ...
    input_stage, output_stage, cfastr, wb_raw, xyz2cam, crop  )
%RUN_PIPELINE Run a raw image through the pipeline and return the image
% after the specified 'stage'. stages are: 'raw', 'normal', 'wb',
% 'demosaic', 'srgb', 'tone'

% check correct SubIFD
metadatax = metadata;
if metadata.BitDepth ~= 16 % for raw DNG
    if isfield(metadata, 'SubIFDs')
        n_sub = numel(metadata.SubIFDs);
        for k = 1 : n_sub
            if metadata.SubIFDs{k}.BitDepth == 16
                metadatax = metadata.SubIFDs{k};
                break;
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% normalization

if strcmp(input_stage, 'raw')
    
    % cropping area
    %if exist ('crop','var') && strcmp(crop, 'crop')
        %activeArea = GetActiveArea(metadata);
        %input_image = input_image(activeArea(1):activeArea(3), ...
         %                         activeArea(2):activeArea(4));
        cropArea = GetCropArea(metadata); % zero-based index, I guess!
        input_image = input_image(cropArea(1)+1:cropArea(3)+0, ...
                                  cropArea(2)+1:cropArea(4)+0);
    %end
    
    % linearization
    if isfield(metadatax, 'LinearizationTable')
        lt = metadatax.LinearizationTable;
        input_image = lt(input_image + 1); % for zero-based index
    end
    
    if isfield(metadata,'BlackLevel')
        black = metadata.BlackLevel(1);
        saturation = metadata.WhiteLevel;
        if isfield(metadata, 'BlackLevelDeltaV')
            bldv = metadata.BlackLevelDeltaV;
        end
        if isfield(metadata, 'BlackLevelDeltaH')
            bldh = metadata.BlackLevelDeltaH;
        end
    else
        black = metadata.SubIFDs{1,1}.BlackLevel(1);
        saturation = metadata.SubIFDs{1,1}.WhiteLevel;
        try
            bldv = metadata.SubIFDs{1,1}.BlackLevelDeltaV;
        catch
        end
        try
            bldh = metadata.SubIFDs{1,1}.BlackLevelDeltaH;
        catch
        end
    end
    
    input_image=double(input_image);
    
    if exist('bldv', 'var') %&& bldv ~= 0
        bldv = bldv';
        bldv = bldv(1 : size(input_image, 1));
        input_image = input_image - repmat(bldv, 1, size(input_image, 2)); 
    end
    if exist('bldh', 'var') %&& bldh ~= 0
        bldh = bldh(1 : size(input_image, 2));
        input_image = input_image - repmat(bldh, size(input_image, 1), 1); 
    end
    
    lin_bayer = (input_image-black)/(saturation(1)-black);
    lin_bayer = max(0,min(lin_bayer,1));
       
    % go to next stage
    input_image=lin_bayer;
    input_stage='normal';
end

if strcmp(output_stage,'normal')
    output_image=single(lin_bayer);
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% get cfapattern

if ~ exist ('cfastr','var') || strcmp(cfastr, 'none') || isempty(cfastr)
    [ cfaidx, cfastr ] = cfa_pattern(metadata);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% white balancing
if strcmp(input_stage,'normal')
    
    lin_bayer=input_image;
    
    if ~ exist ('wb_raw','var') || strcmp(wb_raw, 'none') || isempty(wb_raw)
        wb_multipliers = metadata.AsShotNeutral; 
    else
        wb_multipliers = wb_raw;
    end
        
    wb_multipliers = wb_multipliers.^-1;
    mask = wbmask(size(lin_bayer,1),size(lin_bayer,2),wb_multipliers,cfastr);
    balanced_bayer = lin_bayer .* mask; 
    balanced_bayer = max(0,min(balanced_bayer,1));
    
    % go to next stage
    input_image=balanced_bayer;
    input_stage='wb';
end

if strcmp(output_stage,'wb')
    output_image=single(balanced_bayer);
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% demosaic

if strcmp(input_stage,'wb')
    
    balanced_bayer = input_image;
    
    temp = uint16(balanced_bayer*(2^16));
    lin_rgb = single(demosaic(temp,cfastr))/(2^16);
    
    % go to next stage
    input_image=lin_rgb;
    input_stage='demosaic';
end

if strcmp(output_stage , 'demosaic')
    output_image = single(lin_rgb);
    return;
end


% Orientation 
input_image = FixOrientation(input_image, metadata);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% color space conversion

if strcmp(input_stage,'demosaic')
    
    lin_rgb = input_image;
    
    if ~ exist ('cam2xyz','var') || strcmp(cam2xyz, 'none') || isempty(cam2xyz)
        xyz2cam9=metadata.ColorMatrix2;
        xyz2cam=reshape(xyz2cam9,3,3)';
        
    end
    
    xyz2cam = xyz2cam ./ repmat(sum(xyz2cam,2),1,3); % Normalize rows to 1
    cam2xyz=xyz2cam^-1;
        
    % rgb2cam = xyz2cam * rgb2xyz; % Assuming previously defined matrices
    % rgb2cam = rgb2cam ./ repmat(sum(rgb2cam,2),1,3); % Normalize rows to 1
    % cam2rgb = rgb2cam^-1;
    % lin_srgb = apply_cmatrix(lin_rgb, cam2rgb);
    % lin_srgb = max(0,min(lin_srgb,1)); % Always keep image clipped b/w 0-1
    
    %%%%%%%%%% color space conversion (xyz2rgb)

    lin_xyz=apply_cmatrix(lin_rgb, cam2xyz);
    lin_xyz = max(0,min(lin_xyz,1)); % clip
    
    srgb=xyz2rgb(lin_xyz); % xyz to srgb 
    
    srgb = max(0,min(srgb,1));
    
    % go to next stage
    input_image=srgb;
    input_stage='srgb';
end

if strcmp(output_stage , 'srgb')
    output_image = single(srgb);
    return;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% tone curve

if strcmp(input_stage,'srgb')
    
    srgb = input_image;
    
%     tc=[];
%     load('tone_curve.mat','tc');
%     x=uint16(srgb*(size(tc,1)-1) + 1);
%     tone=tc(x);
    
    % simple tone curve
    srgb = srgb .^ (1/1.2);
    tone = 3 .* srgb .^ 2 - 2 .* srgb .^ 3;
    
    % go to next stage
    input_image=tone;
    input_stage='tone';
end

if strcmp(output_stage , 'tone')
    output_image = single(tone);
    return;
else
    % unrecognized stage!
    output_image=single(input_image);
end

end

