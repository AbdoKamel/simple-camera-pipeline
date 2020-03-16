function [ cropArea ] = GetCropArea( metadata )

% check correct SubIFD
if metadata.BitDepth ~= 16 % for raw DNG
    if isfield(metadata, 'SubIFDs')
        n_sub = numel(metadata.SubIFDs);
        for k = 1 : n_sub
            if metadata.SubIFDs{k}.BitDepth == 16
                metadata = metadata.SubIFDs{k};
                break;
            end
        end
    end
end

if isfield(metadata,'DefaultCropOrigin')
    org = metadata.DefaultCropOrigin;
    csz = metadata.DefaultCropSize;
    act = metadata.ActiveArea;
    cropArea = [org(2:-1:1) + act(1:2), org(2:-1:1) + act(1:2) + csz(2:-1:1)];
elseif isfield(metadata, 'extra')
    if isfield(metadata.extra, 'DefaultCropOrigin')
        aa = metadata.extra.DefaultCropOrigin;
        bb = metadata.extra.DefaultCropSize;
        act = metadata.extra.ActiveArea;
        aacells = strsplit(aa, ' ');
        bbcells = strsplit(bb, ' ');
        actcells = strsplit(act, ' ');
        org = str2num(char(aacells))';
        csz = str2num(char(bbcells))';
        act = str2num(char(actcells))';
        %cropArea = str2num([char(aacells), char(bbcells)])';
        cropArea = [org(2:-1:1) + act(1:2), org(2:-1:1) + act(1:2) + csz(2:-1:1)];
    end
elseif isfield(metadata, 'SubIFDs')
    if ~isempty(metadata.SubIFDs) ...
            && isfield(metadata.SubIFDs{1, 1}, 'DefaultCropOrigin')
        org = metadata.SubIFDs{1, 1}.DefaultCropOrigin;
        csz = metadata.SubIFDs{1, 1}.DefaultCropSize;
        act = metadata.SubIFDs{1, 1}.ActiveArea;
        cropArea = [org(2:-1:1) + act(1:2), org(2:-1:1) + act(1:2) + csz(2:-1:1)];
    end
else
    warning('Could not specify CropArea');
    if isfield(metadata, 'ImageHeight') && isfield(metadata, 'ImageWidth') 
        cropArea = [1, 1, metadata.ImageHeight, metadata.ImageWidth];
    else
        if isfield(metadata, 'Height') && isfield(metadata, 'Width')
            cropArea = [1, 1, metadata.Height, metadata.Width];
        end
    end
end

% if cropArea(4) > metadata.Height
%     cropArea(4) = metadata.Height;
% end
% if cropArea(3) > metadata.Width
%     cropArea(3) = metadata.Width;
% end

% if mod(cropArea(1), 2) ~= 0
%     cropArea(1) = cropArea(1) + 1;
%     cropArea(3) = cropArea(3) + 1;
% end
% if mod(cropArea(2), 2) ~= 0
%     cropArea(2) = cropArea(2) + 1;
%     cropArea(4) = cropArea(4) + 1;
% end

if length(cropArea) ~= 4
    warning('CropArea size is not equal to 4!');
end

end

