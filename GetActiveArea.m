function [ activeArea ] = GetActiveArea( metadata )

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

if isfield(metadata,'ActiveArea')
    activeArea = metadata.ActiveArea;
elseif isfield(metadata.extra, 'ActiveArea')
    aa = metadata.extra.ActiveArea;
    aacells = strsplit(aa, ' ');
    activeArea = str2num(char(aacells))';
elseif isfield(metadata.SubIFDs)
    if ~isempty(metadata.SubIFDs) ...
            && isfield(metadata.SubIFDs{1, 1}, 'ActiveArea')
        activeArea = metadata.SubIFDs{1, 1}.ActiveArea;
    end
else
    warning('Could not specify ActiveArea');
    if isfield(metadata, 'ImageHeight') && isfield(metadata, 'ImageWidth') 
        activeArea = [1, 1, metadata.ImageHeight, metadata.ImageWidth];
    end
end

if mod(activeArea(1), 2) == 0
    activeArea(1) = activeArea(1) + 1;
end
if mod(activeArea(2), 2) == 0
    activeArea(2) = activeArea(2) + 1;
end

if length(activeArea) ~= 4
    warning('Active area size is not equal to 4!');
end

end

