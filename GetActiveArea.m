function [ activeArea ] = GetActiveArea( metadata )

if isfield(metadata,'ActiveArea')
    activeArea = metadata.activeArea;
elseif isfield(metadata.extra, 'ActiveArea')
    aa = metadata.extra.ActiveArea;
    aacells = strsplit(aa, ' ');
    activeArea = str2num(char(aacells))';
else
    warning('Could not find ActiveArea');
    activeArea = [1, 1, metadata.ImageHeight, metadata.ImageWidth];
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

