function [ cfaidx, cfastr ] = cfa_pattern( metadata )
%CFA_PATTERN Returns the CFA index and CFA pattern from DNG metadata
%(e.g., [0, 1, 1, 2] and [r, g, g, b])


 
cfachar = ['r', 'g', 'b'];

cfaidx = [];

if isfield(metadata,'UnknownTags')
    ut = metadata.UnknownTags;
    if size(ut, 1) >= 2
        cfaidx = ut(2).Value;
    end
elseif isfield(metadata.extra, 'CFAPattern2')
    cfap = metadata.extra.CFAPattern2;
    cfacells = strsplit(cfap, ' ');
    cfaidx = str2num(char(cfacells))';
else
    error('Could not find CFA Pattern');
end

if length(cfaidx) ~= 4
    cfaidx = metadata.SubIFDs{1, 1}.UnknownTags(2).Value;
end

cfaidx = uint8(cfaidx);
cfastr = cfachar(cfaidx + 1);

end

