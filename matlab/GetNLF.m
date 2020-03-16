function [nlf] = GetNLF(metadata)
% Extracts noise level function (NLF) from DNG metadata

tags = metadata.UnknownTags;
found = 0;

for i = 1 : numel(tags)
    if tags(i).ID == 51041
        nlf = tags(i).Value;
        found = 1;
    end
end

if found 
    if numel(nlf) == 2
        nlf(3:4) = nlf(1:2);
        nlf(5:6) = nlf(1:2);
    end
else 
    disp('NLF not found.');
end

end