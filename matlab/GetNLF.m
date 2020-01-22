function [nlf] = GetNLF(metadata)
% Extracts noise level function (NLF) from DNG metadata

nlf = metadata.UnknownTags(8).Value;

end