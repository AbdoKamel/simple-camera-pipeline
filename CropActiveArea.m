function [ croppedRaw ] = CropActiveArea(raw,  metadata )

activeArea = GetActiveArea(metadata);
croppedRaw = raw(activeArea(1):activeArea(3), ...
          activeArea(2):activeArea(4));
   
end

