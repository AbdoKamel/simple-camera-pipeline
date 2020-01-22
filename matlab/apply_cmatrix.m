function corrected = apply_cmatrix(im, cmatrix)

if size(im, 3) ~= 3
    error('Apply cmatrix to RGB image only.')
end

r = cmatrix(1,1)*im(:,:,1)+cmatrix(1,2)*im(:,:,2)+cmatrix(1,3)*im(:,:,3);
g = cmatrix(2,1)*im(:,:,1)+cmatrix(2,2)*im(:,:,2)+cmatrix(2,3)*im(:,:,3);
b = cmatrix(3,1)*im(:,:,1)+cmatrix(3,2)*im(:,:,2)+cmatrix(3,3)*im(:,:,3);

corrected = cat(3,r,g,b);

end