function [A_r] = reorganize_video(A,factor)
%REORGANIZE_VIDEO 此处显示有关此函数的摘要
%   此处显示详细说明
[H,W,Q,D] = size(A);
A_r = zeros(H,W,Q*factor,D/factor);

for i = 1:size(A_r,4)
    for k = 1:factor
        A_r(:,:,(k-1)*Q+1:k*Q,i) = A(:,:,:,(i-1)*factor+k);
    end
end

end

