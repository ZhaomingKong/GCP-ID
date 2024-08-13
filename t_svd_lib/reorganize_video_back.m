function [A] = reorganize_video_back(A_r,factor)
%REORGANIZE_VIDEO �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
[H,W,Q,D] = size(A_r);
A = zeros(H,W,Q/factor,D*factor);

for i = 1:size(A_r,4)
    for k = 1:factor
         A(:,:,:,(i-1)*factor+k) = A_r(:,:,(k-1)*Q/factor+1:k*Q/factor,i);
    end
end

end

