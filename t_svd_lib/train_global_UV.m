function [U,V] = train_global_UV(im1,ps,N_step)
%TRAIN_GLOBAL_UV 此处显示有关此函数的摘要
addpath('D:\code\denoise\video_denoising\t_svd_lib');
im1 = single(im1);
[H,W,~,~] = size(im1);
t = 1;
% find all reference patches of the first slice
for i = 1:N_step:H-ps+1
    for j = 1:N_step:W-ps+1
        A(:,:,:,t) = im1(i:i+ps-1,j:j+ps-1,:,1);
        t = t + 1;
    end
end

[U,V] = NL_tSVD(A);

end

function [U,V] = NL_tSVD(A)
size_A = size(A);ps = size_A(1);D = size_A(3);
A_F = fft(A,[],3);U = zeros(ps,ps,D);V = U;real_count = floor(D/2) + 1;
for i = 1:real_count
    A_i = A_F(:,:,i,:);
    if(i == 1)
        A_i = real(A_i);
    end
    A1 = my_tenmat(A_i,1);A2 = my_tenmat(A_i,2);
    [Ui,~] = eig(A1*A1');[Vi,~] = eig(A2*A2');
    U(:,:,i) = Ui; V(:,:,i) = Vi;
end

U = com_conj(U); V = com_conj(V);

end

function A  = com_conj(A)
[~,~,D,~] = size(A);
k = 0;
for i = 2:floor(D/2)+1
    A(:,:,D-k,:) = conj(A(:,:,i,:));
    k = k + 1;
end
end

