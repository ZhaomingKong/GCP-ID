function [denoised_video] = CMStSVD_raw_video_with_local_noise_est_mex(noisy_video, ps, N_step_spatial, N_step_time, SR, maxK, sigma, tau, global_est_nois)
%CVMSTSVD_DENOISING Using MATLAB implementation
%% compute global U and V

[H_origin, W_origin, C, N] =  size(noisy_video);

[U_4D,V_4D] = compute_global_4D(noisy_video,N_step_spatial,ps);

noisy_video_enlarge = enlarge_video(noisy_video, 8);

%% Run noise estimation for the whole frame
im1_frame = noisy_video_enlarge(:,:,:,3);
[H,W,~] = size(im1_frame);
chunk_size = 128;
num_chunk_i = floor(H/chunk_size);
num_chunk_j = floor(W/chunk_size);
noise_est_matrix = zeros(num_chunk_i, num_chunk_j);

for i = 1:num_chunk_i
    i_start = (i-1) * chunk_size + 1;
    if(i == num_chunk_i)
        i_end = H;
    else
        i_end = i*chunk_size;
    end
    for j = 1:num_chunk_j
        j_start = (j-1) * chunk_size + 1;
        if(j == num_chunk_j)
            j_end = W;
        else
            j_end = j*chunk_size;
        end
        noisy_img_chunk_ij = im1_frame(i_start:i_end, j_start:j_end,:);
        Noise_levels_ij = Multi_channel_NoiseEstimation(noisy_img_chunk_ij,16);
        Noise_levels_avg_ij = mean(Noise_levels_ij(:));
        sigma_ij = select_sigma_CRVD(Noise_levels_avg_ij);
        noise_est_matrix(i,j) = sigma_ij;
    end
end

%% Denoising using GCP

U_4D = fft(U_4D,[],3); V_4D = fft(V_4D,[],3);

% Parameters
[H,W,~,D] = size(noisy_video_enlarge);
info1 = int32([H,W,D]);
info2 = [ps,N_step_spatial,N_step_time,SR,maxK];
info2 = int32(info2);
modified = 1;
sigma = mean(noise_est_matrix(:))/255;
info3 = [tau,modified,sigma];


% Start denoising
if(global_est_nois == 1)
    tic;denoised_MSTSVD = GCP_VD_raw_global_est_noise_search_full_patch(single(noisy_video_enlarge),single(U_4D),single(V_4D),info1,info2,single(info3));toc;
    denoised_video_enlarge = mat_ten(denoised_MSTSVD,1,size(noisy_video_enlarge));
    denoised_video = denoised_video_enlarge(1:H_origin, 1:W_origin, :, :);
elseif(global_est_nois == 0)
    tic;denoised_MSTSVD = color_video_denoising_global_Eigen_local_est_noise_RGGB(single(noisy_video_enlarge),single(U_4D),single(V_4D),info1,info2,single(info3),single(noise_est_matrix));toc;
    denoised_video_enlarge = mat_ten(denoised_MSTSVD,1,size(noisy_video_enlarge));
    denoised_video = denoised_video_enlarge(1:H_origin, 1:W_origin, :, :);  
end

end



%% Other functions

function video_reshape = enlarge_video(video, factor)
[H,W,C,N] = size(video);
H_new = (ceil(H/factor)+1) * factor;
W_new = (ceil(W/factor)+1) * factor;
video_reshape = zeros(H_new, W_new, C, N);
video_reshape(1:H, 1:W, :,:) = video;
video_reshape = single(video_reshape);

end


% Decide patch mode
function patch_mode = decide_mode(refpatch, divide_ratio)

patchR = refpatch(:,:,1);
patchG = refpatch(:,:,2);
patchB = refpatch(:,:,3);

R_sum = sum(patchR(:));
G_sum = sum(patchG(:));
B_sum = sum(patchB(:));

R_norm = norm(patchR(:));
G_norm = norm(patchG(:));
B_norm = norm(patchB(:));

if G_norm > (B_norm/divide_ratio)  && G_norm > (R_norm/divide_ratio)
    patch_mode = 'green_extra';
else
    patch_mode = 'normal';
end

end

% Compute Global 3D transforms U and V
function [U,V] = compute_global_3D(im1,N_step,ps)
[H,W,D,N] = size(im1);
count = 0;
for t = 1:2:N
    for i=1:N_step:H-ps+1 %why start from i=104?
        for j=1:N_step:W-ps+1 %why start from j=49?
            count = count + 1;
            global_indice(count,:) = [i,j,t];
        end
    end
end
A = zeros(ps,ps,D,count,'single');
for k=1:count
    yindex = global_indice(k,1);
    xindex = global_indice(k,2);
    tindex = global_indice(k,3);
    A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:,tindex);
end

[U,V]=NL_tSVD(A);

end

% Compute Global 4D transforms U and V
function [U,V] = compute_global_4D(im1,N_step,ps)
[H,W,D,T] = size(im1);
count = 0;
for t = 1:2:T
    for i=1:N_step:H-ps+1 %why start from i=104?
        for j=1:N_step:W-ps+1 %why start from j=49?
            count = count + 1;
            global_indice(count,:) = [i,j,t];
        end
    end
end

A = zeros(ps,ps,D,count,'single');
for k=1:count
    yindex = global_indice(k,1);
    xindex = global_indice(k,2);
    tindex = global_indice(k,3);
    A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:,tindex);
end

% A_learn = zeros(ps,ps,4,count,'single');
% A_learn(:,:,1,:) = A(:,:,1,:);
% A_learn(:,:,2,:) = A(:,:,2,:);
% A_learn(:,:,3,:) = A(:,:,2,:);
% A_learn(:,:,4,:) = A(:,:,3,:);

[U,V]=NL_tSVD(A);
end

% Compute local transforms U and V
function [U,V] = NL_tSVD(A)
size_A = size(A);ps = size_A(1);N = size_A(end); D = size_A(3);
A_F = fft(A,[],3);U = zeros(ps,ps,D);V = U;
for i = 1:D
    A_i = A_F(:,:,i,:);
    if(i == 1)
        A_i = real(A_i);
    end
    A1 = my_tenmat(A_i,1);A2 = my_tenmat(A_i,2);
    [Ui,~] = eig(A1*A1');[Vi,~] = eig(A2*A2');
    U(:,:,i) = Ui; V(:,:,i) = Vi;
end

U = ifft(U,[],3); V = ifft(V,[],3);

end

% Perform the hard thresholding operation
function A_thre = threshold(A,U,V,sigma,tau)
size_A = size(A);ps = size_A(1);D = size_A(3);count = size_A(end);
A_thre = zeros(size_A);real_count = floor(D/2)+1;
% hard-threshold parameter
coeff_threshold = tau*sigma*sqrt(2*log(ps*ps*D*count));
for i = 1:real_count
    Ai = A(:,:,i,:);
    S1 = my_ttm(Ai,U(:,:,i),1,'t'); S = my_ttm(S1,V(:,:,i),2,'t');
    % hard-threshold
    S(abs(S(:)) < coeff_threshold) = 0;
    A1 = my_ttm(S,U(:,:,i),1,'nt'); A_f = my_ttm(A1,V(:,:,i),2,'nt');
    A_thre(:,:,i,:) = A_f;
end

A_thre = com_conj(A_thre);

end


% Comduct conjugate operation
function A  = com_conj(A)
[~,~,D,~] = size(A);
k = 0;
for i = 2:floor(D/2)+1
    A(:,:,D-k,:) = conj(A(:,:,i,:));
    k = k + 1;
end
end

% Compute local group-level representation
function [U4] = train_U4(A)

[num_patch,~] = size(A);
sum_A = sum(A,1);
average_A = sum_A/num_patch;
A_hat = A - average_A;
[U4,~] = eig(A_hat*A_hat');

end


