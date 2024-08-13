function [denoised_img] = CMStSVD_raw_est_CNN(im1, im, Noise_est_mtx, ps, SR, maxK, N_step, modified, tau, global_learning)

%% settings

[H,W,D] = size(im1);
divide_size = 128;
H_steps = ceil(H/divide_size);
W_steps = ceil(W/divide_size);
denoised_img = zeros(H,W,D);

use_mex = 1;
noise_lvl_vectors = determin_noise_lvl_vectors(im1, ps, N_step, Noise_est_mtx);
% A more adaptive scheme

if(use_mex == 0)
    Noisy_RGGB_imRaw = convert_from_raw(im1);
    [im_denoised_RGGB,~,~] = CMStSVD_raw_adaptive_modified(single(Noisy_RGGB_imRaw), single(Noisy_RGGB_imRaw), ps, SR, Noise_est_mtx, maxK, N_step, modified, tau, global_learning);
    denoised_img = convert_raw_to_origin_raw(im_denoised_RGGB);
end

if(use_mex == 1)
    Noisy_RGGB_imRaw = convert_from_raw(im1);
    [U_4D,V_4D] = compute_global_4D(Noisy_RGGB_imRaw,N_step,ps);
    U_4D = fft(U_4D,[],3);
    V_4D = fft(V_4D,[],3);
    [H,W,D] = size(Noisy_RGGB_imRaw);
    info1 = [H,W,D];
    info2 = [ps,N_step,SR,maxK];
    noise_lvl_ij = mean(Noise_est_mtx(:));
    sigma = noise_lvl_ij/255;
    info3 = [1.1,sigma,1];
    
    [similar_indice] = color_denoising_search_patch_raw(single(Noisy_RGGB_imRaw),int32(info1),int32(info2), single(info3));
    im2 = GCP_ID_raw_global_noise_est(single(Noisy_RGGB_imRaw),int32(similar_indice),single(U_4D),single(V_4D),int32(info1),int32(info2),single(info3));
%     im2 = GCP_ID_raw_local_noise_est_with_neighbours(single(Noisy_RGGB_imRaw),int32(similar_indice),single(U_4D),single(V_4D),int32(info1),int32(info2),single(info3), single(noise_lvl_vectors));
    im_denoised_RGGB = mat_ten(im2,1,size(Noisy_RGGB_imRaw));
    denoised_img = convert_raw_to_origin_raw(im_denoised_RGGB);
end

end

function noise_lvl_vectors = determin_noise_lvl_vectors(img, ps, N_step, noise_lvl_mtx)

[H,W,~] = size(img);
step_size = 128;
count_num = 1;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        
        noise_lvl_ij = determine_local_noise_lvl_by_neighbours(i, j, step_size, noise_lvl_mtx);
%         noise_lvl_ij = mean(Noise_lvl_mtx(:));
        noiselevel = noise_lvl_ij/255;
        noise_lvl_vectors(count_num) = noiselevel;
        count_num = count_num + 1; 
    end
end

end

function noise_lvl_ij_avg = determine_local_noise_lvl_by_neighbours(i, j, step_size, noise_lvl_mtx)

[H,W] = size(noise_lvl_mtx);

idx_i = ceil(i/step_size);
idx_j = ceil(j/step_size);

sr_top = max([idx_i-1 1]);
sr_left = max([idx_j-1 1]);
sr_right = min([idx_j+1 W]);
sr_bottom = min([idx_i+1 H]);

noise_sum = 0; noise_count = 0;
for i1 = sr_top:sr_bottom
    for j1 = sr_left:sr_right
        noise_lvl_ij = noise_lvl_mtx(i1, j1);
        noise_sum = noise_sum + noise_lvl_ij;
        noise_count = noise_count + 1;
    end
end

noise_lvl_ij_avg = noise_sum/noise_count;


end



function [U,V] = compute_global_4D(im1,N_step,ps)
[H,W,D] = size(im1);
count = 0;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        count = count + 1;
        global_indice(count,:) = [i,j];
    end
end
A = zeros(ps,ps,D,count,'single');
for k=1:count
    yindex = global_indice(k,1);
    xindex = global_indice(k,2);
    A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
end

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

function [psnr_img, ssim_img, FSIM_img] = cal_img_metrics(origin_test, noisy_test)

im2=single(origin_test)*255;im=single(noisy_test)*255; [H,W,D] = size(im);

mse = sum((im2(:)-im(:)).^2)/(H*W*D);

psnr_img = 10*log10(255*255/mse);

ssim_img = cal_ssim(im2,im,0,0);

[~,FSIM_img] = FeatureSIM(im2,im);

end

function [PSNR, SSIM] = calculate_index(denoised, clean)
PSNR = 10*log10(1/mean((clean(:)-double(denoised(:))).^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate MMSIM value
K = [0.01 0.03];
window = fspecial('gaussian', 11, 1.5);
L = 1;
[mssim1, ~] = ssim_index(denoised,clean,K,window,L);
SSIM = mssim1;

end

function A_RGGB = convert_from_raw(A_raw)
[M,N] = size(A_raw);
A_RGGB = zeros(M/2,N/2,4);
G1 = zeros(M/2,N/2);
G2 = zeros(M/2,N/2);
R = zeros(M/2,N/2);
B = zeros(M/2,N/2);
% for every patch
for i = 1:2:M - 2 + 1
    for j = 1:2:N - 2 + 1
        local_patch = A_raw(i:i+1,j:j+1);
        g1 = local_patch(1,1);
        g2 = local_patch(2,2);
        r = local_patch(1,2);
        b = local_patch(2,1);
        
        i_insert_idx = floor(i/2) + 1;
        j_insert_idx = floor(j/2) + 1;
        
        G1(i_insert_idx, j_insert_idx) = g1;
        G2(i_insert_idx, j_insert_idx) = g2;
        R(i_insert_idx, j_insert_idx) = r;
        B(i_insert_idx, j_insert_idx) = b;
    end
end

A_RGGB(:,:,1) = R;
A_RGGB(:,:,2) = G1;
A_RGGB(:,:,3) = G2;
A_RGGB(:,:,4) = B;

A_RGGB = single(A_RGGB);
end

function A_raw_origin = convert_raw_to_origin_raw(A_RGGB)

[M,N,~] = size(A_RGGB);

R = A_RGGB(:,:,1);
G1 = A_RGGB(:,:,2);
G2 = A_RGGB(:,:,3);
B = A_RGGB(:,:,4);

A_raw_origin = zeros(2*M,2*N);

for i = 1:2:2*M - 2 + 1
    for j = 1:2:2*N - 2 + 1
        
        i_insert_idx = floor(i/2) + 1;
        j_insert_idx = floor(j/2) + 1;
        
        g1 = G1(i_insert_idx, j_insert_idx);
        g2 = G2(i_insert_idx, j_insert_idx);
        r = R(i_insert_idx, j_insert_idx);
        b = B(i_insert_idx, j_insert_idx);
        
        A_raw_origin(i,j) = g1;
        A_raw_origin(i,j+1) = r;
        A_raw_origin(i+1,j) = b;
        A_raw_origin(i+1,j+1) = g2;
        
    end
end

% A_raw_origin = uint8(A_raw_origin);
% A_color = run_pipeline(A_raw_origin, metadata, 'raw', 'srgb');

end
