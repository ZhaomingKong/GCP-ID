function [denoised_img] = GCP_CID_sRGB_est_CNN(im1, im, Noise_est_mtx, ps, SR, maxK, N_step, modified, tau, global_learning)

%% settings

[H,W,D] = size(im1);
divide_size = 128;
H_steps = ceil(H/divide_size);
W_steps = ceil(W/divide_size);
use_mex = 1;
denoised_img = zeros(H,W,D);

noisy_img = im1;
clean_img = im1 + 0.006;


% A more adaptive scheme
[denoised_img,~,~] = GCP_CID_sRGB_adaptive(single(noisy_img), single(clean_img), ps, SR, Noise_est_mtx, maxK, N_step, modified, tau, global_learning, 1.2);

end

function best_sigma = select_best_sigma(Noisy_raw, Clean_raw, ps, SR, maxK, N_step, modified, tau, global_learning)

Noisy_RGGB_imRaw = convert_from_raw(Noisy_raw);
sigma_all = [1.25, 2.5, 5, 10, 15, 20, 25, 30, 40, 50];
num_sigma = length(sigma_all);

% parameter settings
use_mex = 1;

for i = 1:num_sigma
    
    sigma_i = sigma_all(i);
    if(use_mex == 1)
        [im_denoised_RGGB] = CMStSVD_raw_with_mex(single(Noisy_RGGB_imRaw),sigma_i/255,ps,maxK,SR,N_step,tau);
    else
        [im_denoised_RGGB,~,~] = CMStSVD_raw(single(Noisy_RGGB_imRaw), single(Noisy_RGGB_imRaw), ps, SR, sigma_i/255, maxK, N_step, modified, tau, global_learning);
    end
    
    im_denoised_raw_origin = convert_raw_to_origin_raw(im_denoised_RGGB);
    [PSNR_denoised, SSIM_denoised] = calculate_index(im_denoised_raw_origin, Clean_raw);
    
    PSNR_denoised_all(i) = PSNR_denoised;
    SSIM_denoised_all(i) = SSIM_denoised;
    
end

[~,max_id] = max(PSNR_denoised_all);
best_sigma = sigma_all(max_id);

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
