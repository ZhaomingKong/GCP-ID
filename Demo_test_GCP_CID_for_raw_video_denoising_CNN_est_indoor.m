%% Demo denoising CRVD dataset

% Add related path
addpath(genpath(pwd))
clean_video_path = 'E:\Denoising\Data\Video_data\CRVD\Raw_mat_file\Clean_selected\';
noisy_video_path = 'E:\Denoising\Data\Video_data\CRVD\Raw_mat_file\Noisy_selected\';
warning('off')

addpath(clean_video_path);
addpath(noisy_video_path);

noisy_name = dir(fullfile(noisy_video_path, '*.mat'));
clean_name = dir(fullfile(clean_video_path, '*.mat'));
num_videos = length(noisy_name);

% load Patch Representation;
cur_path = pwd;

% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modifiable parameters

ps = 8; SR = 16; maxK = 50; N_step_spatial = 4;N_step_time = 1; global_est_nois = 1;
tau = 1.1;  modified = 1; global_learning = 1; divide_ratio = 1.2;

disp(['ps: ', num2str(ps), ' SR: ', num2str(SR),' N_step_spatial: ',num2str(N_step_spatial), ' N_step_time: ',  num2str(N_step_time),  ' maxK: ',  num2str(maxK)])
disp(['global_learning: ', num2str(global_learning), ' divide_ratio: ', num2str(divide_ratio)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       MODIFY BELOW THIS POINT ONLY IF YOU KNOW WHAT YOU ARE DOING       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(['*******************',noisy_video_path,'**********************']);
disp(['global_est_nois: ',num2str(global_est_nois)])
disp('****************************************************************')

file_name = 'noise_lvl_est_CRVD_indoor.mat';
load(file_name);

disp(['------------------------ file_name ----------------------------']);
disp(['file_name: ', file_name])
disp(['------------------------ file_name ----------------------------']);

psnr_all = 0; ssim_all = 0;
psnr_noisy_all = 0; ssim_noisy_all = 0; elapsed_time = 0;
k = 0;

for i = 1:num_videos
    noisy_i = noisy_name(i).name;
    load(noisy_i);
    
    mean_i = clean_name(i).name;
    load(mean_i);
    
    noisy_video = single(Noisy_video); clean_video = single(Clean_video);
    
    tic;
    
    Noisy_RGGB_viRaw = convert_from_raw(noisy_video);
    Clean_RGGB_viRaw = convert_from_raw(clean_video);
    
    [psnr_noisy_i, ssim_noisy_i] = cal_raw_video_metrics(Noisy_RGGB_viRaw, Clean_RGGB_viRaw);
    psnr_noisy_all = psnr_noisy_all + psnr_noisy_i;
    ssim_noisy_all = ssim_noisy_all + ssim_noisy_i;
    
    noise_est_matrix = squeeze(noise_lvl_est_CRVD_indoor_mtx(i,:,:));
    avg_noise_lvl = mean(noise_est_matrix(:));
    
    [denoised_RGGB_video] = GCP_ID_raw_video_with_neighouring_noise_est_mex(Noisy_RGGB_viRaw, noise_est_matrix, ps, N_step_spatial, N_step_time, SR, maxK, tau, global_est_nois);
    denoised_raw_origin = convert_RGGB_to_raw_origin(denoised_RGGB_video);
    
    time = toc;
    elapsed_time = elapsed_time + time;
    
    [psnr_i, ssim_i] = cal_raw_video_metrics(denoised_RGGB_video, Clean_RGGB_viRaw);
    psnr_all = psnr_all + psnr_i;
    ssim_all = ssim_all + ssim_i;
    
    disp(['i = ',num2str(i), ' time = ',num2str(time), ' avg_noise_lvl_est = ',num2str(avg_noise_lvl), ' psnr_noisy = ',num2str(psnr_noisy_i), ' ssim_noisy = ',num2str(ssim_noisy_i), ' psnr_predicted = ',num2str(psnr_i), ' ssim_predicted = ',num2str(ssim_i)])
    
    
    avg_psnr_predicted = psnr_all/i; avg_ssim_predicted = ssim_all/i;
    avg_psnr_noisy = psnr_noisy_all/i; avg_ssim_noisy = ssim_noisy_all/i;
    avg_elapsed_time = elapsed_time / i;
    
    disp('############################## print statistics #####################################')
    disp(['noise level = ',num2str(avg_noise_lvl), ' avg_elapsed_time = ', num2str(avg_elapsed_time), ' avg_psnr_noisy = ', num2str(avg_psnr_noisy), ' avg_psnr_denoised = ', num2str(avg_psnr_predicted), ' avg_ssim_noisy = ', num2str(avg_ssim_noisy), ' avg_ssim_denoised = ', num2str(avg_ssim_predicted)]);
    disp('############################## print statistics #####################################')
    
end

%% Other functions

function A_RGGB = convert_from_raw(A_raw)
[M,N,T] = size(A_raw);
A_RGGB = zeros(M/2,N/2,4,T);

for t = 1:T
    raw_t = A_raw(:,:,t);
    A_RGGB_t = convert_from_raw_single_frame(raw_t);
    A_RGGB(:,:,:,t) = A_RGGB_t;
end

A_RGGB = single(A_RGGB);

end

function A_RGGB = convert_from_raw_single_frame(A_raw)
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

function A_raw_origin = convert_RGGB_to_raw_origin(A_RGGB)

[M,N,D,T] = size(A_RGGB);
A_raw_origin = zeros(2*M,2*N, T);

for t = 1:T
    A_RGGB_t = A_RGGB(:,:,:,t);
    A_raw_origin(:,:,t) = convert_RGGB_to_raw_origin_single_frame(A_RGGB_t);
end

A_raw_origin = single(A_raw_origin);

end

function A_raw_origin = convert_RGGB_to_raw_origin_single_frame(A_RGGB)

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

end


function [psnr, ssim] = cal_raw_video_metrics(denoised, origin)

sum_psnr = 0; sum_ssim = 0;
num_frames = size(denoised,4);

for i = 1:num_frames
    denoised_i = denoised(:,:,:,i);
    origin_i = origin(:,:,:,i);
    
    [psnr_i, ssim_i] = calculate_one_frame(double(denoised_i), double(origin_i));
    sum_psnr = sum_psnr + psnr_i;
    sum_ssim = sum_ssim + ssim_i;
end

psnr = sum_psnr / num_frames;
ssim = sum_ssim / num_frames;

end

function [PSNR, SSIM] = calculate_one_frame(denoised, clean)
PSNR = 10*log10(1/mean((clean(:)-double(denoised(:))).^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate MMSIM value
K = [0.01 0.03];
window = fspecial('gaussian', 11, 1.5);
L = 1;
[mssim1, ~] = ssim_index(denoised(:,:,1),clean(:,:,1),K,window,L);
[mssim2, ~] = ssim_index(denoised(:,:,2),clean(:,:,2),K,window,L);
[mssim3, ~] = ssim_index(denoised(:,:,3),clean(:,:,3),K,window,L);
[mssim4, ~] = ssim_index(denoised(:,:,4),clean(:,:,4),K,window,L);
SSIM = (mssim1 + mssim2 + mssim3 + mssim4)/4.0;

% SSIM = ssim_index3d(denoised,clean);

end


