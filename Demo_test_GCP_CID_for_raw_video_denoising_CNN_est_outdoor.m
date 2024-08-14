%% Demo denoising CRVD outdoor dataset

% Add related path
addpath('mex');
addpath(genpath(pwd))
clean_video_path = 'Video_data\CRVD\Raw_mat_file\Clean_selected\';
noisy_video_path = 'Video_data\CRVD\Raw_mat_file\Noisy_selected\';
warning('off')

addpath(outdoor_raw_path);
addpath(write_path);

noisy_name = dir(fullfile(outdoor_raw_path, '*.mat'));
num_videos = length(noisy_name);

% load Patch Representation;
cur_path = pwd;

% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modifiable parameters

ps = 8; SR = 16; maxK = 30; N_step_spatial = 4;N_step_time = 1; global_est_nois = 0;
tau = 1.1;  modified = 1; global_learning = 1; divide_ratio = 1.2;

disp(['ps: ', num2str(ps), ' SR: ', num2str(SR),' N_step_spatial: ',num2str(N_step_spatial), ' N_step_time: ',  num2str(N_step_time)])
disp(['global_est_nois: ', num2str(global_est_nois), ' maxK: ', num2str(maxK)])


% Select noise est matrix
file_name = 'noise_lvl_est_CRVD_outdoor.mat';
load(file_name)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       MODIFY BELOW THIS POINT ONLY IF YOU KNOW WHAT YOU ARE DOING       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


disp(['*******************',noisy_video_path,'**********************']);
disp(['global_est_nois: ',num2str(global_est_nois)])
disp(['file_name: ', file_name])
disp('****************************************************************')

elapsed_time = 0;
k = 0;

for i = 14:num_videos

    noisy_i = noisy_name(i).name;
    load(noisy_i);

    noisy_video = single(Noisy)/4096;

    tic;

    % Noise est using CNN
    Noisy_RGGB_viRaw = convert_from_raw(noisy_video);
    noise_est_matrix = squeeze(noise_lvl_est_CRVD_outdoor_mtx(i,:,:));
    avg_noise_lvl = mean(noise_est_matrix(:));

    [denoised_video] = CMStSVD_raw_video_with_neighouring_noise_est_mex(Noisy_RGGB_viRaw, noise_est_matrix, ps, N_step_spatial, N_step_time, SR, maxK, tau, global_est_nois);

    time = toc;
    elapsed_time = elapsed_time + time;

    save_name = strcat(num2str(i),'_',noisy_i,'_global_est_','denoised.mat');

    disp(['i: ',num2str(i),' ',noisy_i, ' global_est_noise_lvl:', num2str(avg_noise_lvl), ' time: ',num2str(time)]);

%     cd(write_path)
%     denoised_video = uint8(denoised_video*255);
%     save(save_name,'denoised_video');
%     cd(cur_path);

%     clear Noisy; clear denoised_video; clear denoised_raw_origin;

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

% Cal_metrics (PSNR and SSIM)
function [psnr, ssim] = cal_video_metrics(denoised, origin)

sum_psnr = 0; sum_ssim = 0;
num_frames = size(denoised,4);

for i = 1:num_frames
    denoised_i = denoised(:,:,i);
    origin_i = origin(:,:,i);
    
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
% [mssim1, ~] = ssim_index(denoised(:,:,1),clean(:,:,1),K,window,L);
% [mssim2, ~] = ssim_index(denoised(:,:,2),clean(:,:,2),K,window,L);
% [mssim3, ~] = ssim_index(denoised(:,:,3),clean(:,:,3),K,window,L);
% SSIM = (mssim1+mssim2+mssim3)/3.0;

SSIM = ssim_index(denoised,clean,K,window,L);

end
