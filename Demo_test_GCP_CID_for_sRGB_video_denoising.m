addpath(genpath(pwd))
%% Demo Denoising with IOCV

% load Patch Representation;
% load U;load V;
% cur_path = pwd;
% % addpath(write_path);
% video_path = 'E:\Denoising\Data\Video_data\IOCV\Mat_all';
% addpath(video_path);
% 
% noisy_name = dir(fullfile(video_path, '*noisy.avi'));
% clean_name = dir(fullfile(video_path, '*mean_filter_mean.avi'));
% num_videos = length(noisy_name);
% 
% % Parameters
% disp('*******************************************************************************************')
% ps = 8; SR = 16; maxK = 30; N_step_spatial = 6;N_step_time = 1;
% tau = 1.1;  modified = 1; global_est = 1; divide_ratio = 1.2;
% disp(['ps: ', num2str(ps), ' maxK: ', num2str(maxK), ' SR: ', num2str(SR),' N_step_spatial: ',num2str(N_step_spatial), ' N_step_time: ',  num2str(N_step_time)])
% disp(['global_est: ', num2str(global_est), ' divide_ratio: ', num2str(divide_ratio)])
% disp('*******************************************************************************************')
% 
% file_name = 'noise_est_mtx_IOCV_sRGB.mat';
% load(file_name);
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% 
% psnr_all = 0; ssim_all = 0;
% psnr_noisy_all = 0; ssim_noisy_all = 0; elapsed_time = 0;
% 
% for i = 1:num_videos
%     noisy_i = noisy_name(i).name;
%     noisy_video = load_video(noisy_i);
%     
%     mean_i = clean_name(i).name;
%     mean_video = load_video(mean_i);
%     disp(noisy_i)
%     
%     noisy_video = single(noisy_video); mean_video = single(mean_video);
%     
%     [psnr_noisy_i, ssim_noisy_i] = cal_video_metrics(mean_video/255, noisy_video/255);
%     psnr_noisy_all = psnr_noisy_all + psnr_noisy_i;
%     ssim_noisy_all = ssim_noisy_all + ssim_noisy_i;
%     
%     noise_lvl_mtx_i = squeeze(noise_est_mtx_IOCV_sRGB(i,:,:));
%     sigma = mean(noise_lvl_mtx_i(:));
%     
%     tic;
%     
%     denoised_video = GCP_CID_sRGB_color_video_with_mex(single(noisy_video), noise_lvl_mtx_i, global_est, ps, maxK, SR, N_step_spatial, N_step_time, tau, divide_ratio);
%     
% %     [H,W,~,~] = size(noisy_video);
% %     if(sigma<32)
% %         noisy_video_reshape = noisy_video;
% %         reshape = false;
% %     end
% %     if(sigma>=32 && sigma < 40)
% %         noisy_video_reshape = imresize(noisy_video,0.75);
% %         reshape = true;
% %     end
% %     if(sigma>=40)
% %         noisy_video_reshape = imresize(noisy_video,0.6);
% %         reshape = true;
% %     end
% %     
% %     denoised_video = GCP_CID_sRGB_color_video_with_mex_enlarge(single(noisy_video_reshape), noise_lvl_mtx_i, global_est, ps, maxK, SR, N_step_spatial, N_step_time, tau, divide_ratio);
% %     
% %     if(reshape == true)
% %         denoised_video = imresize(denoised_video,[H,W]);
% %     end
%     
%     time = toc;
%     elapsed_time = elapsed_time + time;
%     
%     [psnr_i, ssim_i] = cal_video_metrics(denoised_video/255, mean_video/255);
%     psnr_all = psnr_all + psnr_i;
%     ssim_all = ssim_all + ssim_i;
%     
%     disp(['i = ',num2str(i), ' time = ',num2str(time), ' psnr_noisy = ',num2str(psnr_noisy_i), ' ssim_noisy = ',num2str(ssim_noisy_i), ' psnr_predicted = ',num2str(psnr_i), ' ssim_predicted = ',num2str(ssim_i)])
%     
%     avg_psnr_predicted = psnr_all/i; avg_ssim_predicted = ssim_all/i;
%     avg_psnr_noisy = psnr_noisy_all/i; avg_ssim_noisy = ssim_noisy_all/i;
%     avg_elapsed_time = elapsed_time / i;
%     
%     % disp(['noise level = ',num2str(sigma),' avg_psnr_noisy = ',num2str(avg_psnr_noisy),' avg_ssim_noisy = ',num2str(avg_ssim_noisy), ' avg_psnr_predicted = ',num2str(avg_psnr_predicted), ' avg_ssim_predicted = ',num2str(avg_ssim_predicted)])
%     disp('############################## print statistics #####################################')
%     disp(['noise level = ',num2str(sigma), ' avg_elapsed_time = ', num2str(avg_elapsed_time), ' avg_psnr_noisy = ', num2str(avg_psnr_noisy), ' avg_psnr_denoised = ', num2str(avg_psnr_predicted), ' avg_ssim_noisy = ', num2str(avg_ssim_noisy), ' avg_ssim_denoised = ', num2str(avg_ssim_predicted)]);
%     disp('############################## print statistics #####################################')
%     
% end


%% Demo denoising CRVD dataset

% Addpath
clean_video_path = 'Video_data\CRVD\sRGB_mat_file\Clean_selected\';
noisy_video_path = 'Video_data\CRVD\sRGB_mat_file\Noisy_selected\';

addpath(clean_video_path);
addpath(noisy_video_path);

noisy_name = dir(fullfile(noisy_video_path, '*.mat'));
clean_name = dir(fullfile(clean_video_path, '*.mat'));
num_videos = length(noisy_name);

% load Patch Representation;
load U_3D;load V_3D;
load U_4D;load V_4D;
cur_path = pwd;

% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Modifiable parameters

ps = 8; SR = 16; maxK = 30; N_step_spatial = 4;N_step_time = 1;
tau = 1.1;  modified = 1; global_est = 1; divide_ratio = 1.2;

disp(['ps: ', num2str(ps), ' SR: ', num2str(SR),' N_step_spatial: ',num2str(N_step_spatial), ' N_step_time: ',  num2str(N_step_time)])
disp(['global_est: ', num2str(global_est), ' divide_ratio: ', num2str(divide_ratio)])

% Select implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       MODIFY BELOW THIS POINT ONLY IF YOU KNOW WHAT YOU ARE DOING       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_name = 'noise_est_mtx_CRVD_sRGB_indoor.mat';
load(file_name);
disp('------------------------------------------------------')
disp(file_name)
disp('------------------------------------------------------')


psnr_all = 0; ssim_all = 0;
psnr_noisy_all = 0; ssim_noisy_all = 0; elapsed_time = 0;
k = 0;

for i = 1:num_videos
    noisy_i = noisy_name(i).name;
    load(noisy_i);
    disp(noisy_i)
    
    mean_i = clean_name(i).name;
    load(mean_i);
    
    noisy_video = single(noisy_video); clean_video = single(clean_video);
    
    noise_lvl_mtx_i = squeeze(noise_est_mtx_CRVD_sRGB_indoor(i,:,:));
    sigma = mean(noise_lvl_mtx_i(:));
    
    [psnr_noisy_i, ssim_noisy_i] = cal_video_metrics(noisy_video/255, clean_video/255);
    psnr_noisy_all = psnr_noisy_all + psnr_noisy_i;
    ssim_noisy_all = ssim_noisy_all + ssim_noisy_i;
    
    tic;
    
    % % Use the resize strategy to handle severe noise.
    [H,W,~,~] = size(noisy_video);
    if(sigma<29.6)
        noisy_video_reshape = noisy_video;
        reshape = false;
    end
    if(sigma>=29.6 && sigma < 40)
        noisy_video_reshape = imresize(noisy_video,0.75);
        reshape = true;
    end
    if(sigma>=40)
        noisy_video_reshape = imresize(noisy_video,0.6);
        reshape = true;
    end
    
    denoised_video = GCP_CID_sRGB_color_video_with_mex_enlarge(single(noisy_video_reshape), noise_lvl_mtx_i, global_est, ps, maxK, SR, N_step_spatial, N_step_time, tau, divide_ratio);
    
    if(reshape == true)
        denoised_video = imresize(denoised_video,[H,W]);
    end
    
    k = k + 1;
    
    time = toc;
    elapsed_time = elapsed_time + time;
    
    [psnr_i, ssim_i] = cal_video_metrics(denoised_video/255, clean_video/255);
    psnr_all = psnr_all + psnr_i;
    ssim_all = ssim_all + ssim_i;
    
    disp(['i = ',num2str(i), ' time = ',num2str(time), ' noise_est: ',num2str(sigma), ' psnr_noisy = ',num2str(psnr_noisy_i), ' ssim_noisy = ',num2str(ssim_noisy_i), ' psnr_predicted = ',num2str(psnr_i), ' ssim_predicted = ',num2str(ssim_i)])
    
    % % save denoised results as mat files
    %         save_path = 'CRVD_sRGB_denoised';
    %         cur_path = pwd;
    %         cd(save_path);
    %         noisy_name_split = split(noisy_i,'.');
    %         save_name = strcat(noisy_name_split{1},'_GCP-ID.mat');
    %         save(save_name, 'denoised_video');
    %         cd(cur_path);
    %
    %         clear noisy_video
    %         clear clean_video
    %         clear denoised_video
    
end


avg_psnr_predicted = psnr_all/i; avg_ssim_predicted = ssim_all/i;
avg_psnr_noisy = psnr_noisy_all/i; avg_ssim_noisy = ssim_noisy_all/i;
avg_elapsed_time = elapsed_time / i;

disp('############################## print statistics #####################################')
disp(['noise level = ',num2str(sigma), ' avg_elapsed_time = ', num2str(avg_elapsed_time), ' avg_psnr_noisy = ', num2str(avg_psnr_noisy), ' avg_psnr_denoised = ', num2str(avg_psnr_predicted), ' avg_ssim_noisy = ', num2str(avg_ssim_noisy), ' avg_ssim_denoised = ', num2str(avg_ssim_predicted)]);
disp('############################## print statistics #####################################')


%% Demo Denoising Realistic Without Reference

% Parameters
% load video_clip;
% % load Patch Representation;
% load U_3D;load V_3D;
% load U_4D;load V_4D;
% load U; load V;
% U_ifft = ifft(U,[],3);V_ifft = ifft(V,[],3);
%
% U_3D_fft = fft(U_3D,[],3);V_3D_fft = fft(V_3D,[],3);
% U_4D_fft = fft(U_4D,[],3);V_4D_fft = fft(V_4D,[],3);
%
% cur_path = pwd;
%
% video_clip = video_clip(1:128,1:128,:,1:18);
%
% [H,W,~,D] = size(video_clip);
% info1 = int32([H,W,D]);
% SR = 16;
% maxK = 30;
% N_step_spatial = 6;
% N_step_time = 1;
% ps = 8;
% divide_ratio = 1.2;
% info2 = [ps,N_step_spatial, N_step_time, SR,maxK];
% info2 = int32(info2);
% tau = 1.1;
% modified = 1;
% sigma = 36;
% info3 = [tau,modified,sigma];
% display_video = 0;
% global_learning = 1;
%
% % Select implementation
% method = 'GCP-CID';
%
% % Start denoising
% tic
%
% denoised_video = GCP_CID_sRGB_color_video_with_mex(single(video_clip), sigma, ps, maxK, SR, N_step_spatial, N_step_time, tau, divide_ratio);
%
% toc
%
%
% % Display two videos
% for loop = 1:3
%     for i = 1:size(denoised_video,4)
%         subplot(121)
%         imshow(single(video_clip(:,:,:,i))/255*1.18);title('Noisy Video');
%         subplot(122)
%         imshow(single(denoised_video(:,:,:,i))/255*1.18);title('denoised');
%         pause(0.01);
%     end
% end

% % cal metrics
% [psnr_denoised, ssim_denoised] = cal_video_metrics(denoised_video/255, video_clip/255);




%% Other useful functions

% Obtain the optimal value for an input
function best_sigma = select_best_sigma(noise_model, noisy_name, clean_name)

load U; load V;
ps = 8; SR = 16; maxK = 30; N_step_spatial = 6;N_step_time = 1;
num_videos = length(noisy_name);

for choice = 1:length(noise_model)
    psnr_count = 0; ssim_count = 0;
    modelSigma  = noise_model(choice);
    disp(['************ model noise is: ',num2str(modelSigma),' **************'])
    
    for i = 1:num_videos
        
        noisy_i = noisy_name(i).name;
        noisy_video = load_video(noisy_i);
        
        mean_i = clean_name(i).name;
        mean_video = load_video(mean_i);
        
        [denoised_video] = efficient_tSVD_denoising(noisy_video,U, V, ps, N_step_spatial, SR, maxK, modelSigma);
        [PSNR_denoised, SSIM_denoised] = cal_video_metrics(denoised_video, mean_video);
        psnr_count = psnr_count + PSNR_denoised;
        ssim_count = ssim_count + SSIM_denoised;
        
    end
    
    disp(['modelSigma =',num2str(modelSigma),' psnr_count = ',num2str(psnr_count/num_videos),' ssim_count = ',num2str(ssim_count/num_videos)]);
    sigma_all(choice) = modelSigma;
    psnr_all(choice) = psnr_count/num_videos;
    
end

[~,max_id] = max(psnr_all);
best_sigma = sigma_all(max_id);

end

% Cal_metrics (PSNR and SSIM)
function [psnr, ssim] = cal_video_metrics(denoised, origin)

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
SSIM = (mssim1+mssim2+mssim3)/3.0;

end

% Display two videos
function [] = display_videos(noisy_clip, denoised_video)

for loop = 1:2
    for i = 1:size(noisy_clip,4)
        subplot(121)
        imshow(single(noisy_clip(:,:,:,i))/255*1.8);title('Noisy Video');
        subplot(122)
        imshow(denoised_video(:,:,:,i)/255*1.8);title('denoised');
        pause(0.06);
    end
end

end

% Write video
function write_video(video, write_name)

outputVideo = VideoWriter(write_name);
% 启动写入器
open( outputVideo )

for i = 1 : size(video,4)
    img = uint8(video(:,:,:,i));
    % imshow( img )
    writeVideo( outputVideo, img )  % 写入
end
% 关闭写入器，使视频生效
close(outputVideo)

end

% Load video
function noisy_video = load_video(video_name)

video_noisy = VideoReader(video_name);
k = 1;
while hasFrame(video_noisy)
    vidFrame_noisy = readFrame(video_noisy);
    noisy_video(:,:,:,k) = vidFrame_noisy;
    k = k+1;
end

end

