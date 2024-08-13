addpath(genpath(pwd))
addpath(genpath('D:\Code\denoise\New_ideas\GCP-CID_raw'))
addpath(genpath('D:\Code\denoise\MSI_denoising\MSt-SVD'))

%% SIDD Validation set
load ValidationGtBlocksRaw
load ValidationNoisyBlocksRaw

GT = ValidationGtBlocksRaw;
Noisy = ValidationNoisyBlocksRaw;

% parameter settings
ps = 8; SR = 20; N_step = 8; maxK = 90;
modified = 1; tau = 1.1; global_learning = 1;

[N,B,H,W] = size(GT);
step_size = 128;
H_steps = ceil(H/128);
W_steps = ceil(W/128);

for epoch_chosen = 89
    file_name = strcat('noise_lvl_est_SIDDvalidation_mtx_single_channel_CNN_epoch_',num2str(epoch_chosen),'_SIDD_without_DND.mat');
    load(file_name);
    disp('------------------------------------------------------')
    disp(file_name)
    disp('------------------------------------------------------')
    
    PSNR_all = 0; SSIM_all = 0;
    for i = 1:N
        psnr_i = 0;ssim_i = 0;
        for b = 1:B
            clean_ib = squeeze(GT(i,b,:,:));
            noisy_ib = squeeze(Noisy(i,b,:,:));
            noise_lvl_mtx_ib = squeeze(noise_lvl_est_valid_mtx(i,b,:,:));
            
            tic
            sigma = mean(noise_lvl_mtx_ib(:));
            
            [im_denoised_raw_origin] = CMStSVD_raw_est_CNN(noisy_ib, clean_ib, noise_lvl_mtx_ib, ps, SR, maxK, N_step, modified, tau, global_learning);
            %             [im_denoised_RGGB] = CMStSVD_raw_with_mex(single(Noisy_RGGB_imRaw),sigma/255,ps,maxK,SR,N_step,tau);
            Time_ib = toc;
            
            
            [PSNR_noisy, SSIM_noisy] = calculate_index(noisy_ib, clean_ib);
            [PSNR_ib, SSIM_ib] = calculate_index(im_denoised_raw_origin, clean_ib);
            
            disp(['img_idx = ',num2str(i), '-',num2str(b),'.',' Time = ',num2str(Time_ib), ' Avg Noise level: ',num2str(sigma), ' Sigma_input: ',num2str(sigma),' Noisy: PSNR = ', num2str(PSNR_noisy), ' SSIM = ', num2str(SSIM_noisy), '; Denoised: PSNR = ', num2str(PSNR_ib), ' SSIM = ', num2str(SSIM_ib)])
%             disp(['img_idx = ',num2str(i), '--',num2str(b),'. Noisy: PSNR = ', num2str(PSNR_noisy), ' SSIM = ', num2str(SSIM_noisy)])
            
            PSNR_all = PSNR_all + PSNR_ib;
            SSIM_all = SSIM_all + SSIM_ib;
            
            psnr_i = psnr_i + PSNR_ib;
            ssim_i = ssim_i + SSIM_ib;
            
        end
        
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        psnr_avg_i = psnr_i/B; ssim_avg_i = ssim_i/B;
        disp(['i = ',num2str(i),' PSNR_avg = ',num2str(psnr_avg_i), ' SSIM_avg = ',num2str(ssim_avg_i)])
        disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        
    end
    
    disp('*****************************************************************')
    disp(['sigma =: ',num2str(sigma)])
    disp(['Average PSNR = ',num2str(PSNR_all/(N*B)), ' Average SSIM = ',num2str(SSIM_all/(N*B))]);
    disp('*****************************************************************')
end


%% SIDD Benchmark set
% clear all;
% cur_path = pwd;
% load BenchmarkNoisyBlocksRaw
% load noise_lvl_est_SIDDbenchmark_mtx_single_channel_CNN_epoch_89_SIDD_with_DND.mat
% submitDir ='D:\Code\denoise\New_ideas\DNN_and_tSVD\Denoised\Raw\SIDD';
% 
% GT = BenchmarkNoisyBlocksRaw + 0.01;
% Noisy = BenchmarkNoisyBlocksRaw ;
% 
% DenoisedBlocksRaw = cell(40, 32);
% 
% % parameter settings
% ps = 8; SR = 20; sigma = 20; N_step = 4; maxK = 120;
% modified = 1; tau = 1.1; global_learning = 1;
% 
% [M,N,H,W] = size(GT);
% 
% disp('*******************************************')
% disp(['sigma =: ',num2str(sigma),' ps = ',num2str(ps), ' N_step = ',num2str(N_step), ' maxK = ',num2str(maxK)])
% disp('*******************************************')
% 
% for i = 1:M
%     disp(['%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% i = ',num2str(i),' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'])
%     for b = 1:N
%         clean_ib = squeeze(GT(i,b,:,:));
%         noisy_ib = squeeze(Noisy(i,b,:,:));
%         noise_lvl_mtx_ib = squeeze(noise_lvl_est_SIDDbenchmark_mtx(i,b,:,:));
%         
%         tic
%         sigma = mean(noise_lvl_mtx_ib(:));
%         
%         [im_denoised_raw_origin] = CMStSVD_raw_est_CNN(noisy_ib, clean_ib, noise_lvl_mtx_ib, ps, SR, maxK, N_step, modified, tau, global_learning);
% %         [im_denoised_RGGB] = CMStSVD_raw_with_mex(single(Noisy_RGGB_imRaw),sigma/255,ps,maxK,SR,N_step,tau);
%         Time_ib = toc;
%         
%         DenoisedBlocksRaw{i, b} = single(im_denoised_raw_origin);
%         
%         
%         cd(submitDir);
%         denoised_RGGB_imRaw = convert_from_raw(im_denoised_raw_origin);
%         write_name = strcat('Denoised_img_',num2str(i),'_',num2str(b),'.png');
%         imwrite(denoised_RGGB_imRaw(:,:,2)*1.38, write_name);
%         
%         Noisy_RGGB_imRaw = convert_from_raw(noisy_ib);
%         write_name = strcat('Noisy_img_',num2str(i),'_',num2str(b),'.png');
%         imwrite(Noisy_RGGB_imRaw(:,:,2)*1.38, write_name);
%         cd(cur_path);
%         
%         disp(['img_idx = ',num2str(i), '-',num2str(b),'.',' Time = ',num2str(Time_ib), ' Noise_mtx: ',num2str(noise_lvl_mtx_ib(:)'), ' Avg_Noise_level: ',num2str(sigma)])
%        
%     end 
% end
% 
% TimeMPRaw = 0.01;
% OptionalData.MethodName = 'DummyDenoiser';
% OptionalData.Authors = 'Jane Doe and Jone Doa';
% OptionalData.PaperTitle = 'Dummy Image Denoising';
% OptionalData.Venue = 'SIDD Demo';
% 
% fprintf('Saving resutls...\n');
% save(fullfile(submitDir, 'SubmitRaw.mat'), 'DenoisedBlocksRaw', ...
%     'TimeMPRaw', 'OptionalData', '-v7.3');
% fprintf('Done!\n');

%% DND set
% clear all;
% noise_est_name = 'noise_lvl_est_mtx_DND_single_channel_epoch89_SIDD_with_DND.mat';
% load(noise_est_name);
% disp('###########################################')
% disp(noise_est_name)
% disp('###########################################')
% % parameter settings
% cur_folder = pwd;
% data_folder = 'DnD';
% output_folder = 'D:\Code\denoise\New_ideas\DNN_and_tSVD\Denoised\Raw\DND';
% infos = load(fullfile(data_folder, 'info.mat'));
% info = infos.info;
% 
% % Parameter setting
% disp('--------------------------------------------------------------')
% disp(['denoising_method: ',num2str('CMStSVD_raw')])
% ps = 8;maxK = 120;global_learning = 1; noise_est_func = 0; modelSigma = 10;
% SR = 20; N_step = 4;modified = 1; tau = 1.1; divide_factor = 1.2;
% disp([' ps: ', num2str(ps), ' maxK: ',num2str(maxK),  ' N_step: ',num2str(N_step), ' modelSigma: ',num2str(modelSigma)])
% disp([' SR: ', num2str(SR), ' tau: ',num2str(tau),  ' divide_factor: ',num2str(divide_factor), ' global_learning: ',num2str(global_learning)])
% disp('--------------------------------------------------------------')
% 
% if(noise_est_func == 1)
% end
% 
% N_imgs = 50; N_blocks = 20;
% % iterate over images
% for i=6:50
%     img = load(fullfile(data_folder, 'images_raw', sprintf('%04d.mat', i)));
%     Inoisy = img.Inoisy;
%     
%     % iterate over bounding boxes
%     Idenoised_crop_bbs = cell(1,N_blocks);
%     
%     for b=1:20
%         
%         bb = info(i).boundingboxes(b,:);
%         Inoisy_crop = Inoisy(bb(1):bb(3), bb(2):bb(4), :);
%         nlf = info(i).nlf;
%         
%         nlf.sigma_all = info(i).sigma_raw(b,:,:);
%         noisy_test = Inoisy_crop;
%         origin_test = noisy_test + 0.08;
%         noise_lvl_mtx_ib = squeeze(noise_est_mtx_DND(i,b,:,:));
%         
%         nlf.sigma_avg = mean(nlf.sigma_all(:));
%         modelSigma = mean(noise_lvl_mtx_ib);
%         
%         tic
%         [im_denoised_raw_origin] = CMStSVD_raw_est_CNN(noisy_test, origin_test, noise_lvl_mtx_ib, ps, SR, maxK, N_step, modified, tau, global_learning);
%         time = toc;
% 
%         disp(['image_i = ',num2str(i), ' image_crop_b = ', num2str(b), ' Noise_mtx: ',num2str(noise_lvl_mtx_ib(:)'), ' avg_noise_lvl = ',num2str(mean(noise_lvl_mtx_ib(:))), ' Time = ',num2str(time)])
%         
%         Idenoised_crop_bbs{b} = single(im_denoised_raw_origin);
%         cd(output_folder)
%         denoised_RGGB_imRaw = convert_from_raw(im_denoised_raw_origin);
%         Inoisy_crop_RGGB = convert_from_raw(Inoisy_crop);
%         write_name_noisy = strcat('Noisy_',num2str(i),'_',num2str(b),'_SigmaEst_',num2str(nlf.sigma_avg),'.png');
%         imwrite(Inoisy_crop_RGGB(:,:,2) * 2.86, write_name_noisy)
%         
%         write_denoised_name = strcat('Denoised_',num2str(i),'_',num2str(b),'Avg_SigmaChosen_',num2str(mean(noise_lvl_mtx_ib(:))),'.png');
%         imwrite(denoised_RGGB_imRaw(:,:,2)*2.86, write_denoised_name)
%         
%         cd(cur_folder)
%         
%     end
%     
%     for b=1:20
%         Idenoised_crop = Idenoised_crop_bbs{b};
%         save(fullfile(output_folder, sprintf('%04d_%02d.mat', i, b)), 'Idenoised_crop');
%     end
%     
%     fprintf('Image %d/%d done\n', i,50);
% end

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

function A_RGGB = convert_from_raw_new(A_raw)
[M,N] = size(A_raw);
A_RGGB = zeros(M,N,4);
G1 = zeros(M,N);
G2 = zeros(M,N);
R = zeros(M,N);
B = zeros(M,N);
% for every patch
for i = 1:2:M - 2 + 1
    for j = 1:2:N - 2 + 1
        local_patch = A_raw(i:i+1,j:j+1);
        g1 = local_patch(1,1);
        g2 = local_patch(2,2);
        r = local_patch(1,2);
        b = local_patch(2,1);
        
        G1(i, j) = g1;
        G2(i+1, j+1) = g2;
        R(i, j+1) = r;
        B(i+1, j) = b;
    end
end

A_RGGB(:,:,1) = R;
A_RGGB(:,:,2) = G1;
A_RGGB(:,:,3) = G2;
A_RGGB(:,:,4) = B;

A_RGGB = single(A_RGGB);

end

function A_raw_origin = convert_raw_to_origin_raw_new(A_RGGB)

[M,N,~] = size(A_RGGB);

R = A_RGGB(:,:,1);
G1 = A_RGGB(:,:,2);
G2 = A_RGGB(:,:,3);
B = A_RGGB(:,:,4);

A_raw_origin = zeros(M,N);

for i = 1:2:M - 2 + 1
    for j = 1:2:N - 2 + 1
        
        
        g1 = G1(i, j);
        g2 = G2(i+1, j+1);
        r = R(i, j+1);
        b = B(i+1, j);
        
        A_raw_origin(i,j) = g1;
        A_raw_origin(i,j+1) = r;
        A_raw_origin(i+1,j) = b;
        A_raw_origin(i+1,j+1) = g2;
        
    end
end

end

%% Sample test script
% render DNG image file into sRGB image
% warning('off')
% 
% % dngFilename = 'colorchart.dng';
% % [imRaw, metadata] = Load_Data_and_Metadata_from_DNG(...
% %     fullfile('data', dngFilename));
% 
% load('0044.mat');
% 
% imRaw = single(Inoisy*255);
% A_RGGB_imRaw = convert_from_raw(imRaw);
% 
% ps = 8; SR = 20; sigma = 25; N_step = 8; maxK = 30;
% modified = 1; tau = 1.1; global_learning = 1;
% 
% % tic
% % [im_denoised_RGGB,~,~] = CMStSVD_raw(single(A_RGGB_imRaw), single(A_RGGB_imRaw), ps, SR, sigma, maxK, N_step, modified, tau, global_learning);
% % time = toc
% 
% im_denoised_raw_origin = convert_raw_to_origin_raw(im_denoised_RGGB);
