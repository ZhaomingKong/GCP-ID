addpath(genpath(pwd))
addpath(genpath('mex'))
addpath(genpath('lib'))
addpath('t_svd_lib');
%% test SIDD sRGB validation dataset
% (SIDD Validation set)
% load ValidationGtBlocksSrgb
% load ValidationNoisyBlocksSrgb
% file_name = 'noise_lvl_est_SIDD_sRGB_validation.mat'; % noise_lvl_est_SIDD_sRGB_validation_modified.mat
% load(file_name);
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% 
% GT = ValidationGtBlocksSrgb;
% Noisy = ValidationNoisyBlocksSrgb;
% 
% % parameter settings
% ps = 8; % patch size
% SR = 20; % search window size
% N_step = 4; % Nstep --> could be 4/6/8. Choose a smaller size for slightly better results. Choose a larger one for faster speed. 
% maxK = 30; % number of patches
% divide_factor = 1.2; % controls the seach scheme
% modified = 1; tau = 1.1; global_learning = 1;
% 
% [N,B,H,W, C] = size(GT);
% step_size = 128;
% H_steps = ceil(H/128);
% W_steps = ceil(W/128);
% 
% 
% PSNR_all = 0; SSIM_all = 0;
% for i = 1:N
%     psnr_i = 0;ssim_i = 0;
%     for b = 1:B
%         clean_ib = squeeze(GT(i,b,:,:,:));
%         noisy_ib = squeeze(Noisy(i,b,:,:,:));
%         noise_lvl_mtx_ib = squeeze(noise_lvl_est_valid_mtx(i,b,:,:));
% 
%         noisy_ib = single(noisy_ib);
%         clean_ib = single(clean_ib);
% 
%         tic
%         sigma = mean(noise_lvl_mtx_ib(:));
% 
% %         [Denoised] = GCP_CID_with_mex(noisy_ib, sigma, ps, maxK, SR, N_step, tau, divide_factor);
%         [Denoised] = GCP_CID_with_mex_modified(noisy_ib, noise_lvl_mtx_ib, sigma, ps, maxK, SR, N_step, tau, divide_factor);
%         Time_ib = toc;
% 
%         [PSNR_noisy, SSIM_noisy] = calculate_index(noisy_ib/255, clean_ib/255);
%         [PSNR_ib, SSIM_ib] = calculate_index(Denoised/255, clean_ib/255);
% 
%         disp(['img_idx = ',num2str(i), '-',num2str(b),'.',' Time = ',num2str(Time_ib), ' Avg Noise level: ',num2str(sigma), ' Sigma_input: ',num2str(sigma),' Noisy: PSNR = ', num2str(PSNR_noisy), ' SSIM = ', num2str(SSIM_noisy), '; Denoised: PSNR = ', num2str(PSNR_ib), ' SSIM = ', num2str(SSIM_ib)])
% 
%         PSNR_all = PSNR_all + PSNR_ib;
%         SSIM_all = SSIM_all + SSIM_ib;
% 
%         psnr_i = psnr_i + PSNR_ib;
%         ssim_i = ssim_i + SSIM_ib;
% 
%     end
%     disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
%     psnr_avg_i = psnr_i/B; ssim_avg_i = ssim_i/B;
%     disp(['i = ',num2str(i),' PSNR_avg = ',num2str(psnr_avg_i), ' SSIM_avg = ',num2str(ssim_avg_i)])
%     disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
% end
% 
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% disp('*****************************************************************')
% disp(['sigma =: ',num2str(sigma)])
% disp(['Average PSNR = ',num2str(PSNR_all/(N*B)), ' Average SSIM = ',num2str(SSIM_all/(N*B))]);
% disp('*****************************************************************')

%% test SIDD sRGB Benchmark dataset
% clear all;
% cur_path = pwd;
% load BenchmarkNoisyBlocksSrgb.mat;
% file_name = 'noise_lvl_est_SIDD_sRGB_benchmark.mat'; % noise_lvl_est_SIDD_sRGB_benchmark_modified.mat
% load(file_name);
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% submitDir = 'Submit_SIDD';
% cur_folder = pwd;
% 
% GT = BenchmarkNoisyBlocksSrgb + 0.01;
% Noisy = BenchmarkNoisyBlocksSrgb;
% 
% DenoisedBlocksSrgb = cell(40, 32);
% 
% % % parameter settings
% ps = 8; % patch size
% SR = 20; % search window size
% N_step = 4; % Nstep --> could be 4/6/8. Choose a smaller size for slightly better results. Choose a larger one for faster speed. 
% maxK = 30; % number of patches
% divide_factor = 1.2; % controls the seach scheme
% modified = 1; tau = 1.1; global_learning = 1;
% 
% [M,N,H,W,C] = size(GT);
% 
% disp('*******************************************')
% disp(['ps = ',num2str(ps), ' N_step = ',num2str(N_step), ' maxK = ',num2str(maxK)])
% disp('*******************************************')
% 
% for i = 1:M
%     disp(['%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% i = ',num2str(i),' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'])
%     for b = 1:N
%         noisy_ib = squeeze(Noisy(i,b,:,:,:));
%         noisy_ib = single(noisy_ib);
%         noise_lvl_mtx_ib = squeeze(noise_lvl_est_test_mtx(i,b,:,:));
%         modelSigma = mean(noise_lvl_mtx_ib(:));
% 
% 
%         tic
%         [Denoised] = GCP_CID_with_mex_modified(noisy_ib, noise_lvl_mtx_ib, modelSigma, ps, maxK, SR, N_step, tau, divide_factor);
%         Time_ib = toc;
% 
%         DenoisedBlocksSrgb{i, b} = single(Denoised);
% 
% 
%         cd(submitDir)
%         write_name = strcat('GCP-CID',num2str(i),'_',num2str(b),'_sigma_',num2str(floor(modelSigma)),'.png');
%         imwrite(uint8(Denoised), write_name);
%         write_name = strcat('noisy_img',num2str(i),'_',num2str(b),'.png');
%         imwrite(uint8(noisy_ib), write_name);
%         cd(cur_folder)
% 
%         disp(['img_idx = ',num2str(i), '-',num2str(b),'.',' Time = ',num2str(Time_ib), ' Noise_mtx: ',num2str(noise_lvl_mtx_ib(:)')])
% 
%     end
% end
% 
% OptionalData.MethodName = 'DummyDenoiser';
% OptionalData.Authors = 'Jane Doe and Jone Doa';
% OptionalData.PaperTitle = 'Dummy Image Denoising';
% OptionalData.Venue = 'SIDD Demo';
% 
% TimeMPSrgb = 0;
% 
% fprintf('Saving resutls...\n');
% save(fullfile(submitDir, 'SubmitSrgb.mat'), 'DenoisedBlocksSrgb', ...
%     'TimeMPSrgb', 'OptionalData', '-v7.3');
% fprintf('Done!\n');

%% test DND sRGB Benchmark dataset
% clear all;
% file_name = 'noise_est_mtx_sRGB_DND.mat'; % noise_est_mtx_sRGB_DND_modified.mat
% load(file_name);
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% % parameter settings
% cur_folder = pwd;
% data_folder = 'DnD'; % You need to download the DND dataset
% output_folder = 'Submit_DND';
% infos = load(fullfile(data_folder, 'info.mat'));
% info = infos.info;
% 
% % Parameter setting
% disp('--------------------------------------------------------------')
% ps = 8; % patch size
% SR = 20; % search window size
% N_step = 4; % Nstep --> could be 4/6/8. Choose a smaller size for slightly better results. Choose a larger one for faster speed. 
% maxK = 30; % number of patches.  Choose a larger number for slightly better results. Choose a smaller one for faster speed. 
% divide_factor = 1.2; % controls the seach scheme
% modified = 1; tau = 1.1; global_learning = 1;
% disp([' ps: ', num2str(ps), ' maxK: ',num2str(maxK),  ' N_step: ',num2str(N_step)])
% disp([' SR: ', num2str(SR), ' tau: ',num2str(tau),  ' divide_factor: ',num2str(divide_factor), ' global_learning: ',num2str(global_learning)])
% disp('--------------------------------------------------------------')
% 
% N_imgs = 50; N_blocks = 20;
% % iterate over images
% for i=1:50
%     img = load(fullfile(data_folder, 'images_srgb', sprintf('%04d.mat', i)));
%     Inoisy = img.InoisySRGB;
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
%         nlf.sigma_all = info(i).sigma_srgb(b,:,:);
%         noisy_test = Inoisy_crop;
%         origin_test = noisy_test + 0.08;
%         noise_lvl_mtx_ib = squeeze(noise_est_mtx_DND(i,b,:,:));
%         
%         nlf.sigma_avg = mean(nlf.sigma_all(:));
%         modelSigma = mean(noise_lvl_mtx_ib(:));
%         
%         tic
%         [Denoised] = GCP_CID_with_mex(single(noisy_test), modelSigma/255, ps, maxK, SR, N_step, tau, divide_factor);
%         time = toc;
% 
%         disp(['image_i = ',num2str(i), ' image_crop_b = ', num2str(b), ' Noise_mtx: ',num2str(noise_lvl_mtx_ib(:)'), ' avg_noise_lvl = ',num2str(mean(noise_lvl_mtx_ib(:))), ' Time = ',num2str(time)])
%         
%         Idenoised_crop_bbs{b} = single(Denoised);
%         
%         cd(output_folder)
%         write_name = strcat('GCP_CID_','_',num2str(i),'_',num2str(b),'_sigma_',num2str(floor(modelSigma)),'.png');
%         imwrite(uint8(Denoised*255), write_name);
%         
%         write_name_noisy = strcat('Noisy_',num2str(i),'_',num2str(b),'_SigmaEst_',num2str(nlf.sigma_avg),'.png');
%         imwrite(uint8(noisy_test*255), write_name_noisy)
%         
%         cd(cur_folder)
%     end
%     
%     for b=1:20
%         Idenoised_crop = Idenoised_crop_bbs{b};
%         save(fullfile(output_folder, sprintf('%04d_%02d.mat', i, b)), 'Idenoised_crop');
%     end
%     
%     fprintf('Image %d/%d done\n', i,50);
% end

%% Test other real_world images (CC, PolyU, HighISO)
% Parameter setting
disp('--------------------------------------------------------------')
ps = 8; % patch size
SR = 20; % search window size
N_step = 4; % Nstep --> could be 4/6/8. Choose a smaller size for slightly better results. Choose a larger one for faster speed. 
maxK = 30; % number of patches.  Choose a larger number for slightly better results. Choose a smaller one for faster speed. 
divide_factor = 1.2; % controls the seach scheme
modified = 1; tau = 1.1; global_learning = 1;
disp([' ps: ', num2str(ps), ' maxK: ',num2str(maxK),  ' N_step: ',num2str(N_step)])
disp([' SR: ', num2str(SR), ' tau: ',num2str(tau),  ' divide_factor: ',num2str(divide_factor), ' global_learning: ',num2str(global_learning)])
disp('--------------------------------------------------------------')

% % CC15

GT_Original_image_dir = 'Data/CC15/';
GT_fpath = fullfile(GT_Original_image_dir, '*mean.png');
TT_Original_image_dir = 'Data/CC15/';
TT_fpath = fullfile(TT_Original_image_dir, '*real.png');

% % load noise est matrix
file_name = strcat('noise_est_mtx_CC15.mat');
load(file_name)
disp('------------------------------------------------------')
disp(file_name)
disp('------------------------------------------------------')
noise_est_mtx = noise_est_mtx_CC15;

% CC60
% disp('test dataset: CC60')
% GT_Original_image_dir = 'Data/CCImages/CC_60MeanImage/';
% GT_fpath = fullfile(GT_Original_image_dir, '*.png');
% TT_Original_image_dir = 'Data/CCImages/CC_60NoisyImage/';
% TT_fpath = fullfile(TT_Original_image_dir, '*.png');
% 
% % % load noise est matrix
% file_name = strcat('noise_est_mtx_CC60.mat');
% load(file_name)
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% noise_est_mtx = noise_est_mtx_CC60;
% 
% % Xujun-100 dataset
% disp('test dataset: PolyU')
% GT_Original_image_dir = 'Data/PolyU/';
% GT_fpath = fullfile(GT_Original_image_dir, '*mean.JPG');
% TT_Original_image_dir = 'Data/PolyU/';
% TT_fpath = fullfile(TT_Original_image_dir, '*real.JPG');
% % % load noise est matrix
% file_name = strcat('noise_est_mtx_PolyU.mat');
% load(file_name)
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% noise_est_mtx = noise_est_mtx_PolyU;


% HighISO
% disp('test dataset: HighISO')
% GT_Original_image_dir = 'Data/HighISO_Cropped/';
% GT_fpath = fullfile(GT_Original_image_dir, '*clean.png');
% TT_Original_image_dir = 'Data/HighISO_Cropped/';
% TT_fpath = fullfile(TT_Original_image_dir, '*noisy.png');
% % load noise est matrix
% file_name = strcat('noise_est_mtx_HighISO.mat');
% load(file_name)
% disp('------------------------------------------------------')
% disp(file_name)
% disp('------------------------------------------------------')
% noise_est_mtx = noise_est_mtx_HighISO;


GT_im_dir  = dir(GT_fpath);
TT_im_dir  = dir(TT_fpath);
im_num = length(TT_im_dir);

% % Choose pretrained models
k = 0; psnr_count = 0; ssim_count = 0;
for i = 1:im_num
    origin_test = double(imread(fullfile(GT_Original_image_dir, GT_im_dir(i).name)) );
    S = regexp(GT_im_dir(i).name, '\.', 'split');
    fprintf('%s :\n', GT_im_dir(i).name);
    noisy_test = double(imread(fullfile(TT_Original_image_dir, TT_im_dir(i).name)) );
    noise_lvl_mtx_i = squeeze(noise_est_mtx(i,:,:));
    sigma = mean(noise_lvl_mtx_i(:));
    
    tic
    [Denoised] = GCP_CID_with_mex_modified(noisy_test, noise_lvl_mtx_i, sigma, ps, maxK, SR, N_step, tau, divide_factor);
    time = toc;
    
    [psnr_h, im_ssim] = calculate_index(Denoised/255, origin_test/255);
    
    disp([num2str(time),' ',num2str(i),' ',num2str(sigma),' ',num2str(psnr_h),' ',num2str(im_ssim)]);
    psnr_count = psnr_count + psnr_h;
    ssim_count = ssim_count + im_ssim;
    k = k + 1;
    
end

disp(['sigma = ',num2str(sigma),' maxK = ',num2str(maxK),' psnr_average = ',num2str(psnr_count/k), ' ssim_average = ',num2str(ssim_count/k)]);
disp('-------------------------------------------------------');
disp('-------------------------------------------------------');

%% Test IOCI dataset (My Own dataset)
% Parameter setting
% disp('--------------------------------------------------------------')
% ps = 8; % patch size
% SR = 20; % search window size
% N_step = 4; % Nstep --> could be 4/6/8. Choose a smaller size for slightly better results. Choose a larger one for faster speed. 
% maxK = 30; % number of patches.  Choose a larger number for slightly better results. Choose a smaller one for faster speed. 
% divide_factor = 1.2; % controls the seach scheme
% modified = 1; tau = 1.1; global_learning = 1;
% disp([' ps: ', num2str(ps), ' maxK: ',num2str(maxK),  ' N_step: ',num2str(N_step)])
% disp([' SR: ', num2str(SR), ' tau: ',num2str(tau),  ' divide_factor: ',num2str(divide_factor), ' global_learning: ',num2str(global_learning)])
% disp('--------------------------------------------------------------')
% 
% camera_set = {'XIAOMI8','SONY_A6500','OPPO_R11s','IPHONE_6S','IPHONE_5S','CANON_600D','HUAWEI_honor6X','CANON_100D','Fujifilm_X100T', 'NIKON_D5300', 'IPHONE13', 'HUAWEI_Mate40Pro', 'CANON_5D_Mark4'};
% data_root_path = 'IOCI\';
% 
% for image_set = 1:13
%     
%     Camera_name = camera_set{image_set};
%     disp(['*******************',Camera_name,'**********************']);
%     data_folder = strcat(data_root_path, Camera_name);
%     addpath(data_folder);
%     img_path = data_folder;
%     fpath = fullfile(img_path, '*.bmp');
%     
%     all_name = dir(fpath);
%     noisy_name = dir(fullfile(img_path, '*noisy.bmp'));
%     clean_name = dir(fullfile(img_path, '*clean.bmp'));
%     
%     im_num = length(noisy_name);
%     
%     file_name = strcat('noise_est_mtx_',Camera_name,'.mat');
%     noise_lvl_mtx = load(file_name);
%     noise_est_mtx = noise_lvl_mtx.noise_est_mtx;
%     disp('------------------------------------------------------')
%     disp(file_name)
%     disp('------------------------------------------------------')
%     
%     psnr_count = 0; ssim_count = 0; k = 0;
%     psnr_noisy = 0; ssim_noisy = 0; elapsed_time = 0;
%     for i = 1:im_num
%         % Read images
%         origin_test = single(imread(clean_name(i).name));
%         S = regexp(noisy_name(i).name, '\.', 'split');
%         fprintf('%s :\n', noisy_name(i).name);
%         [w, h, ch] = size(origin_test);
%         noisy_test = single(imread(noisy_name(i).name));
%         
%         noise_lvl_mtx_i = squeeze(noise_est_mtx(i,:,:));
%         sigma = mean(noise_lvl_mtx_i(:));
%         
%         tic
%         [Denoised] = GCP_CID_with_mex_modified(single(noisy_test), noise_lvl_mtx_i, sigma, ps, maxK, SR, N_step, tau, divide_factor);
%         
%         [psnr_n, ssim_n] = calculate_index(double(noisy_test)/255, double(origin_test)/255);
%         [psnr_denoised, ssim_denoised] = calculate_index(double(Denoised)/255, double(origin_test)/255);
%         time = toc;
%         elapsed_time = elapsed_time + time;
%         
%         disp(['img: ', num2str(i), ' time: ', num2str(time),' avg_noise_lvl: ',num2str(sigma),' psnr_noisy: ',num2str(psnr_n),' psnr_denoised: ', num2str(psnr_denoised), ' ssim_noisy: ',num2str(ssim_n),' ssim_denoised: ',num2str(ssim_denoised)]);
%         psnr_count = psnr_count + psnr_denoised;
%         ssim_count = ssim_count + ssim_denoised;
%         psnr_noisy = psnr_noisy + psnr_n;
%         ssim_noisy = ssim_noisy + ssim_n;
%         
%         k = k + 1;
%         
%         % Write images
%         cur_path = pwd;
%         write_root_path = 'output\IOCI\';
%         write_path = strcat(write_root_path, Camera_name);
%         cd(write_path)
%         noisy_img_name = noisy_name(i).name;
%         write_name = strcat('GCP_ID+CNN_',noisy_img_name);
%         imwrite(uint8(Denoised),write_name);
%         cd(cur_path)
%         
%         avg_psnr_noisy = psnr_noisy / i; avg_ssim_noisy = ssim_noisy / i;
%         avg_psnr_denoised = psnr_count / i; avg_ssim_denoised = ssim_count / i;
%         avg_elapsed_time = elapsed_time / i;
%         
%         disp('############################## print statistics #####################################')
%         disp(['avg_elapsed_time = ', num2str(avg_elapsed_time), ' avg_psnr_noisy = ', num2str(avg_psnr_noisy), ' avg_psnr_denoised = ', num2str(avg_psnr_denoised), ' avg_ssim_noisy = ', num2str(avg_ssim_noisy), ' avg_ssim_denoised = ', num2str(avg_ssim_denoised)]);
%         disp('############################## print statistics #####################################')
%     end
%     
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
[mssim1, ~] = ssim_index(denoised(:,:,1),clean(:,:,1),K,window,L);
[mssim2, ~] = ssim_index(denoised(:,:,2),clean(:,:,2),K,window,L);
[mssim3, ~] = ssim_index(denoised(:,:,3),clean(:,:,3),K,window,L);
SSIM = (mssim1+mssim2+mssim3)/3.0;

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

