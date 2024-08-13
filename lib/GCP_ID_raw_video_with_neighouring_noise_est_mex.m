function [denoised_video] = CMStSVD_raw_video_with_neighouring_noise_est_mex(noisy_video, noise_est_matrix, ps, N_step_spatial, N_step_time, SR, maxK, tau, global_est_nois)
%CVMSTSVD_DENOISING Using MATLAB implementation
%% compute global U and V

[H_origin, W_origin, C, N] =  size(noisy_video);

[U_4D,V_4D] = compute_global_4D(noisy_video,N_step_spatial,ps);

noisy_video_enlarge = enlarge_video(noisy_video, 128);

%% Obtain local noise_est_mtx
frame_chosen = 3;
im_noisy_frame = noisy_video_enlarge(:,:,:,frame_chosen);
noise_lvl_vectors = determin_noise_lvl_vectors(im_noisy_frame, ps, N_step_spatial, noise_est_matrix);
noise_lvl_vectors = noise_lvl_vectors/255;
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
    denoised_MSTSVD = color_video_denoising_global_Eigen_global_est_noise_RGGB(single(noisy_video_enlarge),single(U_4D),single(V_4D),info1,info2,single(info3));
    denoised_video_enlarge = mat_ten(denoised_MSTSVD,1,size(noisy_video_enlarge));
    denoised_video = denoised_video_enlarge(1:H_origin, 1:W_origin, :, :);
elseif(global_est_nois == 0)
    %     denoised_MSTSVD = color_video_denoising_global_Eigen_local_est_noise_RGGB(single(noisy_video_enlarge),single(U_4D),single(V_4D),info1,info2,single(info3),single(noise_est_matrix));
    denoised_MSTSVD = color_video_denoising_globa_est_noise_by_neighbours_RGGB(single(noisy_video_enlarge),single(U_4D),single(V_4D),info1,info2,single(info3),single(noise_lvl_vectors));
    denoised_video_enlarge = mat_ten(denoised_MSTSVD,1,size(noisy_video_enlarge));
    denoised_video = denoised_video_enlarge(1:H_origin, 1:W_origin, :, :);
end

end



%% Other functions


function video_reshape = enlarge_video(video, factor)
[H,W,C,N] = size(video);
H_new = (ceil(H/factor)) * factor;
W_new = (ceil(W/factor)) * factor;
video_reshape = zeros(H_new, W_new, C, N);
video_reshape(1:H, 1:W, :,:) = video;
video_reshape = single(video_reshape);

end

% Compute Global 4D transforms U and V
function [U,V] = compute_global_4D(im1,N_step,ps)
[H,W,D,T] = size(im1);
count = 0;
for t = 1:3:T
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

function noise_lvl_vectors = determin_noise_lvl_vectors(img, ps, N_step, noise_lvl_mtx)

[H,W,~] = size(img);
step_size = 128;
count_num = 0;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?\
        count_num = count_num + 1;
    end
end

noise_lvl_vectors = zeros(count_num, 1);
count_num = 0;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        
        noise_lvl_ij = determine_local_noise_lvl_by_neighbours(i, j, step_size, noise_lvl_mtx);
        noiselevel = noise_lvl_ij;
        count_num = count_num + 1;
%         noise_lvl_vectors(count_num) = mean(noise_lvl_mtx(:));
        noise_lvl_vectors(count_num) = noiselevel;
    end
end

end

function noise_lvl_ij_avg = determine_local_noise_lvl_by_neighbours(i, j, step_size, noise_lvl_mtx)

[H,W] = size(noise_lvl_mtx);

idx_i = ceil(i/step_size);
idx_j = ceil(j/step_size);

sr_top = max([idx_i-3 1]);
sr_left = max([idx_j-3 1]);
sr_right = min([idx_j+3 W]);
sr_bottom = min([idx_i+3 H]);

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

