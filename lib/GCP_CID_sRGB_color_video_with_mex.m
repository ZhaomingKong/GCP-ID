function [im_denoised] = GCP_CID_sRGB_color_video_with_mex(im1, noise_lvl_mtx, global_est, ps, maxK, SR, N_step_spatial, N_step_time, tau, divide_factor)
%% The following code implements GCP_CID with mex functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[H,W,~,D] = size(im1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute global U and V with all reference cubes.
load U_3D; load V_3D;
load U_4D; load V_4D;
%% determine patch search scheme
count_green_mode = 0;
for k = 1:N_step_time:D
    for i=1:N_step_spatial:H-ps+1
        for j=1:N_step_spatial:W-ps+1
            count_green_mode = count_green_mode + 1;
        end
    end
end

green_mode_all = zeros(count_green_mode,1);
count_green_mode = 1;

for k = 1:N_step_time:D
    for i=1:N_step_spatial:H-ps+1
        for j=1:N_step_spatial:W-ps+1
            
            patch_ij = squeeze(im1(i:i+ps-1,j:j+ps-1,:,k));
            
            patch_mode = decide_mode(patch_ij, divide_factor);
            if strcmp(patch_mode, 'green') == 1
                green_mode_all(count_green_mode) = 1;
            end
            count_green_mode = count_green_mode + 1;
            
        end
    end
end

%% Obtain local noise_est_mtx
frame_chosen = 3;
im_frame = im1(:,:,:,frame_chosen);
noise_lvl_vectors = determin_noise_lvl_vectors(im_frame, ps, N_step_spatial, noise_lvl_mtx);
sigma = mean(noise_lvl_mtx(:));
%% compute local similarity with a simple mex function
modified = 1;
info1 = [H,W,D];
info2 = [ps, N_step_spatial, N_step_time, SR, maxK];
info3 = [tau,modified,sigma];
U_3D_fft = fft(U_3D, [], 3); V_3D_fft = fft(V_3D, [], 3);
U_4D_fft = fft(U_4D, [], 3); V_4D_fft = fft(V_4D, [], 3);

if(global_est == 1)   
    im2 = GCP_CID_video_denoising(single(im1), int32(green_mode_all), single(U_3D_fft),single(V_3D_fft), single(U_4D_fft),single(V_4D_fft), int32(info1), int32(info2), single(info3));
    im_denoised = mat_ten(im2,1,size(im1));
elseif(global_est == 0)
    im2 = GCP_CID_video_denoising_local_neighbour_noise_est(single(im1), int32(green_mode_all), single(U_3D_fft), single(V_3D_fft), single(U_4D_fft), single(V_4D_fft), int32(info1), int32(info2), single(info3), single(noise_lvl_vectors));
    im_denoised = mat_ten(im2,1,size(im1));
end


end


%% Related functions


function patch_mode = decide_mode(refpatch, divide_factor)

patchR = refpatch(:,:,1);
patchG = refpatch(:,:,2);
patchB = refpatch(:,:,3);

R_norm = norm(patchR(:));
G_norm = norm(patchG(:));
B_norm = norm(patchB(:));

if G_norm > (B_norm/divide_factor)  && G_norm > (R_norm/divide_factor) 
    patch_mode = 'green';
else
    patch_mode = 'normal';
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
        noiselevel = noise_lvl_ij;
        noise_lvl_vectors(count_num) = noiselevel;
        count_num = count_num + 1; 
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

