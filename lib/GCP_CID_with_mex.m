function [im_denoised] = GCP_CID_with_mex(im1, sigma, ps, maxK, SR, N_step, tau, divide_factor)
%% The following code implements GCP_CID with mex functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[H,W,D] = size(im1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% compute global U and V with all reference cubes.
load U_3D; load V_3D;
load U_4D; load V_4D;
%% determine patch search scheme
count_green_mode = 0;
for i=1:N_step:H-ps+1 
    for j=1:N_step:W-ps+1
        count_green_mode = count_green_mode + 1;
    end
end

green_mode_all = zeros(count_green_mode,1);
count_green_mode = 1;

for i=1:N_step:H-ps+1 
    for j=1:N_step:W-ps+1

        patch_ij = im1(i:i+ps-1,j:j+ps-1,:);
        
        patch_mode = decide_mode(patch_ij, divide_factor);
        if strcmp(patch_mode, 'green') == 1
            green_mode_all(count_green_mode) = 1;
        end
        count_green_mode = count_green_mode + 1;

    end
end

%% compute local similarity with a simple mex function
info1 = [H,W,D];
info2 = [ps,N_step,SR,maxK];
info3 = [tau,sigma];
[similar_indice] = GCP_CID_search(single(im1), int32(green_mode_all), int32(info1),int32(info2), single(info3));
%% Perform denoising
U_3D = fft(U_3D, [], 3); V_3D = fft(V_3D, [], 3);
U_4D = fft(U_4D, [], 3); V_4D = fft(V_4D, [], 3);

im2 = GCP_CID(single(im1), int32(similar_indice),single(U_3D),single(V_3D), single(U_4D),single(V_4D), int32(green_mode_all), int32(info1),int32(info2),single(info3));
im_denoised = mat_ten(im2,1,size(im1));
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

