function [denoised_MSTSVD] = GCP_CID_color_video(noisy_video, U_3D, V_3D, U_4D, V_4D, ps, N_step_spatial, N_step_time, SR, maxK, sigma, tau, global_learning, divide_ratio)
%CVMSTSVD_DENOISING Using MATLAB implementation

%% compute global U and V

[H,W,D,N] = size(noisy_video);
im2 = zeros(H,W,D,N);
numcount = zeros(H,W,D,N);
modified = 1;

imRed = squeeze(noisy_video(:,:,1,:));
imGreen = squeeze(noisy_video(:,:,2,:));
imBlue = squeeze(noisy_video(:,:,3,:));

% if global_learning == 1
%     [U_3D,V_3D] = compute_global_3D(noisy_video,N_step_spatial,ps);
%     [U_4D,V_4D] = compute_global_4D(noisy_video,N_step_spatial,ps);
% end

%% Denoising using GCP 

im1_F = imRed + imGreen + imBlue;

% if global_learning == 1
%     [U_3D,V_3D] = compute_global_3D(noisy_video,2*N_step_spatial,ps);
%     [U_4D,V_4D] = compute_global_4D(noisy_video,2*N_step_spatial,ps);
%      load U_3D; load V_3D; load U_4D; load V_4D;
% end

U_3D = fft(U_3D,[],3); V_3D = fft(V_3D,[],3);
U_4D = fft(U_4D,[],3); V_4D = fft(V_4D,[],3);

patch_mode = 'green_extra';

for t = 1:N_step_time:N
    
    cur_frame = noisy_video(:,:,:,t);
    
    for i=1:N_step_spatial:H-ps+1 %why start from i=104?
        for j=1:N_step_spatial:W-ps+1 %why start from j=49?
            
            patch_ij = cur_frame(i:i+ps-1,j:j+ps-1,:);
            
            patch_mode = decide_mode(patch_ij, divide_ratio);
            if strcmp(patch_mode, 'green_extra') == 1
                noiselevel = sigma;
                refpatch = imGreen(i:i+ps-1,j:j+ps-1,t);
            elseif strcmp(patch_mode, 'normal') == 1
                noiselevel = sigma;
                refpatch = im1_F(i:i+ps-1,j:j+ps-1,t);
            end
            
            
            sr_top = max([i-SR 1]);
            sr_left = max([j-SR 1]);
            sr_right = min([j+SR W-ps+1]);
            sr_bottom = min([i+SR H-ps+1]);
            
            count = 0;
            similarity_indices = zeros(10*(2*SR+1)^2,3);
            distvals = similarity_indices(:,1); %distance value of refpatch and each target patch.
            
            start_frame = max(1, t - 3);
            end_frame = min(t+3, N);
            
            for t1 = start_frame:end_frame
                
                for i1=sr_top:sr_bottom
                    for j1=sr_left:sr_right

                        if strcmp(patch_mode, 'green_extra') == 1
                            currpatch = imGreen(i1:i1+ps-1,j1:j1+ps-1,t1);                            
                        elseif strcmp(patch_mode, 'normal') == 1
                            currpatch = im1_F(i1:i1+ps-1,j1:j1+ps-1,t1);
                        end
                        
                        dist = sum((refpatch(:)-currpatch(:)).^2);
                        count = count+1;
                        distvals(count) = dist;
                        similarity_indices(count,:) = [i1 j1 t1];
                        
                    end
                end
                
            end
            
            similarity_indices(1,:)=[i j t];
            similarity_indices = similarity_indices(1:count,:);
            distvals = distvals(1:count);
            
            if count > maxK
                [~,sortedindices] = sort(distvals,'ascend');
                similarity_indices = similarity_indices(sortedindices(1:maxK),:);
                count = maxK;
            end
            
            A = zeros(ps,ps,D,count,'single'); % construct a 4-D tensor with count patches
            A_F = zeros(ps,ps,count,'single'); % construct a 4-D tensor with count patches
            for k=1:count
                yindex = similarity_indices(k,1);
                xindex = similarity_indices(k,2);
                tindex = similarity_indices(k,3);
                A(:,:,:,k) = noisy_video(yindex:yindex+ps-1,xindex:xindex+ps-1,:,tindex);
                
                if strcmp(patch_mode, 'green_extra') == 1
                    A_F(:,:,k) = imGreen(yindex:yindex+ps-1,xindex:xindex+ps-1,tindex);
                elseif strcmp(patch_mode, 'normal') == 1
                    A_F(:,:,k)  = im1_F(yindex:yindex+ps-1,xindex:xindex+ps-1,tindex);
                end
            
            end
            
            
            if strcmp(patch_mode, 'green_extra') == 1
                
                A_learn = zeros(ps,ps,4,count,'single');
                A_learn(:,:,1,:) = A(:,:,1,:);
                A_learn(:,:,2,:) = A(:,:,2,:);
                A_learn(:,:,3,:) = A(:,:,2,:);
                A_learn(:,:,4,:) = A(:,:,3,:);
                
                A = A_learn;
                
            end
            
            
            if(modified == 1)
                mat_A = my_tenmat(A,ndims(A));
                mat_A_F = my_tenmat(A_F,ndims(A_F));
                [U4] = train_U4(mat_A_F);
                %             [U4,~] = eig(mat_A*mat_A');
                size_A = size(A);
                mat_A = U4'*mat_A;A = mat_ten(mat_A,ndims(A),size_A);
            end
            
            if global_learning == 0
                [U,~,V] = NL_t_svd(A);
            elseif strcmp(patch_mode, 'green_extra') == 1
                U = U_4D; V = V_4D;
            else
                U = U_3D; V = V_3D;
            end
            
            A = fft(A,[],3);
            
            if(count == 1)
            else
                A=threshold(A,U,V,noiselevel,tau);
            end
            
            A = ifft(A,[],3);
            
            if(modified == 1)
                A = my_ttm(A,U4,ndims(A),'nt');
            end
            
            if strcmp(patch_mode, 'green_extra') == 1
                
                A_denoised = zeros(ps,ps,3,count,'single');
                A_denoised(:,:,1,:) = A(:,:,1,:);
                A_denoised(:,:,2,:) = A(:,:,2,:);
                A_denoised(:,:,3,:) = A(:,:,4,:);
                A = A_denoised;
                
            end
            
            for k=1:count
                yindex = similarity_indices(k,1);
                xindex = similarity_indices(k,2);
                tindex = similarity_indices(k,3);
                im2(yindex:yindex+ps-1,xindex:xindex+ps-1,:,tindex) = im2(yindex:yindex+ps-1,xindex:xindex+ps-1,:,tindex)+A(:,:,:,k);
                numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,:,tindex) = numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,:,tindex)+1;
            end
            
        end
    end
    
end

ind_zero = numcount==0;
numcount(ind_zero) = 1;
im2(ind_zero) = noisy_video(ind_zero);
denoised_MSTSVD = im2./numcount;

end



%% Other functions

% Decide patch mode
function patch_mode = decide_mode(refpatch, divide_ratio)

patchR = refpatch(:,:,1);
patchG = refpatch(:,:,2);
patchB = refpatch(:,:,3);

R_sum = sum(patchR(:));
G_sum = sum(patchG(:));
B_sum = sum(patchB(:));

R_norm = norm(patchR(:));
G_norm = norm(patchG(:));
B_norm = norm(patchB(:));

if G_norm > (B_norm/divide_ratio)  && G_norm > (R_norm/divide_ratio) 
    patch_mode = 'green_extra';
else
    patch_mode = 'normal';
end

end

% Compute Global 3D transforms U and V
function [U,V] = compute_global_3D(im1,N_step,ps)
[H,W,D,N] = size(im1);
count = 0;
for t = 1:2:N
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

% Compute Global 4D transforms U and V
function [U,V] = compute_global_4D(im1,N_step,ps)
[H,W,D,N] = size(im1);
count = 0;
for t = 1:2:N
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

A_learn = zeros(ps,ps,4,count,'single');
A_learn(:,:,1,:) = A(:,:,1,:);
A_learn(:,:,2,:) = A(:,:,2,:);
A_learn(:,:,3,:) = A(:,:,2,:);
A_learn(:,:,4,:) = A(:,:,3,:);

[U,V]=NL_tSVD(A_learn);
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

% Perform the hard thresholding operation
function A_thre = threshold(A,U,V,sigma,tau)
size_A = size(A);ps = size_A(1);D = size_A(3);count = size_A(end);
A_thre = zeros(size_A);real_count = floor(D/2)+1; 
% hard-threshold parameter
coeff_threshold = tau*sigma*sqrt(2*log(ps*ps*D*count));
for i = 1:real_count
    Ai = A(:,:,i,:);
    S1 = my_ttm(Ai,U(:,:,i),1,'t'); S = my_ttm(S1,V(:,:,i),2,'t');
    % hard-threshold
    S(abs(S(:)) < coeff_threshold) = 0;
    A1 = my_ttm(S,U(:,:,i),1,'nt'); A_f = my_ttm(A1,V(:,:,i),2,'nt');
    A_thre(:,:,i,:) = A_f;
end

A_thre = com_conj(A_thre);

end

% function A_thre = threshold(A,U,V,sigma,tau)
% size_A = size(A);ps = size_A(1);D = size_A(3);count = size_A(end);
% A_thre = zeros(size_A); U = fft(U,[],3); V = fft(V,[],3);
% % hard-threshold parameter
% coeff_threshold = tau*sigma*sqrt(2*log(ps*ps*D*count));
% for i = 1:D
%     Ai = A(:,:,i,:);
%     S1 = my_ttm(Ai,U(:,:,i),1,'t'); S = my_ttm(S1,V(:,:,i),2,'t');
%     % hard-threshold
%     S(abs(S(:)) < coeff_threshold) = 0;
%     A1 = my_ttm(S,U(:,:,i),1,'nt'); A_f = my_ttm(A1,V(:,:,i),2,'nt');
%     A_thre(:,:,i,:) = A_f;
% end
% 
% end

% Comduct conjugate operation
function A  = com_conj(A)
[~,~,D,~] = size(A);
k = 0;
for i = 2:floor(D/2)+1
    A(:,:,D-k,:) = conj(A(:,:,i,:));
    k = k + 1;
end
end

% Compute local group-level representation
function [U4] = train_U4(A)

[num_patch,~] = size(A);
sum_A = sum(A,1);
average_A = sum_A/num_patch;
A_hat = A - average_A;
[U4,~] = eig(A_hat*A_hat');

end


