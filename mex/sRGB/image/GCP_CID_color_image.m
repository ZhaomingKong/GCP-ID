function [im2,psnr,im_ssim] = GCP_CID_color_image(im1, im, ps, SR, sigma, maxK, N_step, modified, tau, global_learning, divide_factor)
%   Detailed explanation goes here

%% compute global U and V

[H,W,D] = size(im1);
im2 = zeros(H,W,D);
numcount = zeros(H,W,D);
info1 = [H,W,D];
info2 = [ps,N_step,SR,maxK];
info3 = [tau,sigma,modified];

imRed = im1(:,:,1);
imGreen = im1(:,:,2);
imBlue = im1(:,:,3);

if global_learning == 1
    [U,V] = compute_global_4D(im1,N_step,ps);
end

%% adaptive patch search using only green channels 

im1_F = imRed + imGreen + imBlue;
% im1_F = imGreen;

if global_learning == 1
    [U_3D,V_3D] = compute_global_3D(im1,N_step,ps);
    [U_4D,V_4D] = compute_global_4D(im1,N_step,ps);
end

patch_mode = 'green';

for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        
        patch_ij = im1(i:i+ps-1,j:j+ps-1,:);
        
        patch_mode = decide_mode(patch_ij, divide_factor);
        if strcmp(patch_mode, 'green_extra') == 1
            noiselevel = sigma;
            simpatch = maxK;
            refpatch = imGreen(i:i+ps-1,j:j+ps-1);
        elseif strcmp(patch_mode, 'green') == 1
            noiselevel = sigma;
            simpatch = maxK;
            refpatch = imGreen(i:i+ps-1,j:j+ps-1);
        elseif strcmp(patch_mode, 'normal') == 1
            noiselevel = sigma;
            simpatch = maxK;
            refpatch = im1_F(i:i+ps-1,j:j+ps-1);    
        end
        
        sr_top = max([i-SR 1]);
        sr_left = max([j-SR 1]);
        sr_right = min([j+SR W-ps+1]);
        sr_bottom = min([i+SR H-ps+1]);

        count = 0;
        similarity_indices = zeros((2*SR+1)^2,2);

        distvals = similarity_indices(:,1); %distance value of refpatch and each target patch.
        for i1=sr_top:sr_bottom
            for j1=sr_left:sr_right
   
                if strcmp(patch_mode, 'green_extra') == 1 
                    currpatch = imGreen(i1:i1+ps-1,j1:j1+ps-1);
                    
                elseif strcmp(patch_mode, 'green') == 1
                    currpatch = imGreen(i1:i1+ps-1,j1:j1+ps-1);   
                    
                elseif strcmp(patch_mode, 'normal') == 1
                    currpatch = im1_F(i1:i1+ps-1,j1:j1+ps-1);
                end
                
                dist = sum((refpatch(:)-currpatch(:)).^2);
                count = count+1;
                distvals(count) = dist;
                similarity_indices(count,:) = [i1 j1];
            end
        end
        
        similarity_indices(1,:)=[i j];
        similarity_indices = similarity_indices(1:count,:);
        distvals = distvals(1:count);

        if count > simpatch
            [~,sortedindices] = sort(distvals,'ascend');
            similarity_indices = similarity_indices(sortedindices(1:simpatch),:);
            count = simpatch;
        end

        A = zeros(ps,ps,3,count,'single'); % construct a 4-D tensor with count patches
        A_F = zeros(ps,ps,count,'single');
        for k=1:count
            yindex = similarity_indices(k,1);
            xindex = similarity_indices(k,2);
            A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
            
            if strcmp(patch_mode, 'green_extra') == 1
                A_F(:,:,k) = imGreen(yindex:yindex+ps-1,xindex:xindex+ps-1);
            elseif strcmp(patch_mode, 'green') == 1
                A_F(:,:,k) = imGreen(yindex:yindex+ps-1,xindex:xindex+ps-1);
            elseif strcmp(patch_mode, 'normal') == 1
                A_F(:,:,k)  = im1_F(yindex:yindex+ps-1,xindex:xindex+ps-1);
            end
            
        end
        
        
        if strcmp(patch_mode, 'green_extra') == 1
            
            A_learn = zeros(ps,ps,4,count,'single');
            A_learn(:,:,1,:) = A(:,:,2,:);
            A_learn(:,:,2,:) = A(:,:,2,:);
            A_learn(:,:,3,:) = A(:,:,1,:);
            A_learn(:,:,4,:) = A(:,:,3,:);
            
            A = A_learn;
            
        end
                
        if(modified == 1)
            mat_A = my_tenmat(A,ndims(A));
            mat_A_F = my_tenmat(A_F,ndims(A_F));
            [U4] = train_U4(mat_A_F);
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
            A_denoised(:,:,1,:) = A(:,:,3,:);
            A_denoised(:,:,2,:) = A(:,:,1,:);
            A_denoised(:,:,3,:) = A(:,:,4,:);
            A = A_denoised;
            
        end                          

        for k=1:count
            yindex = similarity_indices(k,1);
            xindex = similarity_indices(k,2);
            im2(yindex:yindex+ps-1,xindex:xindex+ps-1,:) = im2(yindex:yindex+ps-1,xindex:xindex+ps-1,:)+A(:,:,:,k);
            numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,:) = numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,:)+1;
        end

    end
end


ind_zero = numcount==0;numcount(ind_zero) = 1;
im2(ind_zero) = im1(ind_zero);
im2 = im2./numcount;

im2=double(im2);im=double(im);

mse = sum((im2(:)-im(:)).^2)/(H*W*D);

psnr = 10*log10(255*255/mse);

im_ssim = cal_ssim(im2,im,0,0);


end


function patch_mode = decide_mode(refpatch, divide_factor)

patchR = refpatch(:,:,1);
patchG = refpatch(:,:,2);
patchB = refpatch(:,:,3);

R_sum = sum(patchR(:));
G_sum = sum(patchG(:));
B_sum = sum(patchB(:));

R_norm = norm(patchR(:));
G_norm = norm(patchG(:));
B_norm = norm(patchB(:));

if G_norm > (B_norm/divide_factor)  && G_norm > (R_norm/divide_factor) 
    patch_mode = 'green_extra';
% elseif G_norm > B_norm/2 && G_norm > R_norm/2
%     patch_mode = 'green';
else
    patch_mode = 'normal';
end

end

%% adaptive patch search using different channels, including both red and blue channels

% im1_F = imRed + imGreen + imBlue;
% % im1_F = imGreen;
% 
% if global_learning == 1
%     [U_3D,V_3D] = compute_global_3D(im1,N_step,ps);
%     [U_4D_red, V_4D_red] = compute_global_4D_by_channel(im1,N_step,ps, 'red');
%     [U_4D_green, V_4D_green] = compute_global_4D_by_channel(im1,N_step,ps, 'green');
%     [U_4D_blue, V_4D_blue] = compute_global_4D_by_channel(im1,N_step,ps, 'blue');
% end
% 
% patch_mode = 'green';
% 
% for i=1:N_step:H-ps+1 %why start from i=104?
%     for j=1:N_step:W-ps+1 %why start from j=49?
%         
%         patch_ij = im1(i:i+ps-1,j:j+ps-1,:);
%         
%         patch_mode = decide_mode(patch_ij);
%         if strcmp(patch_mode, 'green_extra') == 1
%             noiselevel = sigma;
%             simpatch = maxK;
%             refpatch = imGreen(i:i+ps-1,j:j+ps-1);
%         elseif strcmp(patch_mode, 'red_extra') == 1
%             noiselevel = sigma - 10;
%             simpatch = maxK;
%             refpatch = imRed(i:i+ps-1,j:j+ps-1);
%         elseif strcmp(patch_mode, 'blue_extra') == 1
%             noiselevel = sigma - 10;
%             simpatch = maxK;
%             refpatch = imBlue(i:i+ps-1,j:j+ps-1);
%         elseif strcmp(patch_mode, 'green') == 1
%             noiselevel = sigma;
%             simpatch = maxK;
%             refpatch = imGreen(i:i+ps-1,j:j+ps-1);
%         elseif strcmp(patch_mode, 'normal') == 1
%             noiselevel = sigma + 5;
%             simpatch = maxK;
%             refpatch = im1_F(i:i+ps-1,j:j+ps-1);    
%         end
%         
%         sr_top = max([i-SR 1]);
%         sr_left = max([j-SR 1]);
%         sr_right = min([j+SR W-ps+1]);
%         sr_bottom = min([i+SR H-ps+1]);
% 
%         count = 0;
%         similarity_indices = zeros((2*SR+1)^2,2);
% 
%         distvals = similarity_indices(:,1); %distance value of refpatch and each target patch.
%         for i1=sr_top:sr_bottom
%             for j1=sr_left:sr_right
%    
%                 if strcmp(patch_mode, 'green_extra') == 1 
%                     currpatch = imGreen(i1:i1+ps-1,j1:j1+ps-1);
%                 
%                 elseif strcmp(patch_mode, 'red_extra') == 1
%                     currpatch = imRed(i1:i1+ps-1,j1:j1+ps-1);
%                     
%                 elseif strcmp(patch_mode, 'blue_extra') == 1 
%                     currpatch = imBlue(i1:i1+ps-1,j1:j1+ps-1);
%                     
%                 elseif strcmp(patch_mode, 'green') == 1
%                     currpatch = imGreen(i1:i1+ps-1,j1:j1+ps-1);   
%                     
%                 elseif strcmp(patch_mode, 'normal') == 1
%                     currpatch = im1_F(i1:i1+ps-1,j1:j1+ps-1);
%                 end
%                 
%                 dist = sum((refpatch(:)-currpatch(:)).^2);
%                 count = count+1;
%                 distvals(count) = dist;
%                 similarity_indices(count,:) = [i1 j1];
%             end
%         end
%         
%         similarity_indices(1,:)=[i j];
%         similarity_indices = similarity_indices(1:count,:);
%         distvals = distvals(1:count);
% 
%         if count > simpatch
%             [~,sortedindices] = sort(distvals,'ascend');
%             similarity_indices = similarity_indices(sortedindices(1:simpatch),:);
%             count = simpatch;
%         end
% 
%         A = zeros(ps,ps,3,count,'single'); % construct a 4-D tensor with count patches
%         A_F = zeros(ps,ps,count,'single');
%         for k=1:count
%             yindex = similarity_indices(k,1);
%             xindex = similarity_indices(k,2);
%             A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
%             
%             if strcmp(patch_mode, 'green_extra') == 1
%                 A_F(:,:,k) = imGreen(yindex:yindex+ps-1,xindex:xindex+ps-1);
%             elseif strcmp(patch_mode, 'red_extra') == 1
%                 A_F(:,:,k) = imRed(yindex:yindex+ps-1,xindex:xindex+ps-1);
%             elseif strcmp(patch_mode, 'blue_extra') == 1
%                 A_F(:,:,k) = imBlue(yindex:yindex+ps-1,xindex:xindex+ps-1);
%             elseif strcmp(patch_mode, 'green') == 1
%                 A_F(:,:,k) = imGreen(yindex:yindex+ps-1,xindex:xindex+ps-1);
%             elseif strcmp(patch_mode, 'normal') == 1
%                 A_F(:,:,k)  = im1_F(yindex:yindex+ps-1,xindex:xindex+ps-1);
%             end
%             
%         end
%         
%         if strcmp(patch_mode, 'red_extra') == 1           
%             A_learn = zeros(ps,ps,4,count,'single');
%             A_learn(:,:,1,:) = A(:,:,1,:);
%             A_learn(:,:,2,:) = A(:,:,2,:);
%             A_learn(:,:,3,:) = A(:,:,3,:);
%             A_learn(:,:,4,:) = A(:,:,1,:);      
%             A = A_learn;
%         end
%         
%         
%         if strcmp(patch_mode, 'green_extra') == 1         
%             A_learn = zeros(ps,ps,4,count,'single');
%             A_learn(:,:,1,:) = A(:,:,1,:);
%             A_learn(:,:,2,:) = A(:,:,2,:);
%             A_learn(:,:,3,:) = A(:,:,3,:);
%             A_learn(:,:,4,:) = A(:,:,2,:);         
%             A = A_learn;          
%         end
%         
%         if strcmp(patch_mode, 'blue_extra') == 1           
%             A_learn = zeros(ps,ps,4,count,'single');
%             A_learn(:,:,1,:) = A(:,:,1,:);
%             A_learn(:,:,2,:) = A(:,:,2,:);
%             A_learn(:,:,3,:) = A(:,:,3,:);
%             A_learn(:,:,4,:) = A(:,:,3,:);   
%             A = A_learn;     
%         end
%         
%                 
%         if(modified == 1)
%             mat_A = my_tenmat(A,ndims(A));
%             mat_A_F = my_tenmat(A_F,ndims(A_F));
%             [U4] = train_U4(mat_A_F);
%             size_A = size(A);
%             mat_A = U4'*mat_A;A = mat_ten(mat_A,ndims(A),size_A);
%         end
%         
%         
%         if global_learning == 0
%             [U,~,V] = NL_t_svd(A);
%         elseif strcmp(patch_mode, 'green_extra') == 1
%             U = U_4D_green; V = V_4D_green;
%         elseif strcmp(patch_mode, 'red_extra') == 1
%             U = U_4D_red; V = V_4D_red;
%         elseif strcmp(patch_mode, 'blue_extra') == 1
%             U = U_4D_blue; V = V_4D_blue;
%         else
%             U = U_3D; V = V_3D;
%         end
%         
%         A = fft(A,[],3);
%        
%         if(count == 1)
%         else
%             A=threshold(A,U,V,noiselevel,tau);
%         end
%         
%         A = ifft(A,[],3);
%         
%         
%         if(modified == 1)
%             A = my_ttm(A,U4,ndims(A),'nt');
%         end
%         
%         
%         A_denoised = A(:,:,1:3,:);
%         A = A_denoised;
%         
% 
%         for k=1:count
%             yindex = similarity_indices(k,1);
%             xindex = similarity_indices(k,2);
%             im2(yindex:yindex+ps-1,xindex:xindex+ps-1,:) = im2(yindex:yindex+ps-1,xindex:xindex+ps-1,:)+A(:,:,:,k);
%             numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,:) = numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,:)+1;
%         end
% 
%     end
% end
% 
% 
% ind_zero = numcount==0;numcount(ind_zero) = 1;
% im2(ind_zero) = im1(ind_zero);
% im2 = im2./numcount;
% 
% im2=double(im2);im=double(im);
% 
% mse = sum((im2(:)-im(:)).^2)/(H*W*D);
% 
% psnr = 10*log10(255*255/mse);
% 
% im_ssim = cal_ssim(im2,im,0,0);
% 
% 
% end
% 
% 
% function patch_mode = decide_mode(refpatch)
% 
% patchR = refpatch(:,:,1);
% patchG = refpatch(:,:,2);
% patchB = refpatch(:,:,3);
% 
% R_sum = sum(patchR(:));
% G_sum = sum(patchG(:));
% B_sum = sum(patchB(:));
% 
% if G_sum > B_sum * 2 && G_sum > R_sum * 2
%     patch_mode = 'green_extra';
% elseif G_sum > B_sum/2 && G_sum > R_sum/2
%     patch_mode = 'green';
% else
%     patch_mode = 'normal';
% end
% 
% if R_sum > B_sum * 2 && R_sum > G_sum * 2
%     patch_mode = 'red_extra';
% end
% 
% if B_sum > R_sum * 2 && B_sum > G_sum * 2
%     patch_mode = 'blue_extra';
% end
% 
% end

%% patch search using most similar patches across all channels 
% this is not reasonable and produdes worse results because
% 别的channel的patch如果用来表示当前channel的patch会导致色彩的重复
% 比如用r和bchannel表示green channel的话肯定会有color artifact

% im1_reorder = im1;
% R_channel = im1_reorder(:,:,1);
% G_channel = im1_reorder(:,:,2);
% B_channel = im1_reorder(:,:,3);
% 
% im1_reorder(:,:,1) = G_channel;
% im1_reorder(:,:,2) = R_channel;
% im1_reorder(:,:,3) = B_channel;
% 
% for channel_selected = 1:D
%     im1_F = im1(:,:,channel_selected);
%     cur_channel = channel_selected;
%     
%     for i=1:N_step:H-ps+1
%         for j=1:N_step:W-ps+1
%             
%             refpatch = im1_F(i:i+ps-1,j:j+ps-1,:);
%             sr_top = max([i-SR 1]);
%             sr_left = max([j-SR 1]);
%             sr_right = min([j+SR W-ps+1]);
%             sr_bottom = min([i+SR H-ps+1]);
%             
%             count = 0;
%             similarity_indices = zeros((2*SR+1)^2,3);
%             
%             distvals = similarity_indices(:,1); %distance value of refpatch and each target patch.
%             for i1=sr_top:sr_bottom
%                 for j1=sr_left:sr_right
%                     for c = 1:D
%                         currpatch = im1_reorder(i1:i1+ps-1,j1:j1+ps-1,c); %current patch
%                         dist = sum((refpatch(:)-currpatch(:)).^2);
%                         count = count+1;
%                         distvals(count) = dist;
%                         similarity_indices(count,:) = [i1 j1 c];
%                     end
%                 end
%             end
%             
%             similarity_indices(1,:)=[i j 1];
%             similarity_indices = similarity_indices(1:count,:);
%             distvals = distvals(1:count);
%             
%             if count > maxK
%                 [~,sortedindices] = sort(distvals,'ascend');
%                 similarity_indices = similarity_indices(sortedindices(1:maxK),:);
%                 count = maxK;
%             end
%             
%             channel_indice = similarity_indices(:,3);
%             channel_count = zeros(D,1);
%             for c = 1:D
%                 channel_count(c) = length(find(channel_indice == c));
%             end
%             [max_channel_count, max_channel] = max(channel_count);
%             
%             A = zeros(ps,ps,D,max_channel_count,'single'); % construct a 4-D tensor with count patches
%             for c = 1:D
%                 channel_count_c = channel_count(c);
%                 if(channel_count_c == 0)
%                     continue;
%                 end
%                 
%                 row_idx = find(channel_indice == c);
%                 for k=1:channel_count_c
%                     row_idx_k = row_idx(k);
%                     yindex = similarity_indices(row_idx_k,1);
%                     xindex = similarity_indices(row_idx_k,2);
%                     A(:,:,c,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,c);
%                 end
%             end
%             
%             A_F = squeeze(A(:,:,max_channel,:));
%             if(cur_channel ~= max_channel)
%                 A_F(:,:,2:max_channel_count) = A_F(:,:,1:max_channel_count-1);
%                 A_F(:,:,1) = refpatch;
%             end
%             
%             if(modified == 1)
%                 mat_A = my_tenmat(A,ndims(A));
%                 mat_A_F = my_tenmat(A_F,ndims(A_F));
%                 [U4] = train_U4(mat_A_F);
%                 size_A = size(A);
%                 mat_A = U4'*mat_A;A = mat_ten(mat_A,ndims(A),size_A);
%             end
%             
%             A = fft(A,[],3);
%             
%             if(count == 1)
%             else
%                 A=threshold(A,U,V,sigma,tau);
%             end
%             
%             A = ifft(A,[],3);
%             
%             if(modified == 1)
%                 A = my_ttm(A,U4,ndims(A),'nt');
%             end
%             
%             
%             for c = 1:D
%                 channel_count_c = channel_count(c);
%                 if(channel_count_c == 0)
%                     continue;
%                 end
%                 
%                 row_idx = find(channel_indice == c);
%                 for k=1:channel_count_c
%                     row_idx_k = row_idx(k);
%                     yindex = similarity_indices(row_idx_k,1);
%                     xindex = similarity_indices(row_idx_k,2);
%                     im2(yindex:yindex+ps-1,xindex:xindex+ps-1,c) = im2(yindex:yindex+ps-1,xindex:xindex+ps-1,c) + A(:,:,c,k);
%                     numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,c) = numcount(yindex:yindex+ps-1,xindex:xindex+ps-1,c)+1;
%                 end
%             end
%             
%         end
%     end
%     
% end
% 
% ind_zero = numcount==0;numcount(ind_zero) = 1;
% im2(ind_zero) = im1(ind_zero);
% im2 = im2./numcount;
% 
% im2=double(im2);im=double(im);
% 
% mse = sum((im2(:)-im(:)).^2)/(H*W*D);
% 
% psnr = 10*log10(255*255/mse);
% 
% im_ssim = cal_ssim(im2,im,0,0);

% end


function [U,V] = compute_global(im1,N_step,ps)
[H,W,D] = size(im1);
count = 0;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        count = count + 1;
        global_indice(count,:) = [i,j];
    end
end
A = zeros(ps,ps,D,count,'single');
for k=1:count
    yindex = global_indice(k,1);
    xindex = global_indice(k,2);
    A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
end

if D == 4
    A_learn = zeros(ps,ps,4,count,'single');
    A_learn(:,:,1,:) = A(:,:,1,:);
    A_learn(:,:,2,:) = A(:,:,2,:);
    A_learn(:,:,3,:) = A(:,:,2,:);
    A_learn(:,:,4,:) = A(:,:,3,:);
else
    A_learn = A;
end

[U,V]=NL_tSVD(A_learn);
end

function [U,V] = compute_global_3D(im1,N_step,ps)
[H,W,D] = size(im1);
count = 0;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        count = count + 1;
        global_indice(count,:) = [i,j];
    end
end
A = zeros(ps,ps,D,count,'single');
for k=1:count
    yindex = global_indice(k,1);
    xindex = global_indice(k,2);
    A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
end

[U,V]=NL_tSVD(A);

end

function [U,V] = compute_global_4D(im1,N_step,ps)
[H,W,D] = size(im1);
count = 0;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        count = count + 1;
        global_indice(count,:) = [i,j];
    end
end
A = zeros(ps,ps,D,count,'single');
for k=1:count
    yindex = global_indice(k,1);
    xindex = global_indice(k,2);
    A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
end

A_learn = zeros(ps,ps,4,count,'single');
A_learn(:,:,1,:) = A(:,:,2,:);
A_learn(:,:,2,:) = A(:,:,2,:);
A_learn(:,:,3,:) = A(:,:,1,:);
A_learn(:,:,4,:) = A(:,:,3,:);

[U,V]=NL_tSVD(A_learn);
end


function [U,V] = compute_global_4D_by_channel(im1,N_step,ps, patch_mode)
[H,W,D] = size(im1);
count = 0;
for i=1:N_step:H-ps+1 %why start from i=104?
    for j=1:N_step:W-ps+1 %why start from j=49?
        count = count + 1;
        global_indice(count,:) = [i,j];
    end
end
A = zeros(ps,ps,D,count,'single');
for k=1:count
    yindex = global_indice(k,1);
    xindex = global_indice(k,2);
    A(:,:,:,k) = im1(yindex:yindex+ps-1,xindex:xindex+ps-1,:);
end

A_learn = zeros(ps,ps,4,count,'single');


if strcmp(patch_mode, 'red') == 1
    
    A_learn(:,:,1,:) = A(:,:,1,:);
    A_learn(:,:,2,:) = A(:,:,2,:);
    A_learn(:,:,3,:) = A(:,:,3,:);
    A_learn(:,:,4,:) = A(:,:,1,:);
    
end

if strcmp(patch_mode, 'green') == 1
    
    A_learn(:,:,1,:) = A(:,:,1,:);
    A_learn(:,:,2,:) = A(:,:,2,:);
    A_learn(:,:,3,:) = A(:,:,3,:);
    A_learn(:,:,4,:) = A(:,:,2,:);
    
end

if strcmp(patch_mode, 'blue') == 1
    
    A_learn(:,:,1,:) = A(:,:,1,:);
    A_learn(:,:,2,:) = A(:,:,2,:);
    A_learn(:,:,3,:) = A(:,:,3,:);
    A_learn(:,:,4,:) = A(:,:,3,:);
    
end



[U,V]=NL_tSVD(A_learn);
end


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


function A_thre = threshold(A,U,V,sigma,tau)
size_A = size(A);ps = size_A(1);D = size_A(3);count = size_A(end);
A_thre = zeros(size_A);real_count = floor(D/2)+1; 
U = fft(U,[],3); V = fft(V,[],3);
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


function A  = com_conj(A)
[~,~,D,~] = size(A);
k = 0;
for i = 2:floor(D/2)+1
    A(:,:,D-k,:) = conj(A(:,:,i,:));
    k = k + 1;
end
end


function [U4] = train_U4(A)
[num_patch,~] = size(A);
sum_A = sum(A,1);
average_A = sum_A/num_patch;
A_hat = A - average_A;
[U4,~] = eig(A_hat*A_hat');
end
