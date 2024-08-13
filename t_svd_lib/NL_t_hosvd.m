function [ S,U,V,W ] = NL_t_hosvd(A)
%t_hosvd compute the hosvd-like t_SVD
%   input A must be a cell 4-order tensor that contains N tensors, where N
%   represents the number of similar pictures.

S = A;A = tensor(A);

for k=1:2
    [U(:,:,k),V(:,:,k),W(:,:,k)]=tucker(A,k);
end


for k = 1:2
    if(k == 1)
%         Sk = ttm(A(:,:,k,:),{U(:,:,k),V(:,:,k)},-3,'t');
        Sk = ttm(A(:,:,k,:),{U(:,:,k),V(:,:,k),W(:,:,k)},'t');
    else
%         Sk = ttm(A(:,:,k,:),{U(:,:,k),V(:,:,k),W(:,:,k)},'t');
        Sk = ttm(A(:,:,k,:),{U(:,:,k),V(:,:,k)},-3,'t');
    end
%     Sk = ttm(A(:,:,k,:),{U(:,:,k),V(:,:,k),W(:,:,k)},'t');
    S(:,:,k,:) = Sk;
end
S(:,:,3,:) = conj(S(:,:,2,:));

% for i=1:N_pictures
%     A_f=A(:,:,:,i);
%     for k=1:2
%          S(:,:,k,i)=U(:,:,k)'*A_f(:,:,k)*V(:,:,k);
%     end
%     S(:,:,3,i)=conj(S(:,:,2,i));
% end



end


function[U,V,W]=tucker(A,k)
A = tensor(A);
A_f = A(:,:,k,:);
A1 = tenmat(A_f,1);A1 = double(A1.data);
A2 = tenmat(A_f,2);A2 = double(A2.data);
A3 = tenmat(A_f,3);A3 = double(A3.data);

[U,~] = svd(A1*A1');[V,~] = svd(A2*A2');[W,~] = svd(A3*A3');

% for i=1:N_pictures
%     A_f=A(:,:,:,i);
% %     sum_A = sum_A + A_f(:,:,k);
%     sum_U=sum_U+A_f(:,:,k)*A_f(:,:,k)';
%     sum_V=sum_V+A_f(:,:,k)'*A_f(:,:,k);
% end

% [U,S,V] = svd(sum_A);
% ss=diag(S);[~,index]=sort(ss,'descend');U=U(:,index(1:R));V=V(:,index(1:R));

% [U,~]=svd(sum_U);
% % 
% [V,~]=svd(sum_V);

end