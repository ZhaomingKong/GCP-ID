function [ U,S,V ] = NL_t_svd2(A,R)
%t_hosvd compute the hosvd-like t_SVD
%   input A must be a cell 4-order tensor that contains N tensors, where N
%   represents the number of similar pictures.

size_A=size(A);length_A = length(size_A);B = zeros(size_A);
N_pictures=size_A(end);[~,~,D]=size(A(:,:,:,1));
% U=zeros(H,H,D);S=zeros(H,W,D,N_pictures);V=zeros(W,W,D);

for k=1:2
    [U(:,:,k),V(:,:,k)]=tucker(A,R,k);
end

for i=1:N_pictures
    A_f=A(:,:,:,i);
    for k=1:2
         S(:,:,k,i)=U(:,:,k)'*A_f(:,:,k)*V(:,:,k);
    end
    S(:,:,3,i)=conj(S(:,:,2,i));
end



end


function[U,V]=tucker(A,R,k)
size_A=size(A);N_pictures=size_A(end);
sum_U=0;sum_V=0;sum_A = 0;



for i=1:N_pictures
    A_f=A(:,:,:,i);
%     sum_A = sum_A + A_f(:,:,k);
    sum_U=sum_U+A_f(:,:,k)*A_f(:,:,k)';
    sum_V=sum_V+A_f(:,:,k)'*A_f(:,:,k);
end

% [U,S,V] = svd(sum_A);
% ss=diag(S);[~,index]=sort(ss,'descend');U=U(:,index(1:R));V=V(:,index(1:R));

[U,~]=svd(sum_U);
% 
[V,~]=svd(sum_V);

end