function [ A_f ] = my_fft( A,n )
%MY_FFT Summary of this function goes here
%   Detailed explanation goes here
A3 = my_tenmat(A,n);A_f = A3;W = exp(2*pi*1i/-3);conjW = conj(W);
m = [1 W conjW];
A_f1 = A_f(1,:) + A_f(2,:) + A_f(3,:);
A_f2 = m*A3;
A_f3 = conj(A_f2);

A_f(1,:) = A_f1;A_f(2,:) = A_f2;A_f(3,:) = A_f3;
A_f = mat_ten(A_f,n,size(A));



end

