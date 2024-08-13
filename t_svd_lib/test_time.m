function [  ] = test_time( n )
%TEST_TIME Summary of this function goes here
%   Detailed explanation goes here
A = rand(8,8,3,n);
tic
A_a = fft(A,[],3);
time1 = toc

tic
A_f = my_fft(A,3);
time2 = toc

norm(A_a(:) - A_f(:))

end

