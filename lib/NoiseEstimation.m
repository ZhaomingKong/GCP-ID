% -----------------------------------------------------------------------     
%Inputs:
% im_noisy:  the noisy image whose noise level requires to be estimated
% PatchSize: the predefined size of patches
% 
%Outputs:
% estsigma: Estimated result given by our method
% -----------------------------------------------------------------------
function estsigma =NoiseEstimation(im_noisy,PatchSize)

p_out = image2cols(im_noisy, PatchSize, 3);

mu = mean(p_out,2);
sigma=(p_out-repmat(mu,[1,size(p_out,2)])) ...
        *(p_out-repmat(mu,[1,size(p_out,2)]))'/(size(p_out,2));
eigvalue = (sort((eig(sigma)),'ascend'));
 
for CompCnt = size(p_out,1):-1:1
    Mean = mean(eigvalue(1:CompCnt));
    
    if(sum(eigvalue(1:CompCnt)>Mean) == sum(eigvalue(1:CompCnt)<Mean))
        break
    end
   
end
estsigma = sqrt(Mean);

function res = image2cols(im, pSz, stride)
  res = [];

  range_y = 1:stride:(size(im,1)-pSz+1);
  range_x = 1:stride:(size(im,2)-pSz+1);
  channel = size(im,3);
  if (range_y(end)~=(size(im,1)-pSz+1))
    range_y = [range_y (size(im,1)-pSz+1)];
  end
  if (range_x(end)~=(size(im,2)-pSz+1))
    range_x = [range_x (size(im,2)-pSz+1)];
  end
  sz = length(range_y)*length(range_x);

  tmp = zeros(pSz^2*channel, sz);

  idx = 0;
  for y=range_y
    for x=range_x
      p = im(y:y+pSz-1,x:x+pSz-1,:);
      idx = idx + 1;
      tmp(:,idx) = p(:);
    end
  end

  res = [res, tmp];
return
