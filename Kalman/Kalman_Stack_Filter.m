function imageStack=Kalman_Stack_Filter(imageStack,gain,percentvar)
if nargin<2, gain=0.5;          end
if nargin<3, percentvar = 0.05; end
if gain>1.0||gain<0.0
    gain = 0.8;
end

if percentvar>1.0 || percentvar<0.0
    percentvar = 0.05;
end
imageStack(:,:,end+1)=imageStack(:,:,end);

width = size(imageStack,1);
height = size(imageStack,2);
stacksize = size(imageStack,3);

tmp=ones(width,height);

predicted = imageStack(:,:,1); 
predictedvar = tmp*percentvar;
noisevar=predictedvar;

x=0;
for i=2:stacksize-1
  stackslice = imageStack(:,:,i+1); 
  observed = stackslice;
  Kalman = predictedvar ./ (predictedvar+noisevar);
  % set motion compensation
%   figure(499)
%   imshow(predicted)
%   if i>3
%     predicted=MotionCompensate(predicted,observed);
%   end
  corrected = gain*predicted + (1.0-gain)*observed + Kalman.*(observed-predicted);        
  correctedvar = predictedvar.*(tmp - Kalman);

%   figure(500)
%   imshow(stackslice)
%   figure(501)
%   imshow(predicted)
%   figure(502)
%   imshow(corrected)
%   figure(503)
%   imshow(imageStack(:,:,i+1))
%   figure(504)
%   imshow(imageStack(:,:,i))
  predictedvar = correctedvar;
  predicted = corrected;
  imageStack(:,:,i)=corrected;

end

imageStack(:,:,end)=[];
