%% Test Images with Noise addition
halfwid = 240;
Image1 = imread('liam gallagher.jpg');
grayImage1 = double(rgb2gray(Image1));
[nrows1,ncols1] = size(grayImage1); 
rcen1 = nrows1/2 + 1;
ccen1 = round(ncols1/2);
subgrayImage1 = grayImage1((rcen1-halfwid):(rcen1+halfwid),(ccen1-halfwid):(ccen1+halfwid));
[nrows1,ncols1] = size(subgrayImage1); 
Image2 = imread('noel gallagher.jpg');
grayImage2 = double(rgb2gray(Image2));
[nrows2,ncols2] = size(grayImage2); 
rcen2 = nrows2/2 + 1;
ccen2 = ncols2/2 + 1;
subgrayImage2 = grayImage2((rcen2-halfwid):(rcen2+halfwid),(ccen2-halfwid):(ccen2+halfwid));
[nrows2,ncols2] = size(subgrayImage2); 
Image3 = imread('Oasis.jpg');
grayImage3 = double(rgb2gray(Image3));
[nrows3,ncols3] = size(grayImage3); 
rcen3 = round(nrows3/2);
ccen3 = round(ncols3/2);
subgrayImage3 = grayImage3((rcen3-halfwid):(rcen3+halfwid),(ccen3-halfwid):(ccen3+halfwid));
[nrows3,ncols3] = size(subgrayImage3); 
figure
subplot(2,3,1)
imagesc(Image1)
axis 'equal'
colormap(gray(256)) 
title 'Image 1'
subplot(2,3,2)
imagesc(Image2)
axis 'equal'
colormap(gray(256)) 
title 'Image 2'
subplot(2,3,3)
imagesc(Image3)
axis 'equal'
colormap(gray(256)) 
title 'Image 3'
subplot(2,3,4)
imagesc(subgrayImage1)
axis 'equal'
colormap(gray(256)) 
title 'GrayImage 1'
subplot(2,3,5)
imagesc(subgrayImage2)
axis 'equal'
colormap(gray(256)) 
title 'GrayImage 2'
subplot(2,3,6)
imagesc(subgrayImage3)
axis 'equal'
colormap(gray(256)) 
title 'GrayImage 3'

% Gaussian Noise
m = 15; 
std = 3.3; 
pd = makedist('Normal','mu',m,'sigma',std); 
gauss_noise1 = std*random(pd, [nrows1 ncols1]); 
gauss_noise_img1 = subgrayImage1 + round(gauss_noise1); 
figure 
subplot(1,3,1) 
imagesc(gauss_noise_img1) 
colormap(gray(256)) 
title 'Gauss Noise on Image 1'
gauss_noise2 = std*random(pd, [nrows2 ncols2]); 
gauss_noise_img2 = subgrayImage2 + round(gauss_noise2); 
subplot(1,3,2) 
imagesc(gauss_noise_img2)  
colormap(gray(256)) 
title 'Gauss Noise on Image 2'
gauss_noise3 = std*random(pd, [nrows3 ncols3]); 
gauss_noise_img3 = subgrayImage3 + round(gauss_noise3); 
subplot(1,3,3) 
imagesc(gauss_noise_img3) 
colormap(gray(256)) 
title 'Gauss Noise on Image 3'

% Salt Pepper Noise
den = 0.3; 
normtgt1 = subgrayImage1/max(max(subgrayImage1));
salt_noise_img1 = imnoise(normtgt1,'salt & pepper',den);
snoise1 = salt_noise_img1 - subgrayImage1;
snoise1 = (-1).*snoise1;
figure 
subplot(1,3,1) 
imagesc(salt_noise_img1) 
colormap(gray(256)) 
title 'Salt Pepper Noise on Image1'
normtgt2 = subgrayImage2/max(max(subgrayImage2)); 
tgt2 = ones(nrows2,ncols2);
salt2 = tgt2/max(max(tgt2));
saltpepper_noise2 = max(max(tgt2))*imnoise(salt2,'salt & pepper',den);
salt_noise_img2 = imnoise(normtgt2,'salt & pepper',den);
snoise2 = salt_noise_img2 - subgrayImage2;
subplot(1,3,2) 
imagesc(salt_noise_img2) 
colormap(gray(256)) 
title 'Gauss and Salt Pepper Noise on Image2'
normtgt3 = subgrayImage3/max(max(subgrayImage3));
salt_noise_img3 = imnoise(normtgt3,'salt & pepper',den); 
snoise3 = salt_noise_img3 - subgrayImage3;
subplot(1,3,3) 
imagesc(salt_noise_img3) 
colormap(gray(256)) 
title 'Gauss and Salt Pepper Noise on Image3'

% Exponential Noise
m = 10; 
pd = makedist('Exponential','mu',m); 
exp_noise1 = random(pd,[nrows1,ncols1]); 
exp_noise_img1 = subgrayImage1 + round(exp_noise1);
figure 
subplot(1,3,1) 
imagesc(exp_noise_img1) 
colormap(gray(256)) 
title 'Exponential Noise on Image1'
exp_noise2 = random(pd,[nrows2,ncols2]); 
exp_noise_img2 = subgrayImage2 + round(exp_noise2); 
subplot(1,3,2) 
imagesc(exp_noise_img2)
colormap(gray(256)) 
title 'Exponential Noise on Image3'
exp_noise3 = random(pd,[nrows3,ncols3]); 
exp_noise_img3 = subgrayImage3 + round(exp_noise3);
subplot(1,3,3) 
imagesc(exp_noise_img3) 
colormap(gray(256)) 
title 'Exponential Noise on Image3'

%% Harmonic Filters
rwid = 2;
cwid = 2;
kernel1 = (((rwid+1)*(cwid+1))^(-1))*ones((rwid+1),(cwid+1));
HarmonicS1 = zeros(nrows1,ncols1);
for r=(0.5*rwid+1):(nrows1-rwid/2)
for c=(0.5*cwid+1):(ncols1-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./salt_noise_img1(rstart:rstop,cstart:cstop);
 HarmonicS1(r,c) = ((rwid+1)*(cwid+1))/(sum(sum(subimg1.*kernel1)));
end
end
HarmonicS2 = zeros(nrows2,ncols2);
for r=(0.5*rwid+1):(nrows2-rwid/2)
for c=(0.5*cwid+1):(ncols2-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./salt_noise_img2(rstart:rstop,cstart:cstop);
 HarmonicS2(r,c) = ((rwid+1)*(cwid+1))/(sum(sum(subimg1.*kernel1)));
end
end
HarmonicS3 = zeros(nrows3,ncols3);
for r=(0.5*rwid+1):(nrows3-rwid/2)
for c=(0.5*cwid+1):(ncols3-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./salt_noise_img3(rstart:rstop,cstart:cstop);
 HarmonicS3(r,c) = ((rwid+1)*(cwid+1))/(sum(sum(subimg1.*kernel1)));
end
end
figure
subplot(1,3,1) 
imagesc(HarmonicS1)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Salt Pepper Noise Image1'
subplot(1,3,2) 
imagesc(HarmonicS2)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Salt Pepper Noise Image2'
subplot(1,3,3) 
imagesc(HarmonicS3)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Salt Pepper Noise Image3'

HarmonicE1 = zeros(nrows1,ncols1);
for r=(0.5*rwid+1):(nrows1-rwid/2)
for c=(0.5*cwid+1):(ncols1-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./exp_noise_img1(rstart:rstop,cstart:cstop);
 HarmonicE1(r,c) = ((rwid+1)*(cwid+1))/(sum(sum(subimg1.*kernel1)));
end
end
HarmonicE2 = zeros(nrows2,ncols2);
for r=(0.5*rwid+1):(nrows2-rwid/2)
for c=(0.5*cwid+1):(ncols2-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./exp_noise_img2(rstart:rstop,cstart:cstop);
 HarmonicE2(r,c) = ((rwid+1)*(cwid+1))/(sum(sum(subimg1.*kernel1)));
end
end
HarmonicE3 = zeros(nrows3,ncols3);
for r=(0.5*rwid+1):(nrows3-rwid/2)
for c=(0.5*cwid+1):(ncols3-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./exp_noise_img3(rstart:rstop,cstart:cstop);
 HarmonicE3(r,c) = ((rwid+1)*(cwid+1))/(sum(sum(subimg1.*kernel1)));
end
end
figure
subplot(1,3,1) 
imagesc(HarmonicE1)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Exponential Noise Image1'
subplot(1,3,2) 
imagesc(HarmonicE2)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Exponential Noise Image2'
subplot(1,3,3) 
imagesc(HarmonicE3)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Exponential Noise Image3'

rwid = 2;
cwid = 2;
kernel1 = (((rwid+1)*(cwid+1))^(-1))*ones((rwid+1),(cwid+1));
HarmonicG1 = zeros(nrows1,ncols1);
for r=(0.5*rwid+1):(nrows1-rwid/2)
for c=(0.5*cwid+1):(ncols1-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./gauss_noise_img1(rstart:rstop,cstart:cstop);
 HarmonicG1(r,c) = ((rwid+1)*(cwid+1))/(mean2(subimg1));
end
end
figure
subplot(1,3,1) 
imagesc(HarmonicG1)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Gaussian Noise Image1'
rwid = 2;
cwid = 2;
kernel1 = (((rwid+1)*(cwid+1))^(-1))*ones((rwid+1),(cwid+1));
HarmonicG2 = zeros(nrows2,ncols2);
for r=(0.5*rwid+1):(nrows2-rwid/2)
for c=(0.5*cwid+1):(ncols2-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./gauss_noise_img2(rstart:rstop,cstart:cstop);
 HarmonicG2(r,c) = ((rwid+1)*(cwid+1))/(mean2(subimg1));
end
end
subplot(1,3,2) 
imagesc(HarmonicG2)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Gaussian Noise Image2'
rwid = 2;
cwid = 2;
kernel1 = (((rwid+1)*(cwid+1))^(-1))*ones((rwid+1),(cwid+1));
HarmonicG3 = zeros(nrows3,ncols3);
for r=(0.5*rwid+1):(nrows3-rwid/2)
for c=(0.5*cwid+1):(ncols3-cwid/2)
 rstart = r-rwid/2;
 rstop = r+rwid/2;
 cstart = c- cwid/2;
 cstop = c + cwid/2;
 subimg1 = 1./gauss_noise_img3(rstart:rstop,cstart:cstop);
 HarmonicG3(r,c) = ((rwid+1)*(cwid+1))/(mean2(subimg1));
end
end
subplot(1,3,3) 
imagesc(HarmonicG3)
axis off
colormap(gray(256)) 
title 'Harmonic Filter on Gaussian Noise Image3'

%% Adaptive Local Noise Filter
A = reshape(gauss_noise1,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows1 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols1 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = gauss_noise_img1((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFg1(r,c) = gauss_noise_img1(r,c) - ((GVnoise1/lGvar1).*(gauss_noise_img1(r,c) - lgmean1));
    end
end
figure
subplot(1,3,1)
imagesc(ALNFg1)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Gaussian Noise Image1'

A = reshape(gauss_noise2,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows2 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols2 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = gauss_noise_img2((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFg2(r,c) = gauss_noise_img2(r,c) - ((GVnoise1/lGvar1).*(gauss_noise_img2(r,c) - lgmean1));
    end
end
subplot(1,3,2)
imagesc(ALNFg2)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Gaussian Noise Image2'

A = reshape(gauss_noise3,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows3 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols3 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = gauss_noise_img3((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFg3(r,c) = gauss_noise_img3(r,c) - ((GVnoise1/lGvar1).*(gauss_noise_img3(r,c) - lgmean1));
    end
end
subplot(1,3,3)
imagesc(ALNFg3)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Gaussian Noise Image3'

A = reshape(snoise1,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows1 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols1 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = salt_noise_img1((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFs1(r,c) = salt_noise_img1(r,c) - ((GVnoise1/lGvar1).*(salt_noise_img1(r,c) - lgmean1));
    end
end
subplot(1,3,1)
imagesc(ALNFs1)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Salt Pepper Noise Image1'

A = reshape(snoise2,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows2 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols2 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = salt_noise_img2((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFs2(r,c) = salt_noise_img2(r,c) - ((GVnoise1/lGvar1).*(salt_noise_img2(r,c) - lgmean1));
    end
end
subplot(1,3,2)
imagesc(ALNFs2)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Salt Pepper Noise Image2'

A = reshape(snoise3,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows3 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols3 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = salt_noise_img3((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFs3(r,c) = salt_noise_img3(r,c) - ((GVnoise1/lGvar1).*(salt_noise_img3(r,c) - lgmean1));
    end
end
subplot(1,3,3)
imagesc(ALNFs3)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Salt Pepper Noise Image3'

A = reshape(exp_noise1,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows1 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols1 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = exp_noise_img1((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFe1(r,c) = exp_noise_img1(r,c) - ((GVnoise1/lGvar1).*(exp_noise_img1(r,c) - lgmean1));
    end
end
figure
subplot(1,3,1)
imagesc(ALNFe1)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Exponential Noise Image1'

A = reshape(exp_noise2,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows2 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols2 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = exp_noise_img2((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFe2(r,c) = exp_noise_img2(r,c) - ((GVnoise1/lGvar1).*(exp_noise_img2(r,c) - lgmean1));
    end
end
subplot(1,3,2)
imagesc(ALNFe2)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Exponential Noise Image2'

A = reshape(exp_noise3,1,[]);
mgnoise1 = mean(A);
gnoise11 = A.^2;
m2gnoise1 = mean(gnoise11);
GVnoise1 = m2gnoise1 - (mgnoise1.^2);
width = 3;
hw = floor(3/2);

rstart1 = round(width/2);
rstop1 = nrows3 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols3 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
        subimgLG1 = exp_noise_img3((r-hw):(r+hw),(c-hw):(c+hw));
        lgmean1 = mean2(subimgLG1);
        subimgLG11 = subimgLG1.^2;
        lgmean2 = mean2(subimgLG11);
        lGvar1 = lgmean2 - lgmean1^2;
        ALNFe3(r,c) = exp_noise_img3(r,c) - ((GVnoise1/lGvar1).*(exp_noise_img3(r,c) - lgmean1));
    end
end
subplot(1,3,3)
imagesc(ALNFe3)
axis off
colormap(gray(256)) 
title 'Adaptive local Filter on Exponential Noise Image3'


%% Adaptive Median Filter
rcen1 = nrows1/2 + 1;
ccen1 = round(ncols1/2);
width = 3;
hw = floor(3/2);
rstart1 = round(width/2);
rstop1 = nrows1 - round(width/2);
cstart1 = round(width/2);
cstop1 = ncols1 - round(width/2);
for r = rstart1:rstop1
    for c = cstart1:cstop1
         Zmax(r,c) = max(max(gauss_noise_img1((rstart1):(rstop1),(cstart1):(cstop1))));
         Zmin(r,c) = min(min(gauss_noise_img1((rstart1):(rstop1),(cstart1):(cstop1))));
      Zmedian(r,c) = median(median(gauss_noise_img1((rstart1):(rstop1),(cstart1):(cstop1))));
           z1 = gauss_noise_img1((rstart1):(rstop1),(cstart1):(cstop1));
           
     if(Zmedian(r,c)>Zmin(r,c) && Zmax(r,c)>Zmedian(r,c))
         if(z1(r,c)>Zmin(r,c) && Zmax(r,c)>z1(r,c))
             AMFg1(r,c) = z1(r,c);
         else
             AMFg1(r,c) = Zmedian(r,c);
         end;
     end;
     end;
end;
  
             
         
