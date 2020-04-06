%% Gaussian PSF and Zero Mean Gaussian Noise
Image = imread('test_pattern-1.tif');
[nrows,ncols] = size(Image);
figure
subplot(2,1,1)
imagesc(Image)
axis off
colormap(gray(256)) 
title 'test pattern'
rcen = nrows/2 +1;
ccen = ncols/2 +1;
width = 80;
zrows = nrows+2*width;
zcols = ncols+2*width;
zrcen = (nrows+2*width)/2 + 1;
zccen = (ncols+2*width)/2 + 1;
zeropad = zeros(zrows,zcols);
rstart = zrcen - nrows/2;
rstop = zrcen + nrows/2 - 1;
cstart = zccen - ncols/2;
cstop = zccen + ncols/2 - 1;
zeropad(rstart:rstop,cstart:cstop) = Image;
subplot(2,1,2)
imagesc(zeropad)
axis off
colormap(gray(256)) 
title 'zero padded test pattern'

gauss_psf = fspecial('gaussian',[zrows,zcols],10);
tf = fftshift(fft2(fftshift(gauss_psf)));
tf = tf/tf(zrcen,zccen);
ft_tgt = fftshift(fft2(fftshift(zeropad)));
ft_tgt = ft_tgt/ft_tgt(zrcen,zccen);
prod = tf .* ft_tgt;
noise_free_img = real(fftshift(ifft2(fftshift(prod))));
noise_free_img = noise_free_img/max(max(noise_free_img));
sigma = 0.08; 
gauss_noise = sigma*randn([zrows,zcols]);
gauss_noise_pattern = noise_free_img + gauss_noise;
% m = 0; 
% std = 2; 
% pd = makedist('Normal','mu',m,'sigma',std); 
% gauss_noise = std*random(pd, [zrows,zcols]);
% gauss_noise_pattern = gauss_psf + round(gauss_noise); 
figure

imagesc(gauss_psf)
axis off
colormap(gray(256)) 
title 'Gaussian Point Spread Function'
figure
imagesc(gauss_noise_pattern)
axis off
colormap(gray(256)) 
title 'Pattern with Zero Mean Gaussian Noise'



%% Inverse Filter
psf_tf = fftshift(fft2(fftshift(gauss_psf)));
psf_tf = psf_tf/psf_tf(zrcen,zccen);
noise_pattern_tf = (fftshift(fft2(fftshift(gauss_noise_pattern))));
noise_pattern_tf = noise_pattern_tf/noise_pattern_tf(zrcen,zccen);
inv_filt = 1 ./(psf_tf+0.08);
inv_filt_spec = inv_filt .* noise_pattern_tf;
inv_filt_img = real(fftshift(ifft2(fftshift(inv_filt_spec))));
max(max(inv_filt_img))
figure

imagesc(gauss_noise_pattern)
axis off
colormap(gray(256)) 
title 'Pattern with Zero Mean Gaussian Noise'

figure
imagesc(inv_filt_img)
axis('off')
colormap(gray(256))
title('Inverse filter on Noisy pattern')

%% Wiener Filter
noise_psd = log10(abs(fftshift(fft2(gauss_noise))).^2);
%noise_psd = dspdata.psd(gauss_noise);
pattern_psd = log10(abs(fftshift(fft2(zeropad))).^2);
%pattern_psd = dspdata.psd(zeropad);
%Img2 = (gauss_noise_pattern) .* (gauss_noise_pattern); 
Wiener = (conj(psf_tf))./((abs((conj(psf_tf).*psf_tf))) + (noise_psd./pattern_psd));
Wiener_img = (noise_pattern_tf).*(Wiener) ;
wiener_ft = real(fftshift(ifft2(fftshift(Wiener_img))));
figure
imagesc(abs(wiener_ft))
axis('off')
colormap(gray(256))
title('Wiener filter on Noisy pattern')

%% Pseudo Wiener Filter
Pseudo_coeff = 0.09;
Pseudo_Wiener = (conj(psf_tf))./((abs((conj(psf_tf).*psf_tf))) + (Pseudo_coeff));
Pseudo_img = (noise_pattern_tf).*(Pseudo_Wiener) ;
Pseudo_ft = real(fftshift(ifft2(fftshift(Pseudo_img))));
figure
imagesc(abs(Pseudo_ft))
axis('off')
colormap(gray(256))
title('Pseudo Wiener filter on Noisy pattern')


