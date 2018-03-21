addpath('/home/ohadsh/work/python/thesis/matlab/');

sampling_factor = 4;
start_line = 1;
keep_center = 0.05;
W = 256;
H = 256;
disp(['Working on radom mask - ', num2str(sampling_factor)])

size_dat = W * H;
basic_path = '/media/ohadsh/Data/ohadsh/work/data/T1/sagittal/';
tt = 'train';
N_imgs = 5;
[real_all, imag_all, mask_all] = get_data_memmap(basic_path, tt, W, H);

i = 1;
disp(['Running on i=', num2str(i)]);

real = real_all.Data(i).data;
imag = imag_all.Data(i).data;

data = real + 1i*imag;

org_image = abs(norm_ifft(data));

imagesc(log(sqrt(real^2 + imag^2)));
colormap gray;