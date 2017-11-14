function [real, imag, mask_all] = get_data_memmap(basic_path, tt, W, H)
% Read my data
path_tt = fullfile(basic_path, tt, '000000.k_space_real_gt.bin');
real = memmapfile(path_tt, 'Format',{'single', [W,H],'data'});

path_tt = fullfile(basic_path, tt, '000000.k_space_imag_gt.bin');
imag = memmapfile(path_tt, 'Format',{'single', [W,H],'data'});

names = {'mask.bin','mask2_5.bin', 'mask4.bin', 'mask6.bin', 'mask8.bin', 'mask10.bin', 'mask20.bin'};
mask_all = cell(length(names),1);
index = 1;
for name = names
    path_tt = fullfile('/media/ohadsh/Data/ohadsh/work/matlab/thesis', name);
    f = fopen(path_tt{1}, 'rb');
    mask = fread(f, 1*W*H, 'uint8');
    mask = reshape(mask, W, H, [])';
    fclose(f);
    mask_all{index} = mask;
    index = index + 1;
end

end

