import os
import skimage
from skimage import io
from skimage import exposure
from skimage.filters import threshold_otsu
read_folder = 'C:/Users/User/Desktop/FAKE_SIGN_ALL'
write_folder = 'C:/Users/User/Desktop/FAKE_SIGN_ALL_PROCESSED'
os.makedirs(write_folder)
for i in os.listdir(read_folder):
    clr_image = io.imread(os.path.join(read_folder, i))
    gray_image = skimage.color.rgb2gray(clr_image)
    #equalized_image = exposure.equalize_hist(gray_image)
    equalized_image=skimage.exposure.equalize_adapthist(gray_image)
    thresh_val = threshold_otsu(equalized_image)
    binary_image = equalized_image > thresh_val
    io.imsave(os.path.join(write_folder, i), binary_image.astype(int))
    