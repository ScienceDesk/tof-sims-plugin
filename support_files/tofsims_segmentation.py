import numpy as np
from skimage.segmentation import chan_vese
from scipy.signal import convolve2d

def clip_values(np_array, inf=10, sup=10):
    std = np_array.std()
    avg = np_array.mean()
    return np.clip(np_array,avg-inf*std, avg+sup*std)

def normalize_intesity(np_array):
    im_np_norm = np.copy(np_array)
    im_np_norm = np_array - np_array.min()
    im_np_norm *= 1/im_np_norm.max() 
    return im_np_norm


def gaussian_blur(im):
    kernel = np.array(
        [
            [1 / 16, 1 / 8, 1 / 16],  # 3x3 kernel
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16],
        ]
    )

    data_np = convolve2d(convolve2d(im, kernel, "same"), kernel, "same")
    return data_np
    
def get_segmented_img(self):
    img = self.pixels
    h, v = img.shape
    img = gaussian_blur(img)
    img = normalize_intesity(img)
    img = clip_values(img,2.5,1.8)
    
    cv = chan_vese(img, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=100,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

    return normalize_intesity(cv[1][round(v/10):-round(v/10),round(h/10):-round(h/10)])