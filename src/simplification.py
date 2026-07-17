from abc import ABC, abstractmethod

import numpy as np
import torch
import pywt
from skimage.color import rgb2hsv, hsv2rgb

from PIL import Image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import torchvision.transforms as transforms
from torchvision.transforms import GaussianBlur


class SimplificationStrategy(ABC):
    name = "base"

    @abstractmethod
    def simplify(self, image):
        pass


class WaveletSimplification(SimplificationStrategy):
    def __init__(self, levels=2, lamb_soft_thresholding_compression=0.2, lamb_mul=1.2, wavelet='bior3.7'):
        self.name = "wavelet"
        self.levels = levels
        self.wavelet = wavelet
        self.lamb = lamb_soft_thresholding_compression
        self.lamb_mul = lamb_mul

    def simplify(self, image):
        print(image.shape)
        hsv_image = rgb2hsv(image.permute(1, 2, 0).cpu().numpy())
        value = hsv_image[:, :, 2]

        coeffs = pywt.wavedec2(value, self.wavelet, level=self.levels)
        coeffs_arr, coeff_slices = pywt.coeffs_to_array(coeffs)

        # soft thresholding
        print(
            f"Soft thresholding with lambda={self.lamb}. For new lambda reinitalize the WaveletSimplification class")
        coeffs_arr_thr = np.sign(coeffs_arr) * \
            np.maximum(np.abs(coeffs_arr) - self.lamb, 0.0)
        self.lamb += self.lamb_mul

        # Reconstruct image from thresholded coefficients
        coeffs_thresholded = pywt.array_to_coeffs(
            coeffs_arr_thr, coeff_slices, output_format='wavedec2')
        comp_value = pywt.waverec2(coeffs_thresholded, self.wavelet)
        hsv_image[:, :, 2] = comp_value

        # back to RGB
        simplified_image = hsv2rgb(hsv_image)
        simplified_image = torch.from_numpy(simplified_image).permute(
            2, 0, 1).to(image.device).float()
        return simplified_image


class ColorQuantizationSimplification(SimplificationStrategy):
    def __init__(self, colors=128, lamb_mul=0.75):
        self.name = "color_quantization"
        self.colors = colors
        self.lamb_mul = lamb_mul

    def simplify(self, image):
        
        im_q = to_pil_image(image)
        simplified_image = im_q.quantize(colors=self.colors)
        self.colors = np.clip(int(self.colors * self.lamb_mul), 1, 1000)

        # Convert palette/indexed images back to RGB before tensor conversion.
        simplified_image = simplified_image.convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        return transform(simplified_image).to(image.device)


class Uniform_Image(SimplificationStrategy):
    def __init__(self):
        self.name = "uniform_image"

    def simplify(self, image):

        r = image[0].mean()
        g = image[1].mean()
        b = image[2].mean()

        r_a = torch.full_like(image[0], r)
        g_a = torch.full_like(image[0], g)
        b_a = torch.full_like(image[0], b)

        img = torch.stack([r_a, g_a, b_a])

        return img

class Gaussian_Blurr(SimplificationStrategy):
    def __init__(self, sigma = 1, inc=2):
        self.name = "gaussian_blurring"
        self.sigma = sigma
        self.inc = inc

    def simplify(self, image):
        k = 2 * int(self.sigma) +1
        gaussian_blur = GaussianBlur(kernel_size=(k, k), sigma=self.sigma)
        simp = gaussian_blur(image)
        self.sigma = self.sigma * self.inc
        return simp
