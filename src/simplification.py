from abc import ABC, abstractmethod
from typing import Optional
import torch
import torchvision.transforms.functional as F
import pywt
from skimage.color import rgb2hsv, hsv2rgb
import numpy as np
from matplotlib import pyplot as plt



class SimplificationStrategy(ABC):
    name = "base"

    @abstractmethod
    def simplify(self, image):
        pass
    
    # when performing simplification we may use another color space, so we need to restore the simplified image to the original color space before passing it to the segmentation model
    @abstractmethod
    def restore(self, simplified_image):
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
        print(f"Soft thresholding with lambda={self.lamb}. For new lambda reinitalize the WaveletSimplification class")
        coeffs_arr_thr = np.sign(coeffs_arr) * \
            np.maximum(np.abs(coeffs_arr) - self.lamb, 0.0)
        self.lamb += self.lamb_mul
        
        # Reconstruct image from thresholded coefficients
        coeffs_thresholded = pywt.array_to_coeffs(
            coeffs_arr_thr, coeff_slices, output_format='wavedec2')
        comp_value = pywt.waverec2(coeffs_thresholded, self.wavelet)
        hsv_image[:,:,2] = comp_value

        plt.imshow(comp_value, cmap="gray")
        plt.colorbar()
        plt.show()

        #back to RGB
        simplified_image = hsv2rgb(hsv_image)
        simplified_image = torch.from_numpy(simplified_image).permute(2, 0, 1).to(image.device).float()
        return simplified_image

    def restore(self, simplified_image):
        # Implement wavelet_ restoration logic here
        pass

class SegmentRepairPipeline():
    def __init__(self, 
                model, 
                img, 
                strategy: Optional[SimplificationStrategy],
                lr_opt = 0.01,):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.model.eval()
        self.img = img
        self.simp_strategy = strategy
        self.lr_opt = lr_opt

        self.reference_output = self._get_output(self.img, use_grad=False)
        plt.imshow(self.reference_output.detach().cpu().numpy().argmax(axis=0))
        plt.title("ref out")
        

        # save intermediate results
        self.all_simp_imgs = None
        self.all_seg_simp = None

    def _get_output(self, image, use_grad):
        # implement
        if image.dim() == 3:
            image = image.unsqueeze(0)
        normalized_img = F.normalize(image, mean=(
            0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).to(self.device)

        if use_grad:
            normalized_img = normalized_img.requires_grad_(True)
        with torch.set_grad_enabled(use_grad):
            output = self.model(normalized_img)['out']
        output = torch.nn.functional.softmax(output, dim=1).squeeze()
        return output
    
    def _tv_loss(self, img):
        """Compute the total variation loss of an image."""
        dx = img[:, 1:, :] - img[:, :-1, :]
        dy = img[:, :, 1:] - img[:, :, :-1]
        return (dx.pow(2).mean() + dy.pow(2).mean())
    
    def _calculate_dice_loss(self, x_output, reference_output):
        """
        Calculate the total loss for the repairing step, which includes the Dice loss, total variation loss, and the alpha term.

        """
        intersection = 2.0 * (x_output * reference_output).sum(dim=(1, 2))
        denom = (x_output.pow(2) + reference_output.pow(2)).sum(dim=(1, 2))
        #denom = x_output.sum(dim=(1, 2)) + reference_output.sum(dim=(1, 2))
        dice = (intersection + 1e-6) / (denom + 1e-6)
        dice_loss = 1.0 - dice.mean()
        return dice_loss
        # calculate Dice loss using a stable formulation with smoothing
        # eps = 1e-6
        # intersection = (x_output * reference_output).sum(dim=(1, 2))
        # denom = x_output.sum(dim=(1, 2)) + reference_output.sum(dim=(1, 2))
        # dice = (2.0 * intersection + eps) / (denom + eps)
        # dice_loss = 1.0 - dice[1:].mean()
        # return dice_loss
    
    def _calculate_loss(self, x, alpha, x_output, reference_output, lambda_tv, lambda_alpha, dice_weight=0.1):
        """
                Calculate the total loss for the repairing step, which includes the Dice loss, total variation loss, and the alpha term.

                """
        # calculate Dice loss
        dice_loss = self._calculate_dice_loss(x_output, reference_output)

        loss_alpha = lambda_alpha * alpha.mean()
        loss_tv = lambda_tv * self._tv_loss(alpha.unsqueeze(0))

        loss_simplification = - loss_alpha
        loss_simplification += loss_tv

        loss = dice_weight * dice_loss + loss_simplification
        print("dice_weight", dice_weight)

        return loss, dice_loss, loss_simplification, -loss_alpha, loss_tv
    
    def _optimize_alpha(self, alpha_param, last_simp_img, simp_img, lambda_tv, lambda_alpha, dice_er, dice_weight=0.1, max_it_opt=500, patience=50):

        optimizer = torch.optim.Adam([alpha_param], lr=self.lr_opt)

        best_simp_loss = torch.inf
        best_out_repaired = None
        best_repaired_img = None
        best_alpha = alpha_param.clone()
        best_dice_loss = torch.inf
        best_total_loss = torch.inf
        no_improvement = 0

        log_losses = {"total_loss": [], 
                      "dice_loss": [],
                      "loss_simplification": [],
                      "loss_alpha": [],
                      "loss_tv": [],
                      }


        for opt_it in range(max_it_opt):
            optimizer.zero_grad()

            alpha = torch.clamp(alpha_param, 0, 1)
            a_stacked = torch.stack([alpha] * 3, dim=0).to(self.device)

            repaired_img = last_simp_img * (1 - a_stacked) + simp_img * a_stacked

            #repaired_img_restored = self.simp_strategy.restore(repaired_img)

            out_repaired = self._get_output(
                repaired_img, use_grad=True)

            #calculate loss
            loss, dice_loss, loss_simplification, loss_alpha, loss_tv = self._calculate_loss(
                repaired_img, alpha, out_repaired, self.reference_output, lambda_tv, lambda_alpha, dice_weight)
            
            log_losses["total_loss"].append(loss.item())
            log_losses["dice_loss"].append(dice_loss.item())
            log_losses["loss_simplification"].append(
                loss_simplification.item())
            log_losses["loss_alpha"].append(loss_alpha.item())
            log_losses["loss_tv"].append(loss_tv.item())
            
            if (dice_loss < dice_er):
                # lower dice weight
                dice_weight = np.clip(dice_weight * 0.9, 0., 20.)
                # save if dice_loss is good enough and simplification loss is better
                if (loss_simplification < best_simp_loss):
                    best_dice_loss = dice_loss
                    best_simp_loss = loss_simplification
                    best_total_loss = loss
                    best_repaired_img = repaired_img.clone().detach()
                    best_alpha = alpha.clone().detach()
                    best_out_repaired = out_repaired.clone().detach()
                    no_improvement = 0
                else:
                    no_improvement +=1    
            else:
                no_improvement +=1
                if dice_loss >= dice_er:
                    dice_weight *= 1.2 # much more weight, we did not reach the error
                elif dice_loss >= best_dice_loss: 
                    dice_weight *= 1.1 # more weight
                dice_weight = np.clip(dice_weight, 0., 20.)

            if no_improvement >= patience:
                print(
                    f"No improvement in Dice and loss for {no_improvement} iterations, stopping.")
                break
            
            loss.backward()
            optimizer.step()  

        plt.plot(log_losses["total_loss"], label="total_loss") 
        plt.plot(log_losses["dice_loss"], label="dice_loss")
        plt.plot(log_losses["loss_simplification"],
                 label="loss_simplification")
        plt.plot(log_losses["loss_alpha"], label="loss_alpha")
        plt.plot(log_losses["loss_tv"], label="loss_tv")
        plt.legend()
        plt.show()

        return best_repaired_img, best_out_repaired, best_alpha, best_dice_loss, best_simp_loss, best_total_loss   


    def do_simplification_procedure(self, max_simp_it=20, max_it_opt=500, dice_er=0.01, lambda_tv=0.8, lambda_alpha=0.1, inital_alpha=0.0):

        # save intermediate results
        self.all_simp_imgs = [] # all simplified images (before repair)
        self.all_seg_simp = [] # all segmentation outputs for simplified images
        self.all_rep_imgs = [] # all repaired images
        self.all_seg_rep = [] # all segmentation outputs for repaired images
        self.all_alphas = [] # all alpha masks for repaired images

        last_simp_img = self.img
        last_simp_img = last_simp_img.to(self.device)

        for i in range(max_simp_it):
            # Simplification step
            print("last simp img shape:", last_simp_img.shape)
            simp_img = self.simp_strategy.simplify(last_simp_img)
            self.all_simp_imgs.append(simp_img.detach().cpu().numpy())

            # prepare alpha parameter for optimization
            alpha_param = torch.full_like(simp_img[0], inital_alpha, requires_grad=True, device=self.device, dtype=torch.float)
            a_stacked = torch.stack([alpha_param] * 3, dim=0).to(self.device)
            

            # get segmentation output for simplified image
            #simp_img_restored = self.simp_strategy.restore(simp_img)
            out_simp = self._get_output(simp_img, use_grad=False)
            self.all_seg_simp.append(out_simp.detach().cpu().numpy())

            dice_loss = self._calculate_dice_loss(out_simp, self.reference_output)

            # dice loss is not good enough, so we need to repair the simplified image
            print("dice_loss", dice_loss.item())
            if dice_loss > dice_er:
                print("do repairing")
                best_rep_img, best_out_repaired, rep_alpha, rep_dice_loss, rep_simp_loss, rep_total_loss = self._optimize_alpha(alpha_param, 
                                last_simp_img, 
                                simp_img, 
                                lambda_tv, 
                                lambda_alpha, 
                                dice_er, 
                                dice_weight=0.01, 
                                max_it_opt=max_it_opt)
                
                last_simp_img = best_rep_img.clone().detach()
                self.all_rep_imgs.append(best_rep_img.detach().cpu().numpy())
                self.all_seg_rep.append(
                    best_out_repaired.detach().cpu().numpy())
                self.all_alphas.append(rep_alpha.detach().cpu().numpy())

                plt.imshow(last_simp_img.permute(1, 2, 0).cpu().numpy())
                plt.title(f"repaired image)")
                plt.show()
                plt.subplot(1,2,1)
                plt.imshow(best_out_repaired.detach().cpu().numpy().argmax(axis=0))
                plt.subplot(1,2,2)
                plt.imshow(self.reference_output.detach().cpu().numpy().argmax(axis=0))
                plt.show()

                print("best_dice_loss", rep_dice_loss.item())
                if rep_dice_loss > dice_er:
                    print("Best dice loss after repair is still not good enough, stopping simplification procedure.")
                    break
            else:
                # dice loss is good enough, so we can use the simplified image
                last_simp_img = simp_img.clone().detach()

        return {"all_simp_imgs":self.all_simp_imgs, "all_simp_seg": self.all_seg_simp, "all_rep_imgs":self.all_rep_imgs, "all_seg_rep": self.all_seg_rep, "all_alphas": self.all_alphas}
    