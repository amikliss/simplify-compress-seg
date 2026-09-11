from typing import Optional
import torch
import torchvision.transforms.functional as F
from torch.nn.functional import pad, conv2d, interpolate

import numpy as np
from matplotlib import pyplot as plt
from simplification import SimplificationStrategy

from filter import MedianPool2d, scharr_filter
from torch import autograd


class SegmentRepairPipeline():
    def __init__(self, 
                model, 
                img, 
                strategy: Optional[SimplificationStrategy],
                lr_opt = 0.01,
                seg_pres_loss = "margin"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.model.eval()
        self.img = img
        self.simp_strategy = strategy
        self.lr_opt = lr_opt

        self.reference_output = self._get_output(self.img, use_grad=False)
        self.reference_output_hard = self.reference_output.argmax(dim=0).long()

        self.median_filter = MedianPool2d(kernel_size=35, stride=1, same=True)

        self.img_magnitude = self._compute_img_magnitude(self.img)

        plt.imshow(self.img_magnitude.cpu().numpy())
        plt.show()

        assert seg_pres_loss in ["maximize", "soft_dice", "margin"], "Choose a seg_pres_los! Options are: 'maximize', soft_dice', 'margin'."

        self.seg_pres_loss = seg_pres_loss

        # save intermediate results
        self.all_simp_imgs = None
        self.all_seg_simp = None

    def _get_output(self, image, use_grad):
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

    def _tv_loss(self,img):
        """Compute the total variation loss of an image."""
        dx = img[:,1:, :] - img[:,:-1, :]
        dy = img[:,:, 1:] - img[:,:, :-1]

        zeros = torch.zeros(1,1, img.shape[1]).to("cuda")

        dx = torch.cat([dx, zeros], dim=1)
        dy = torch.cat([dy, torch.zeros(1,img.shape[2], 1).to("cuda")], dim=2)

        # return (abs(dx).sum() + abs(dy).sum()) / (img.shape[-2] * img.shape[-1])
        # return dx.abs().mean() + dy.abs().mean()
        return torch.sqrt(pow(dx, 2) + pow(dy, 2) + 1e-8).mean()
    
    
    def _compute_img_magnitude(self, img, eps=1e-8):

        if img.ndim != 3:
            raise ValueError(f"Expected img shape (C, H, W), got {img.shape}")

        # Convert image to one luminance-like channel for the edge map.
        image_gray = img.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

        kx, ky = scharr_filter()
        kx = kx.to(device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        ky = ky.to(device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

        # Avoid artificial edges introduced by zero-padding at image boundaries.
        image_gray = pad(image_gray, (1, 1, 1, 1), mode="replicate")

        gx = conv2d(image_gray, kx)
        gy = conv2d(image_gray, ky)

        gradient_magnitude = torch.sqrt(gx.square() + gy.square() + eps)
        gradient_magnitude = gradient_magnitude.squeeze(0).squeeze(0)  # (H, W)

        return gradient_magnitude

    
    def _boundary_aware_tv_loss(self, gradient_magnitude, alpha, edge_weight=1.0):
        """
        Edge-aware total-variation loss for alpha.

        Image edges receive lower smoothness penalties, allowing alpha
        boundaries to align with image boundaries.

        Args:
            alpha: Mask/logit tensor with shape (H, W), (1, H, W), or (C, H, W).
            edge_weight: Strength of edge preservation. Must be non-negative.
            eps: Numerical-stability constant.

        Returns:
            Scalar tensor.
        """

        if alpha.ndim == 2:
            alpha = alpha.unsqueeze(0)  # (1, H, W)
        elif alpha.ndim != 3:
            raise ValueError(
                f"Expected alpha shape (H, W) or (C, H, W), got {alpha.shape}"
            )

        # Gradients of alpha, not img.
        dx = alpha[:, 1:, :] - alpha[:, :-1, :]  # (C_alpha, H-1, W)
        dy = alpha[:, :, 1:] - alpha[:, :, :-1]  # (C_alpha, H, W-1)

        # Associate each forward difference with the edge magnitude at its start.
        gm_dx = gradient_magnitude[:-1, :]  # (H-1, W)
        gm_dy = gradient_magnitude[:, :-1]  # (H, W-1)

        weight_dx = 1.0 / (1.0 + edge_weight * gm_dx)
        weight_dy = 1.0 / (1.0 + edge_weight * gm_dy)

        loss_x = (weight_dx.unsqueeze(0) * dx.abs()).mean()
        loss_y = (weight_dy.unsqueeze(0) * dy.abs()).mean()

        # plt.subplot(121)
        # plt.imshow((weight_dx.unsqueeze(0) * dx.abs()).cpu().detach().squeeze())
        # plt.subplot(122)
        # plt.imshow((weight_dy.unsqueeze(0) * dy.abs()
        #             ).cpu().detach().squeeze())
        # plt.show()

        return loss_x + loss_y
    
    def _to_one_hot(self, labels, num_classes):
        """Convert label maps of shape (H, W) to one-hot tensors of shape (C, H, W)."""
        labels = labels.long()
        return torch.nn.functional.one_hot(labels, num_classes=num_classes).permute(2, 0, 1).float()

    def _calculate_dice_loss(self, x_output, reference_output):
        """Soft Dice loss for probability maps of shape (C, H, W)."""
        eps = 1e-6
        intersection = (x_output * reference_output).sum(dim=(1, 2))
        denom = (x_output.pow(2) + reference_output.pow(2)).sum(dim=(1, 2))
        dice = (2.0 * intersection + eps) / (denom + eps)
        return 1- dice.mean()

    def _calculate_hard_dice_loss(self, x_output, reference_output):
        """Hard Dice loss computed from predicted and reference class labels."""
        if x_output.dim() == 3:
            pred_labels = x_output.argmax(dim=0).long()
        else:
            pred_labels = x_output.long()

        if reference_output.dim() == 3:
            ref_labels = reference_output.argmax(dim=0).long()
        else:
            ref_labels = reference_output.long()

        num_classes = 21
        pred_one_hot = self._to_one_hot(pred_labels, num_classes)
        ref_one_hot = self._to_one_hot(ref_labels, num_classes)

        eps = 1e-6
        intersection = (pred_one_hot * ref_one_hot).sum(dim=(1, 2))
        denom = pred_one_hot.sum(dim=(1, 2)) + ref_one_hot.sum(dim=(1, 2))
        dice = (2.0 * intersection + eps) / (denom + eps)

        if dice.numel() > 1:
            return 1.0 - dice[1:].mean()
        return 1.0 - dice.mean()

    def _segmentation_preservation_loss_maximize(self, x_output):
        """sum up the (1- probability) of the wrong pixels. Bigger number for many wrong pixels, small nuber for less wrong pixels"""
        x_output_hard = x_output.argmax(0).detach()
        wrong_mask = (x_output_hard != self.reference_output_hard).detach()

        y, x = torch.nonzero(wrong_mask, as_tuple=True)
        ref_labels = self.reference_output_hard[y, x]

        # pick the reference-class probability at each wrong pixel
        probs = x_output[ref_labels, y, x]

        return (1 - probs).mean()
    

    def _segmentation_preservation_loss_margin(self, x_output):
        """
        Margin-based loss with adaptive margin from reference output.
        """
        # original label
        org_argmax = self.reference_output_hard

        # full ranking of classes (C, H, W) -> indices
        full_argsort = x_output.detach().argsort(dim=0, descending=True)
        top1_idx = full_argsort[0]      # (H,W) indices of highest predicted class
        top2_idx = full_argsort[1]      # (H,W) indices of second-highest

        # choose the "largest other" class: if top1 == org_argmax use top2 else top1
        largest_other = torch.where(top1_idx == org_argmax, top2_idx, top1_idx)


        # output probabilities of reference class 
        org_prob = torch.gather(x_output, dim=0, index=org_argmax.unsqueeze(0))
        # output_probabilities of largest other class
        largest_other_prob = torch.gather(x_output, dim=0, index=largest_other.unsqueeze(0))

        ref_sorted = self.reference_output.detach().sort(
            dim=0, descending=True)[0]
        margin = (ref_sorted[0] - ref_sorted[1]) * 0.5

        pres_loss = torch.clamp(largest_other_prob - org_prob + margin, min=0.0)

        return pres_loss.mean()
    
    def _calculate_loss(self, x, alpha, x_output, reference_output, lambda_tv, lambda_alpha, lambda_alpha_bias, dice_weight=0.1, magnitude_edge_weight=1.0):
        """
                Calculate the total loss for the repairing step, which includes the Dice loss, total variation loss, and the alpha term.

                """
        # calculate Dice loss
        hard_dice_loss = self._calculate_hard_dice_loss(
            x_output, self.reference_output_hard)
        
        if self.seg_pres_loss == "margin":
            seg_pres_loss = self._segmentation_preservation_loss_margin(x_output)
        elif self.seg_pres_loss == "maximize":
            seg_pres_loss = self._segmentation_preservation_loss_maximize(
                x_output)
        elif self.seg_pres_loss == "soft_dice":
            seg_pres_loss = self._calculate_dice_loss(x_output, self.reference_output)

        #print(f"use {self.seg_pres_loss}")

        dice_loss = self._calculate_dice_loss(x_output, self.reference_output)

        loss_alpha_simp = lambda_alpha * alpha.mean() # we want as much as possible from the simplified image
        loss_alpha_bias = lambda_alpha_bias * (alpha * (1- alpha)).mean() #  we want to decide more for one or the other image
        loss_tv = lambda_tv * self._tv_loss(alpha.unsqueeze(0))
        #loss_tv = lambda_tv * self._boundary_aware_tv_loss(self.img_magnitude, alpha, edge_weight = magnitude_edge_weight)

        loss_simplification = - loss_alpha_simp
        loss_simplification += loss_alpha_bias
        loss_simplification += loss_tv

        #loss = dice_weight * dice_loss + loss_simplification
        loss = dice_weight * seg_pres_loss + loss_simplification

        return loss, seg_pres_loss, dice_loss, hard_dice_loss, loss_simplification, -loss_alpha_simp, loss_alpha_bias, loss_tv
    
    
    def _optimize_alpha_multi_res(self, inital_alpha, last_simp_img, simp_img, lambda_tv, lambda_alpha, lambda_alpha_bias, lambda_magnitude_edge_weight, dice_er, dice_weight=0.1, max_it_opt=500, patience=10):

        # prepare alpha parameter for optimization

        old_resolution = torch.tensor(last_simp_img[0].shape)
        f = 20
        new_res = list(old_resolution // f)


        alpha_param = torch.full(
            new_res, inital_alpha, requires_grad=True, device=self.device, dtype=torch.float)

        # add random noise
        # alpha_param = torch.rand(last_simp_img[0].shape, requires_grad=True,
        #                device=self.device)

        grad_history = []
        alpha_history = []

        optimizer = torch.optim.AdamW(
            [alpha_param], lr=self.lr_opt, weight_decay=.1)
        # optimizer = torch.optim.Adadelta([alpha_param], lr=self.lr_opt)
        # optimizer = torch.optim.SGD([alpha_param], lr=self.lr_opt)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=30, threshold=1e-9)

        current_lr = self.lr_opt

        best_simp_loss = torch.inf
        best_out_repaired = None
        best_repaired_img = None
        best_alpha = alpha_param.clone()
        best_dice_loss = torch.inf
        best_total_loss = torch.inf
        no_improvement = 0
        best_iteration = None

        log_losses = {"total_loss": [],
                      "hard_dice_loss": [],
                      "seg_pres_loss": [],
                      "loss_simplification": [],
                      "loss_alpha": [],
                      "loss_alpha_bias": [],
                      "loss_tv": [],
                      }
        

        res_it = []
        for opt_it in range(max_it_opt):
            
            if ((opt_it % 50) ==0) & (opt_it > 100) & (f > 2):
                diffs_simp_loss = torch.tensor(log_losses["loss_simplification"][-50:-1]) - torch.tensor(log_losses["loss_simplification"][-49:])
                if sum(diffs_simp_loss)  < 0.01:
                    res_it.append(opt_it)
                    f = f - 2
                    if f <= 2:
                        f = 2

                    new_res = list(old_resolution // f)
                    print(f"Make resolution bigger at iteration {opt_it}. New resolution is {new_res}.")
                    alpha_param = interpolate(alpha_param.unsqueeze(0).unsqueeze(0), size=(
                        new_res[0], new_res[1]), mode="bilinear").squeeze().detach().requires_grad_()

                    # plt.imshow(alpha_param.clone().cpu().detach().numpy())
                    # plt.colorbar()
                    # plt.show()
                    optimizer = torch.optim.AdamW(
                        [alpha_param], lr=self.lr_opt
                        )

            optimizer.zero_grad()
            #alpha = alpha_param + 0.1 * torch.rand(alpha_param.shape, requires_grad=True,
            #                                       device=self.device)

            alpha = torch.clamp(alpha_param, 0, 1)

            # alpha = self.median_filter(alpha.unsqueeze(0).unsqueeze(0)).squeeze()

            alpha = interpolate(alpha.unsqueeze(0).unsqueeze(0), size=(old_resolution[0],old_resolution[1]), mode="bilinear").squeeze()

            a_stacked = torch.stack([alpha] * 3, dim=0).to(self.device)

            repaired_img = last_simp_img * \
                (1 - a_stacked) + simp_img * a_stacked

            out_repaired = self._get_output(
                repaired_img, use_grad=True)

            # calculate loss
            loss, seg_pres_loss, dice_loss, hard_dice_loss, loss_simplification, loss_alpha, loss_alpha_bias, loss_tv = self._calculate_loss(
                x=repaired_img,
                alpha=alpha,
                x_output=out_repaired,
                reference_output=self.reference_output,
                lambda_tv=lambda_tv,
                lambda_alpha=lambda_alpha,
                lambda_alpha_bias=lambda_alpha_bias,
                dice_weight=dice_weight,
                magnitude_edge_weight=lambda_magnitude_edge_weight)

            log_losses["total_loss"].append(loss.item())
            log_losses["hard_dice_loss"].append(hard_dice_loss.item())
            log_losses["seg_pres_loss"].append(
                dice_weight * seg_pres_loss.item())
            log_losses["loss_simplification"].append(
                loss_simplification.item())
            log_losses["loss_alpha"].append(loss_alpha.item())
            log_losses["loss_alpha_bias"].append(loss_alpha_bias.item())
            log_losses["loss_tv"].append(loss_tv.item())

            print("seg_pres_loss", seg_pres_loss)



            if (loss_simplification <= best_simp_loss):  # and (opt_it > 100):
                if seg_pres_loss <= dice_er:
                    best_seg_pres_loss = seg_pres_loss
                    best_simp_loss = loss_simplification
                    best_total_loss = loss
                    best_repaired_img = repaired_img.clone().detach()
                    best_alpha = alpha.clone().detach()
                    best_out_repaired = out_repaired.clone().detach()
                    best_iteration = opt_it
                if (abs(loss_simplification - best_simp_loss) >= 5e-3):
                    no_improvement = 0
                else:
                    no_improvement += 1
            else:
                if opt_it > 100:
                    no_improvement += 1
            if (seg_pres_loss > dice_er):
                dice_weight *= 1.1
                dice_weight = np.clip(dice_weight, 0., 1000.0)
                print(f"increase dice weight to {dice_weight}")
            else:
                dice_weight *= 0.9
                print(f"decrease dice weight to {dice_weight}")
            if (no_improvement >= patience) & (opt_it > 100) & (f == 2):
                print(
                    f"No improvement in Dice and loss for {no_improvement} iterations, stopping.")
                break

            # if (opt_it % 500) == 0:
            #     plt.figure(figsize=(12, 9))
            #     plt_alpha = alpha.clone().detach().cpu().numpy()
            #     a_min = plt_alpha.min()
            #     plt.subplot(3, 3, 1)
            #     plt.imshow(plt_alpha, cmap="hot")
            #     plt.title(
            #         f"alpha min {'{:.4f}'.format(a_min)}, max {'{:.4f}'.format(plt_alpha.max())}")
            #     plt.colorbar()

            #     # seg pres
            #     (seg_pres_grad_alpha,) = autograd.grad(
            #         seg_pres_loss,
            #         alpha_param,
            #         retain_graph=True,
            #         create_graph=False
            #     )

            #     seg_pres_loss_grad = seg_pres_grad_alpha.detach().cpu().clone()
            #     plt.subplot(3, 3, 4)
            #     plt.imshow(seg_pres_loss_grad)
            #     plt.colorbar()
            #     plt.title(f"seg_pres loss, it {opt_it}")

            #     # loss_alpha
            #     (loss_alpha_alpha,) = autograd.grad(
            #         loss_alpha,
            #         alpha_param,
            #         retain_graph=True,
            #         create_graph=False
            #     )

            #     loss_alpha_grad = loss_alpha_alpha.detach().cpu().clone()
            #     plt.subplot(3, 3, 5)
            #     plt.imshow(loss_alpha_grad)
            #     plt.colorbar()
            #     plt.title(
            #         f"alpha loss, it {opt_it}, min {'{:.8f}'.format(loss_alpha_grad.min())}, max {'{:.8f}'.format(loss_alpha_grad.max())}")
            #     print("loss_alpha_grad min, max",
            #           loss_alpha_grad.min(), loss_alpha_grad.max())
            #     # loss_alpha_bias
            #     (loss_alpha_bias_alpha,) = autograd.grad(
            #         loss_alpha_bias,
            #         alpha_param,
            #         retain_graph=True,
            #         create_graph=False
            #     )

            #     loss_alpha_bias_grad = loss_alpha_bias_alpha.detach().cpu().clone()
            #     plt.subplot(3, 3, 6)
            #     plt.imshow(loss_alpha_bias_grad)
            #     plt.colorbar()
            #     plt.title(f"alpha bias loss, it {opt_it}")

            #     # loss_tv
            #     (loss_tv_alpha,) = autograd.grad(
            #         loss_tv,
            #         alpha_param,
            #         retain_graph=True,
            #         create_graph=False
            #     )

            #     loss_tv_grad = loss_tv_alpha.detach().cpu().clone()
            #     plt.subplot(3, 3, 7)
            #     plt.imshow(loss_tv_grad)
            #     plt.colorbar()
            #     plt.title(f"tv loss, it {opt_it}")

            #     # add gradients up
            #     manual_loss_grad = dice_weight * seg_pres_loss_grad + \
            #         loss_alpha_grad + loss_alpha_bias_grad + loss_tv_grad

            #     plt.subplot(3, 3, 8)
            #     plt.imshow(manual_loss_grad.cpu().detach())
            #     plt.colorbar()
            #     plt.title(f"manual_loss, it {opt_it}")

            loss.backward(retain_graph=True)

            if alpha_param.grad is not None:
                loss_grad = alpha_param.grad.detach().cpu().clone()
                grad_history.append(loss_grad)
                alpha_history.append(alpha.detach().cpu().clone())

            # if (opt_it % 500) == 0:
            #     plt.subplot(3, 3, 9)
            #     plt.imshow(loss_grad)
            #     plt.colorbar()
            #     plt.title(f"loss_grad it {opt_it}")
            #     plt.tight_layout()
            #     plt.show()

            optimizer.step()
        print("increase res it", res_it)

        # plt.plot(log_losses["hard_dice_loss"],
        #          label="hard_dice_loss", alpha=0.7)

        
        plt.plot(log_losses["seg_pres_loss"], label="seg_pres_loss", alpha=0.7)
        plt.plot(log_losses["loss_simplification"],
                 label="loss_simplification", alpha=0.7)
        plt.plot(log_losses["loss_alpha"], label="loss_alpha", alpha=0.7)
        plt.plot(log_losses["loss_alpha_bias"],
                 label="loss_alpha_bias", alpha=0.7)
        plt.plot(log_losses["loss_tv"], label="loss_tv", alpha=0.7)
        plt.plot(log_losses["total_loss"], label="total_loss", alpha=0.7)
        if best_iteration:
            plt.plot(
                best_iteration, log_losses["loss_simplification"][best_iteration], "o", label=f"best iteration {best_iteration}")
        plt.plot(
            res_it, np.array(log_losses["loss_simplification"])[res_it], "x", label=f"increase resolution")
        plt.legend()
        plt.show()

        plt.plot(log_losses["hard_dice_loss"],
                 label="hard_dice_loss", alpha=0.7)
        plt.plot(log_losses["loss_simplification"],
                 label="loss_simplification", alpha=0.7)
        plt.plot(log_losses["loss_alpha"], label="loss_alpha", alpha=0.7)
        plt.plot(log_losses["loss_alpha_bias"],
                 label="loss_alpha_bias", alpha=0.7)
        plt.plot(log_losses["loss_tv"], label="loss_tv", alpha=0.7)
        if best_iteration:
            plt.plot(best_iteration,
                     log_losses["loss_simplification"][best_iteration], "x", label=f"best iteration {best_iteration}")
        plt.legend()
        plt.show()

        return best_repaired_img, best_out_repaired, best_alpha, best_seg_pres_loss, best_simp_loss, best_total_loss, grad_history, alpha_history

    def do_simplification_procedure(self, max_simp_it=20, max_it_opt=500, dice_er=0.01, lambda_tv=0.8, lambda_alpha=0.1, lambda_alpha_bias=0.1, inital_alpha=0.0, lambda_magnitude_edge_weight= 1.0, inital_dice_weight=0.01, patience=10):

        # save intermediate results
        self.all_simp_imgs = [] # all simplified images (before repair)
        self.all_seg_simp = [] # all segmentation outputs for simplified images
        self.all_rep_imgs = [] # all repaired images
        self.all_seg_rep = [] # all segmentation outputs for repaired images
        self.all_alphas = [] # all alpha masks for repaired images
        self.all_grad_histories = []
        self.all_alpha_history = []

        last_simp_img = self.img
        last_simp_img = last_simp_img.to(self.device)

        for i in range(max_simp_it):
            print("Simplification step", i)
            
            simp_img = self.simp_strategy.simplify(last_simp_img)
            simp_img = simp_img.to("cuda")
            self.all_simp_imgs.append(simp_img.detach().cpu())

            # get segmentation output for simplified image
            out_simp = self._get_output(simp_img, use_grad=False)
            self.all_seg_simp.append(out_simp.detach().cpu())

            seg_pres_loss = self._segmentation_preservation_loss_margin(out_simp)
            print("seg pres loss", seg_pres_loss)


            # segmentation preservation loss is not good enough, so we need to repair the simplified image
            if seg_pres_loss > dice_er:
                print("do repairing")
                best_rep_img, best_out_repaired, rep_alpha, rep_seg_pres_loss, rep_simp_loss, rep_total_loss, grad_history, alpha_history = self._optimize_alpha_multi_res(
                                inital_alpha=inital_alpha,
                                last_simp_img= last_simp_img, 
                                simp_img= simp_img, 
                                lambda_tv=lambda_tv, 
                                lambda_alpha=lambda_alpha, 
                                lambda_alpha_bias=lambda_alpha_bias,
                                lambda_magnitude_edge_weight=lambda_magnitude_edge_weight,
                                dice_er=dice_er, 
                                dice_weight=inital_dice_weight,
                                max_it_opt=max_it_opt,
                                patience=patience)
                
                # if dice is not good enough after repairing, let's stop
                if rep_seg_pres_loss > dice_er:
                    print(
                        "Best dice loss after repair is still not good enough, stopping simplification procedure.")
                    break
                
                last_simp_img = best_rep_img.clone().detach()
                self.all_rep_imgs.append(best_rep_img.detach().cpu())
                self.all_seg_rep.append(
                    best_out_repaired.detach().cpu())
                self.all_alphas.append(rep_alpha.detach().cpu())
                self.all_grad_histories.append(grad_history)
                self.all_alpha_history.append(alpha_history)
                
                plt.figure(figsize=(15, 5))
                plt.subplot(1,3,1)
                plt.imshow(last_simp_img.permute(1, 2, 0).cpu().numpy())
                plt.title(f"rep_seg_pres_loss {rep_seg_pres_loss}")
                plt.subplot(1,3,2)
                plt.imshow(rep_alpha.detach().cpu(), cmap="hot", vmin = 0, vmax= 1)
                plt.colorbar()
                plt.title(f"alpha mask")
                plt.subplot(1, 3, 3)
                plt.imshow(last_simp_img.permute(1, 2, 0).cpu().numpy())
                plt.imshow(rep_alpha.detach().cpu(),
                           cmap="hot", vmin=0, vmax=1, alpha=0.75)
                plt.show()

                plt.subplot(1,2,1)
                plt.imshow(best_out_repaired.detach().cpu().numpy().argmax(axis=0))
                plt.subplot(1,2,2)
                plt.imshow(self.reference_output.detach().cpu().numpy().argmax(axis=0))
                plt.show()
                
            else:
                # dice loss is good enough, so we can use the simplified image
                last_simp_img = simp_img.clone().detach()
                #add other images even though there was nothing repaired
                self.all_rep_imgs.append(simp_img.detach().cpu())
                self.all_seg_rep.append(out_simp.detach().cpu())
                self.all_alphas.append(torch.full_like(
                    simp_img[0], 1, requires_grad=False, device="cpu", dtype=torch.float))
                

        return {"org_input": self.img.detach().cpu(), 
                "ref_seg": self.reference_output.detach().cpu(), 
                "all_simp_imgs": self.all_simp_imgs, 
                "all_simp_seg": self.all_seg_simp, 
                "all_rep_imgs": self.all_rep_imgs, 
                "all_seg_rep": self.all_seg_rep,
                "all_alphas": self.all_alphas,
                "all_grad_histories": self.all_grad_histories,
                "all_alpha_history": self.all_alpha_history}
    