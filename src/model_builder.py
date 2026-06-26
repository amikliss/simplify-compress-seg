"""
- PAPER
https://arxiv.org/pdf/2008.07357.pdf
- CODE
https://github.com/kechua/DART20/blob/master/damri/model/unet.py
"""
from __future__ import annotations

from typing import (
    Dict,
    List,
)
from datetime import datetime
from omegaconf import OmegaConf
import wandb
import torch
from torch import (
    nn,
    log,
    tensor
)
from torch.special import entr
from torch.nn.functional import one_hot
# from dpipe.layers.resblock import ResBlock2d
# from dpipe.layers.conv import PreActivation2d
from monai.networks.nets import UNet, DynUNet, SwinUNETR, SegResNet
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101
from monai.networks.layers.factories import Norm
from loss_fn import CustomCEDiceLoss
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDiceMetric, MeanIoU
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np


# from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
import matplotlib.pyplot as plt


def get_unet_module_trainer(
    # data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    trainer_cfg: OmegaConf
):
    # infered variables
    patience = model_cfg.patience * 2
    now = datetime.now()
    filename = f'model-{model_cfg.model}_{now.strftime("%Y-%m-%d-%H-%M")}'

    # init logger
    if trainer_cfg.logging:
        wandb.finish()
        logger = WandbLogger(
            project="swin-unet-training",
            log_model=True,
            name=filename
        )
    else:
        logger = None

    # return trainer
    return L.Trainer(
        limit_train_batches=trainer_cfg.limit_train_batches,
        max_epochs=trainer_cfg.max_epochs,
        logger=logger,
        callbacks=[
            EarlyStopping(
                monitor=trainer_cfg.early_stopping.monitor,
                mode=trainer_cfg.early_stopping.mode,
                patience=patience
            ),
            ModelCheckpoint(
                dirpath=trainer_cfg.model_checkpoint.dirpath,
                filename=filename,
                save_top_k=trainer_cfg.model_checkpoint.save_top_k,
                monitor=trainer_cfg.model_checkpoint.monitor,
            )
        ],
        precision='16-mixed',
        gradient_clip_val=0.5,
        devices=1,
        # limit_test_batches=50
    )


def get_unet_module(
    cfg: OmegaConf,
    load_from_checkpoint: bool = False,
    checkpoint_path: str = None
):

    unet = UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        channels=[cfg.n_filters_init * 2 ** i for i in range(cfg.depth)],
        strides=[2] * (cfg.depth - 1),
        num_res_units=cfg.num_res_units,
        norm=Norm.INSTANCE,
        dropout=cfg.dropout,
    )
    if load_from_checkpoint:
        print(cfg.checkpoint_path)
        if checkpoint_path is not None:
            this_path = checkpoint_path
        else:
            this_path = cfg.checkpoint_path
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=this_path,
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            map_location='cuda:0'
        )
    else:
        return LightningSegmentationModel(
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            metadata=cfg
        )


def get_segres_module(
    cfg: OmegaConf,
    load_from_checkpoint: bool = False,
    checkpoint_path: str = None
):

    resnet = SegResNet(
        spatial_dims=cfg.spatial_dims,
        init_filters=cfg.n_filters_init,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        dropout_prob=cfg.dropout,
        act=("PReLU", {"num_parameters": 1}),
        # channels=[cfg.n_filters_init * 2 ** i for i in range(cfg.depth)],
        norm=Norm.INSTANCE,
    )
    if load_from_checkpoint:
        print(cfg.checkpoint_path)
        if checkpoint_path is not None:
            this_path = checkpoint_path
        else:
            this_path = cfg.checkpoint_path
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=this_path,
            model=resnet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            map_location='cuda:0'
        )
    else:
        return LightningSegmentationModel(
            model=resnet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            metadata=cfg
        )


def get_deeplabv3_resnet50_module(
    cfg: OmegaConf,
    load_from_checkpoint: bool = False,
    checkpoint_path: str = None
):

    resnet = deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=False,
        num_classes=cfg.out_channels
    )
    if load_from_checkpoint:
        if checkpoint_path is not None:
            this_path = checkpoint_path
        else:
            this_path = cfg.checkpoint_path
        print(this_path)
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=this_path,
            model=resnet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            map_location='cuda:0',
            class_weights=cfg.class_weights,
        )
    else:
        return LightningSegmentationModel(
            model=resnet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            metadata=cfg,
            class_weights=cfg.class_weights,
        )


def get_deeplabv3_resnet101_module(
    cfg: OmegaConf,
    load_from_checkpoint: bool = False,
    checkpoint_path: str = None
):

    resnet = deeplabv3_resnet101(
        weights=None,
        weights_backbone=None,
        aux_loss=False,
        num_classes=cfg.out_channels
    )
    if load_from_checkpoint:
        print(cfg.checkpoint_path)
        if checkpoint_path is not None:
            this_path = checkpoint_path
        else:
            this_path = cfg.checkpoint_path
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=this_path,
            model=resnet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            map_location='cuda:0'
        )
    else:
        return LightningSegmentationModel(
            model=resnet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            metadata=cfg
        )


# def get_segresvae_module(
#     cfg: OmegaConf,
#     load_from_checkpoint: bool = False,
#     checkpoint_path: str = None
# ):

#     resnet = SegResNetVAE(
#         input_image_size=(512, 896),

#         spatial_dims=cfg.spatial_dims,
#         init_filters=cfg.n_filters_init,
#         in_channels=cfg.in_channels,
#         out_channels=cfg.out_channels,
#         dropout_prob=cfg.dropout,
#         # channels=[cfg.n_filters_init * 2 ** i for i in range(cfg.depth)],
#         norm=Norm.INSTANCE,
#     )
#     if load_from_checkpoint:
#         print(cfg.checkpoint_path)
#         if checkpoint_path is not None:
#             this_path = checkpoint_path
#         else:
#             this_path = cfg.checkpoint_path
#         return LightningSegmentationModel.load_from_checkpoint(
#             checkpoint_path=this_path,
#             model=resnet,
#             binary_target=True if cfg.out_channels == 1 else False,
#             lr=cfg.lr,
#             patience=cfg.patience,
#             map_location='cuda:0'
#         )
#     else:
#         return LightningSegmentationModel(
#             model=resnet,
#             binary_target=True if cfg.out_channels == 1 else False,
#             lr=cfg.lr,
#             patience=cfg.patience,
#             metadata=cfg
#         )


def get_dynunet_module(
    cfg: OmegaConf,
    load_from_checkpoint: bool = False,
    checkpoint_path: str = None
):
    print("type strides", type(cfg.strides))
    print("type kernel_size", type(cfg.kernel_size))
    unet = DynUNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        kernel_size=[int(cfg.kernel_size)] * (cfg.depth),
        filters=cfg.n_filters_init,
        # first stride needs to be 1 to avoid direct downsampling and dimension mismatch
        strides=cfg.strides,
        upsample_kernel_size=[int(cfg.upsample_kernel_size)] * (cfg.depth),
        norm_name=Norm.INSTANCE,
        dropout=cfg.dropout,
        res_block=cfg.res_block
    )
    print("strides", unet.strides)
    if load_from_checkpoint:
        print(cfg.checkpoint_path)
        if checkpoint_path is not None:
            this_path = checkpoint_path
        else:
            this_path = cfg.checkpoint_path
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=this_path,
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            map_location='cuda:0',
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
        )
    else:
        return LightningSegmentationModel(
            model=unet,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            metadata=cfg
        )


def get_swinunetr_module(
    cfg: OmegaConf,
    load_from_checkpoint: bool = False,
    checkpoint_path: str = None
):
    swinunetr = SwinUNETR(
        # img_size=cfg.img_size,
        in_channels=cfg.in_channels,
        depths=cfg.depths,
        num_heads=cfg.num_heads,
        feature_size=cfg.feature_size,
        out_channels=cfg.out_channels,
        spatial_dims=cfg.spatial_dims
    )
    if load_from_checkpoint:
        if checkpoint_path is not None:
            this_path = checkpoint_path
        else:
            this_path = cfg.checkpoint_path
        return LightningSegmentationModel.load_from_checkpoint(
            checkpoint_path=this_path,
            model=swinunetr,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            map_location='cuda:0'
        )
    else:
        return LightningSegmentationModel(
            model=swinunetr,
            binary_target=True if cfg.out_channels == 1 else False,
            lr=cfg.lr,
            patience=cfg.patience,
            include_background=cfg.include_background,
            ignore_index=cfg.ignore_index,
            loss=cfg.loss,
            metadata=cfg
        )

class LightningSegmentationModel(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        patience: int = 5,
        binary_target: bool = False,
        metadata: OmegaConf = None,
        include_background: bool = False,
        loss: str = 'DiceCELoss',
        ignore_index: int = None,  # just relevant for crossentropyloss
        class_weights: list = None,
    ):
        super().__init__()
        # this would save the model as hyperparameter, not desired!
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.patience = patience
        self.metadata = metadata
        self.loss_str = loss
        self.ignore_index = ignore_index if ignore_index != "None" else None
        self.class_weights = torch.tensor(
            class_weights) if (class_weights != "None") & (class_weights != None) else None
        # self.weights = torch.tensor([0.219519, 0.428998, 0.258639, 0.813367, 0.827632, 1.077600, 2.539359, 1.818306,
        #                             0.237626, 0.763106, 0.197252, 2.963214, 6.216211, 0.754739, 2.034963, 2.187746, 1.703594, 5.440860, 3.917624])

        if loss == 'DiceCELoss':
            self.loss = DiceCELoss(
                softmax=False if binary_target else True,
                sigmoid=True if binary_target else False,
                to_onehot_y=False if binary_target else True,
            )
            print('Using DiceCELoss')
        elif loss == 'CrossEntropyLoss':
            if self.ignore_index is not None:
                self.loss = torch.nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index)
                print(
                    f'Using CrossEntropyLoss with ignore_index={self.ignore_index}')
            else:
                self.loss = torch.nn.CrossEntropyLoss()
                print(f'Using CrossEntropyLoss without ignore_index')
        elif loss == 'CustomCEDiceLoss':
            if self.ignore_index is not None:
                self.loss = CustomCEDiceLoss(
                    ignore_index=self.ignore_index, weights=self.class_weights)
                print(
                    f'Using CustomCEDiceLoss with ignore_index={self.ignore_index} and class_weights={self.class_weights}')
            else:
                self.loss = CustomCEDiceLoss(weights=self.class_weights)
                print(
                    f'Using CustomCEDiceLoss and class_weights={self.class_weights}')
        else:
            raise ValueError(
                f"Loss {loss} not available. Choose from 'DiceCELoss' or 'CrossEntropyLoss'.")
        # self.loss = DiceCELoss(
        #     softmax=False if binary_target else True,
        #     sigmoid=True if binary_target else False,
        #     to_onehot_y=False if binary_target else True,
        # )

        self.dsc = DiceMetric(
            include_background=include_background, reduction="none")
        self.iou = MeanIoU(
            include_background=include_background, reduction="none")
        self.hausdorff = HausdorffDistanceMetric(
            include_background=include_background, reduction="none", percentile=95)
        if self.metadata['out_channels'] == 1:
            num_classes = 2
        else:
            num_classes = self.metadata['out_channels']
        self.sdsc = SurfaceDiceMetric(include_background=include_background, class_thresholds=(
            num_classes - 1) * [3], reduction="none")

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        outputs = outputs.get("pred", outputs) if hasattr(
            outputs, "get") else outputs
        outputs = outputs.get("out", outputs) if hasattr(
            outputs, "get") else outputs
        num_classes = max(outputs.shape[1], 2)
        classes_for_target_oh = int(max(target.max().item() + 1, num_classes))

        if ((num_classes + (self.ignore_index == num_classes)) != classes_for_target_oh):
            raise ValueError(
                f"Something is wrong with the number of classes. We have {num_classes} out channels of the network and {classes_for_target_oh} classes in the target. The out channels of the network should be the same number as the classes in the target + an optional ignore index. The ignored index should be the last index if we have one class more than output classes.")

        if self.loss_str == 'CrossEntropyLoss' or self.loss_str == 'CustomCEDiceLoss':
            target = target.squeeze(dim=1).long()
        else:
            target = target.long()

        loss = self.loss(outputs, target)

        # fail early on divergence
        if not torch.isfinite(loss):
            raise ValueError(
                f"Non-finite loss at epoch={self.current_epoch}, batch={batch_idx}, loss={loss.item()}"
            )

        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1

        target_oh = torch.nn.functional.one_hot(
            target.squeeze(dim=1), num_classes=classes_for_target_oh).moveaxis(-1, 1)

        outputs = torch.nn.functional.one_hot(
            outputs, num_classes=num_classes).moveaxis(-1, 1)

        if self.ignore_index is not None:
            indices = torch.where(target == self.ignore_index)
            outputs[indices[0], :, indices[1], indices[2]] = 0

            target_oh = torch.cat([
                target_oh[:, : self.ignore_index], target_oh[:, self.ignore_index + 1:]], dim=1)
            outputs = torch.cat([
                outputs[:, : self.ignore_index], outputs[:, self.ignore_index + 1:]], dim=1)

        dsc = self.dsc(outputs, target_oh)
        dsc = dsc.nanmean().nan_to_num(0)

        self.log_dict({
            'train_loss': loss,
            'train_dsc': dsc,
        })
        return {
            'loss': loss
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        outputs = outputs.get("out", outputs) if hasattr(
            outputs, "get") else outputs
        num_classes = max(outputs.shape[1], 2)
        classes_for_target_oh = int(max(target.max().item() + 1, num_classes))

        if ((num_classes + (self.ignore_index == num_classes)) != classes_for_target_oh):
            raise ValueError(
                f"Something is wrong with the number of classes. We have {num_classes} out channels of the network and {classes_for_target_oh} classes in the target. The out channels of the network should be the same number as the classes in the target + an optional ignore index. The ignored index should be the last index if we have one class more than output classes.")

        if self.loss_str == 'CrossEntropyLoss' or self.loss_str == 'CustomCEDiceLoss':
            target = target.squeeze(dim=1).long()
        else:
            target = target.long()

        loss = self.loss(outputs, target)

        # print(loss)

        if num_classes > 2:
            outputs = outputs.argmax(1)
            # plt.imshow(outputs[0].cpu().detach().numpy())
            # plt.colorbar()
            # plt.show()
        else:
            outputs = (outputs > 0) * 1

        target_oh = torch.nn.functional.one_hot(
            target.squeeze(dim=1), num_classes=classes_for_target_oh).moveaxis(-1, 1)

        outputs = torch.nn.functional.one_hot(
            outputs, num_classes=num_classes).moveaxis(-1, 1)

        if self.ignore_index is not None:
            indices = torch.where(target == self.ignore_index)
            outputs[indices[0], :, indices[1], indices[2]] = 0

            target_oh = torch.cat([
                target_oh[:, : self.ignore_index], target_oh[:, self.ignore_index + 1:]], dim=1)
            outputs = torch.cat([
                outputs[:, : self.ignore_index], outputs[:, self.ignore_index + 1:]], dim=1)

        dsc = self.dsc(outputs, target_oh)
        dsc = dsc.nanmean().nan_to_num(0)

        # iou
        iou = self.iou(outputs, target_oh)
        iou_mean = iou.nanmean().nan_to_num(0)

        self.log_dict({
            'val_loss': loss,
            'val_dsc': dsc,
            'val_iou': iou_mean,
        })
        return {
            'loss': loss,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        outputs = outputs.get("out", outputs) if hasattr(
            outputs, "get") else outputs
        # plt.subplot(1, 2, 1)
        # plt.imshow(input.cpu().squeeze())
        # plt.contour(target.cpu().squeeze(), colors='r')
        # plt.subplot(1, 2, 2)
        # plt.imshow(outputs.argmax(dim=1).cpu().squeeze(), vmin=0, vmax=3)
        num_classes = max(outputs.shape[1], 2)
        classes_for_target_oh = int(max(target.max().item() + 1, num_classes))

        if ((num_classes + (self.ignore_index == num_classes)) != classes_for_target_oh):
            raise ValueError(
                f"Something is wrong with the number of classes. We have {num_classes} out channels of the network and {classes_for_target_oh} classes in the target. The out channels of the network should be the same number as the classes in the target + an optional ignore index. The ignored index should be the last index if we have one class more than output classes.")

        if self.loss_str == 'CrossEntropyLoss' or self.loss_str == 'CustomCEDiceLoss':
            target = target.squeeze(dim=1).long()
            # outputs = outputs.softmax(dim=1)
        else:
            target = target.long()

        loss = self.loss(outputs, target)

        if num_classes > 2:
            outputs = outputs.argmax(1)
        else:
            outputs = (outputs > 0) * 1

        target_oh = torch.nn.functional.one_hot(
            target.squeeze(dim=1), num_classes=classes_for_target_oh).moveaxis(-1, 1)

        outputs = torch.nn.functional.one_hot(
            outputs, num_classes=num_classes).moveaxis(-1, 1)

        if self.ignore_index is not None:
            indices = torch.where(target == self.ignore_index)
            outputs[indices[0], :, indices[1], indices[2]] = 0

            target_oh = torch.cat([
                target_oh[:, : self.ignore_index], target_oh[:, self.ignore_index + 1:]], dim=1)
            outputs = torch.cat([
                outputs[:, : self.ignore_index], outputs[:, self.ignore_index + 1:]], dim=1)

        dsc = self.dsc(outputs, target_oh)
        dsc_mean = dsc.nanmean().nan_to_num(0)

        # Log per-class dice scores
        if dsc.ndim > 1:
            dsc_per_class = dsc.nanmean(dim=0).nan_to_num(0)
        else:
            dsc_per_class = dsc.nan_to_num(0)
        for i, d in enumerate(dsc_per_class):
            self.log(f'test_dsc_class_{i}', d)

        # print("dsc", dsc)
        # print("dsc_mean", dsc_mean)
        # plt.show()

        # iou
        iou = self.iou(outputs, target_oh)
        iou_mean = iou.nanmean().nan_to_num(0)

        # Log per-class dice scores
        if iou.ndim > 1:
            iou_per_class = iou.nanmean(dim=0).nan_to_num(0)
        else:
            iou_per_class = iou.nan_to_num(0)
        for i, d in enumerate(iou_per_class):
            self.log(f'test_iou_class_{i}', d)

        self.log_dict({
            'test_loss': loss,
            'test_dsc': dsc_mean,
            'test_iou': iou_mean
        })
        return {
            'loss': loss,
            'dsc': dsc_mean,
            'test_iou': iou_mean
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input = batch['input']
        target = batch['target']
        target[target < 0] = 0
        outputs = self(input)
        outputs = outputs.get("out", outputs) if hasattr(
            outputs, "get") else outputs
        outputs = outputs.softmax(1)
        print("applied softmax in predict step")
        print("outputs shape", outputs.shape)
        # num_classes = max(outputs.shape[1], 2)
        # if num_classes > 2:
        #     probs = torch.softmax(outputs, 1).detach()
        #     outputs = outputs.argmax(dim=1, keepdim=True).detach()
        # else:
        #     probs = outputs.sigmoid().detach()
        #     outputs = (outputs > 0).long().detach()
        # predicted_segmentation = one_hot(outputs.squeeze(
        #     1), num_classes=num_classes).moveaxis(-1, 1)
        # target_segmentation = one_hot(target.squeeze(
        #     1), num_classes=num_classes).moveaxis(-1, 1)
        # dice = self.dsc(predicted_segmentation,
        #                 target_segmentation).nanmean(-1).nan_to_num(0).cpu().detach()
        # hausdorff = self.hausdorff(
        #     predicted_segmentation, target_segmentation).nanmean(-1).nan_to_num(0).cpu().detach()
        # surface_dice = self.sdsc(
        #     predicted_segmentation, target_segmentation).nanmean(-1).nan_to_num(0).cpu().detach()
        # entropy = 1 - (entr(probs).sum(1).mean((-1, -2)) /
        #                log(tensor(num_classes)))

        # metrics = {
        #     'dice': dice,
        #     'hausdorff': hausdorff,
        #     'surface_dice': surface_dice,
        #     'entropy': entropy
        # }
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }

    def scale_space_step(self, input, device='cuda:0'):

        # reshape so it has the dimension [scales, 1, height, width]
        input = torch.reshape(
            input, (input.shape[0], 1, input.shape[1], input.shape[2]))
        input = input.to(device=self.device)
        outputs = self(input)  # predict for each scale
        outputs = outputs.get("out", outputs) if hasattr(
            outputs, "get") else outputs
        # create segmentation output
        num_classes = max(outputs.shape[1], 2)
        if num_classes > 2:
            outputs = outputs.argmax(dim=1, keepdim=True).detach()
        else:
            outputs = (outputs > 0).long().detach()
        return outputs.squeeze()

    def run_scale_space_prediction(self, dataloader):
        self.eval()
        outputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch_flattened = torch.flatten(
                    batch["input"], start_dim=0, end_dim=1)
                out = self.scale_space_step(batch_flattened)
                out = out.reshape(
                    batch["input"].shape[0], batch["input"].shape[1], out.shape[1], out.shape[2])
                out.cpu()
                outputs.append(out)
        # output shape is [batches, batch_size, scales, height, width]
        return outputs
