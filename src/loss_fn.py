import torch.nn as nn
import torch


class CustomCEDiceLoss(nn.Module):
    def __init__(self,
                 ignore_index: int = -100,
                 weights: list = None,
                 alpha_ce: float = 1.0,
                 alpha_dice: float = 1.0,):
        super(CustomCEDiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha_ce = alpha_ce
        self.alpha_dice = alpha_dice
        self.weights = weights  # weights.to("cuda")
        if self.weights != None:
            self.weights = self.weights.to("cuda")

        print("ignore_index", ignore_index)
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weights)

    def forward(self, inputs, targets):

        # for CE: The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general)
        ce = self.cross_entropy(inputs, targets)

        num_classes = inputs.shape[1]

        if self.weights is not None and len(self.weights) != num_classes:
            raise ValueError(

                f"length of weight tensor should be equal to the number of classes")

        num_classes_target = max(targets.max().item() + 1, num_classes)

        ocurring_classes = targets.unique().tolist()
        if self.ignore_index in ocurring_classes:
            ocurring_classes.remove(self.ignore_index)

        targets = nn.functional.one_hot(
            targets.long(), num_classes=num_classes_target).moveaxis(-1, 1)

        # compute dice for each class

        # if ignore index is inside targets, we need to pretend it is correctly predicted

        # compute softmax probabilities
        inputs_prob = torch.softmax(inputs, dim=1)  # (B, C, H, W)

        # build a mask that zeroes out positions marked as ignore in targets (if applicable)
        # so other classes at this positions do not contribute negativly
        if (self.ignore_index <= (num_classes_target - 1)) and (self.ignore_index >= 0):
            ignore_mask = (
                targets[:, self.ignore_index, :, :] == 1)  # (B, H, W)
            # valid mask: True where NOT ignore
            valid_mask = (~ignore_mask).unsqueeze(1).to(
                dtype=inputs_prob.dtype, device=inputs_prob.device)  # (B,1,H,W)
            # expand to channels and apply by multiplication
            valid_mask = valid_mask.expand(-1, inputs_prob.shape[1], -1, -1)
            inputs_safe = inputs_prob * valid_mask

            # we must adjust the indices of the occuring classes so it fits after removing the ignore class
            ocurring_classes = [
                c - 1 if c > self.ignore_index else c for c in ocurring_classes]
        else:
            inputs_safe = inputs_prob

        smooth = 1e-6

        # shrink targets to actual number of classes of inputs
        if self.ignore_index >= 0 and self.ignore_index <= num_classes_target:

            targets = torch.cat([
                targets[:, : self.ignore_index], targets[:, self.ignore_index + 1:]], dim=1)
            inputs_safe = torch.cat([
                inputs_safe[:, : self.ignore_index], inputs_safe[:, self.ignore_index + 1:]], dim=1)

        intersection = (inputs_safe * targets).sum(dim=(2, 3))
        preds_sum = inputs_safe.sum(dim=(2, 3))
        targets_sum = targets.sum(dim=(2, 3))

        dice = (2. * intersection + smooth) / \
            (preds_sum + targets_sum + smooth)

        if self.weights != None:
            dice = dice * self.weights

        # just take into account ocurring classes in the batch
        dice = dice[:, ocurring_classes]

        dice_loss = 1 - dice.mean()

        ce_dice_loss = self.alpha_ce * ce + self.alpha_dice * dice_loss

        return ce_dice_loss
