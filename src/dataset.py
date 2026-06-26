"""
This script collects the Dataset classes for the CC-359, 
ACDC and M&M data sets.

Usage: serves only as a collection of individual functionalities
Authors: Rasha Sheikh, Jonathan Lennartz
"""


# - standard packages
from pathlib import Path
# - third party packages
import pandas as pd
import torch
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop
import nibabel as nib
import os
from torchvision.io import read_image
import torchvision.transforms.functional as TF
import numpy as np
import random


class MNMv2Dataset(Dataset):
    """
    Vendors: 
        Siemens:
            Domain: Symphony, Min Spacing 1.000, Max Spacing 1.641
                Number of cases: 4024
            Domain: Trio, Min Spacing 1.000, Max Spacing 1.328
                Number of cases: 128
            Domain: Avanto, Min Spacing 0.977, Max Spacing 1.417
                Number of cases: 904 

        GE (Signa): 
            Domain: HDxt, Min Spacing 0.605, Max Spacing 1.563
                Number of cases: 618
            Domain: EXCITE, Min Spacing 1.000, Max Spacing 1.484
                Number of cases: 632
            Domain: Explorer, Min Spacing 0.781, Max Spacing 0.781
                Number of cases: 26

        Philips:
            Domain: Achieva, Min Spacing 0.684, Max Spacing 1.484
                Number of cases: 1796 

        As List: ["Symphony", "Trio", "Avanto", "HDxt", "EXCITE", "Explorer", "Achieva"]    
    """
    # siemens Min Spacing 0.977, Max Spacing 1.641 SymphonyTim, TrioTim, Avanto Fit,
    # GE Min Spacing 0.605, Max Spacing 1.563 Signa HDxt, SIGNA EXCITE, Signa Explorer
    # philips Min Spacing 0.684, Max Spacing 1.484  Achieva

    def __init__(
        self,
        data_dir,
        # domain,
        data_split_ids,
        binary_target: bool = False,
        non_empty_target: bool = True,
        # normalize can be "instanceNormVolume", "instanceNormSlice", "setNormNew", "setNormUseParams"
        normalize: str = None,
        mean_std: tuple = None,
        # mean: float = None,  # 79.3206,
        # std: float = None,  # 111.6201,
    ):
        # assert vendor in ["siemens", "ge", "philips"], f"Invalid vendor {vendor}"
        # assert mode in ["vendor", "scanner"]
        # self.domain = domain
        self.data_split_ids = data_split_ids
        self._data_dir = Path(data_dir).resolve()
        self._binary_target = binary_target
        self._non_empty_target = non_empty_target
        self._normalize = normalize
        self._mean = mean_std[0] if mean_std is not None else None
        self._std = mean_std[1] if mean_std is not None else None
        self._data_info = pd.read_csv(
            self._data_dir / "dataset_information.csv", index_col=0
        )
        self._crop = CenterCrop(256)
        self.target_spacing = 1.
        self._load_data()

    def _load_data(self):
        self.input = []
        self.target = []
        self.meta = []
        print("load data")
        normalized = False
        for idx_ in self.data_split_ids:
            print(idx_)
            # just take samples that matches with the specified domain
            # TODO: erledigt,hier möchte ich etwas ändern, möchte auch einfach angeben können sample 1:160 für Training z.B.
            # if self.domain.lower() in self._data_info.loc[case_].SCANNER.lower():
            if idx_ in self._data_info.index:

                case_path = self._data_dir / "dataset" / f"{idx_:03d}"
                modes = ["ES", "ED"]

                for mode in modes:
                    x = nib.load(case_path / f"{idx_:03d}_SA_{mode}.nii.gz")
                    y = nib.load(case_path / f"{idx_:03d}_SA_{mode}_gt.nii.gz")
                    spacing = x.header.get_zooms()[0]
                    x = torch.tensor(x.get_fdata()).moveaxis(-1, 0)
                    y = torch.tensor(y.get_fdata().astype(
                        int), dtype=torch.long).moveaxis(-1, 0)
                    # interpolate
                    scale_factor = (1 / self.target_spacing *
                                    spacing, 1 / self.target_spacing * spacing)
                    x = interpolate(
                        x.unsqueeze(1),
                        scale_factor=scale_factor,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1)
                    y = interpolate(
                        y.unsqueeze(1).float(),
                        scale_factor=scale_factor,
                        mode='nearest'
                    ).long().squeeze(1)

                    x = self._crop(x)
                    y = self._crop(y)

                    if self._normalize == "instanceNormVolume":
                        x_mean = x.mean()
                        x_std = x.std()
                        x = (x - x_mean) / x_std
                        normalized = True

                    if self._normalize == "instanceNormSlice":
                        x = (x - x.mean(dim=(-2, -1), keepdim=True)) / \
                            x.std(dim=(-2, -1), keepdim=True)
                        normalized = True

                    self.input.append(x)
                    self.target.append(y)

        self.input = torch.cat(self.input, dim=0).unsqueeze(1).float()
        self.target = torch.cat(self.target, dim=0).unsqueeze(1)

        self.target[self.target < 0] = 0

        if self._non_empty_target:
            non_empty_slices = self.target.sum((-1, -2, -3)) > 0
            self.input = self.input[non_empty_slices]
            self.target = self.target[non_empty_slices]

        if self._binary_target:
            self.target[self.target != 0] = 1

        if self._normalize == "setNormNew":
            mean = self.input.mean()
            std = self.input.std()
            self._mean = mean
            self._std = std
            print(
                f"calculated mean {mean} and std {std} for normalization")
            self.input = (self.input - mean) / std
            normalized = True
        if self._normalize == "setNormUseParams":
            mean = self._mean
            std = self._std
            print(
                f"using provided mean {mean} and std {std} for normalization")
            self.input = (self.input - mean) / std
            normalized = True
        if not normalized:
            print(
                f"No (valid) normalization chosen. Pls check your normalization parameter: {self._normalize}.")
        else:
            print(f"Normalized using {self._normalize}")

    def random_split(
        self,
        val_size: float = 0.2,
        test_size: float = None,
    ):
        class MNMv2Subset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]

            def __getitem__(self, idx):
                return {
                    "input": self.input[idx],
                    "target": self.target[idx],
                    "index": idx
                }

        torch.manual_seed(0)
        indices = torch.randperm(len(self.input)).tolist()

        if test_size is not None:

            test_split = int(test_size * len(self.input))
            val_split = int(val_size * len(self.input)) + test_split

            mnmv2_test = MNMv2Subset(
                input=self.input[indices[:test_split]],
                target=self.target[indices[:test_split]],
            )

            mnmv2_val = MNMv2Subset(
                input=self.input[indices[test_split:val_split]],
                target=self.target[indices[test_split:val_split]],
            )

            mnmv2_train = MNMv2Subset(
                input=self.input[indices[val_split:]],
                target=self.target[indices[val_split:]],
            )

            return mnmv2_train, mnmv2_val, mnmv2_test

        mnmv2_train = MNMv2Subset(
            input=self.input[indices[int(val_size * len(self.input)):]],
            target=self.target[indices[int(val_size * len(self.input)):]],
        )

        mnmv2_val = MNMv2Subset(
            input=self.input[indices[:int(val_size * len(self.input))]],
            target=self.target[indices[:int(val_size * len(self.input))]],
        )

        return mnmv2_train, mnmv2_val

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "index": idx
        }


class ACDCDataset(Dataset):
    """

    labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]


    """

    def __init__(
        self,
        data_dir,
        # groups should be a list of strings
        # a string should look like this "condition_set_ref"
        # condition should be: fog, night, rain, snow
        # set should be train or val
        # _ref is optional

        data_groups,
        binary_target: bool = False,
        normalize: bool = True,
        map_unlabled_class_to: int = 255,
        mean: list = [106.7274, 102.5255, 101.1493],  # 104.5298,
        std: list = [65.0944, 67.3873, 72.9185]  # 70.5013,
        # calculated mean tensor([102.0331,  98.0248,  95.8863]) and std tensor([67.8922, 70.1307, 75.8344]) for normalization (night, fog, rain train data)
        # calculated mean tensor([106.7274, 102.5255, 101.1493]) and std tensor([65.0944, 67.3873, 72.9185]) for normalization (night, fog, snow train data)
    ):

        self._data_groups = data_groups
        self._data_dir = Path(data_dir).resolve()
        self._binary_target = binary_target
        self._normalize = normalize
        self._mean = mean
        self._std = std
        self._map_unlabled_class_to = map_unlabled_class_to
        self._load_data()

    def _load_data(self):
        self.input = []
        self.target = []
        self.meta = []
        self.sample_path = []

        split_parts = [s.split("_") for s in self._data_groups]

        # dataset has no regular ascending labels
        original_labels = [0, 7, 8, 11, 12, 13, 17, 19,
                           20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        new_labels = list(range(len(original_labels)))  # 0 to 19

        # Create LUT (lookup table)
        max_label = max(original_labels)
        # Initialize with -1 for invalids
        lut = torch.full((max_label + 1,), -1, dtype=torch.int64)
        for old, new in zip(original_labels, new_labels):
            lut[old] = new

        def remap_labels(label_tensor):
            return torch.take(lut, label_tensor)

        for subset in split_parts:
            print("load subset", subset)

            if (len(subset) < 2) | (len(subset) > 3):
                raise AssertionError(
                    f"data group {subset} is not valid. Correct format is condition_set_ref where ref is optional.")

            if (subset[0] in ["night", "fog", "rain", "snow"]):
                condition = subset[0]
            else:
                raise AssertionError(
                    f"condition {subset[0]} is no correct condition. Choose night, fog, rain or snow.")

            if subset[1] in ["train", "val"]:
                set = subset[1]
            else:
                raise AssertionError(
                    f"{subset[1]} is not valid. Choose train or val. Test data labels are not provided.")
            if len(subset) > 2:
                if subset[2] == "ref":
                    ref = "_ref"
                else:
                    raise AssertionError(
                        f"{subset[2]} is not valid. Use 'ref' for reference labels or leave empty for condition.")
            else:
                ref = ""

            path_input = self._data_dir / \
                f"rgb_anon_trainvaltest/rgb_anon/{condition}/{set}{ref}"
            path_gt = self._data_dir / \
                f"gt_trainval{ref}/gt/{condition}/{set}{ref}"
            # print(case_path)
            dirs = os.listdir(path_input)

            # iterate over directories
            for dir in dirs:
                print("load dir", dir)
                samples = os.listdir(path_input / dir)
                print("samples len", len(samples))

            # iterate over samples
                for sample in samples:
                    sample_input_path = path_input / dir / f"{sample}"
                    im_id = sample.split("rgb")[0]
                    gt_input_path = str(
                        path_gt / dir / im_id) + f"gt{ref}_labelIds.png"
                    if (os.path.exists(gt_input_path) & os.path.exists(sample_input_path)):
                        img = read_image(sample_input_path)
                        self.input.append(img)
                        gt = read_image(gt_input_path).long()
                        gt = remap_labels(gt)

                        if self._map_unlabled_class_to is not None:
                            gt = gt - 1
                            gt[gt == -1] = self._map_unlabled_class_to
                        self.target.append(gt)
                        self.sample_path.append(
                            str(sample_input_path).split("rgb_anon/", 1)[1])

        self.input = torch.stack(self.input).float()
        self.target = torch.cat(self.target, dim=0).unsqueeze(1)

        if self._normalize:
            if self._mean is not None and self._std is not None:
                mean = torch.tensor(self._mean)
                std = torch.tensor(self._std)
                print(
                    f"using provided mean {mean} and std {std} for normalization")
            else:
                mean = self.input.mean(dim=(0, 2, 3))
                std = self.input.std(dim=(0, 2, 3))
                print(
                    f"calculated mean {mean} and std {std} for normalization")
            # Reshape for broadcasting
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
            self.input = (self.input - mean) / std

        print("Loaded ACDC dataset with ", self.input.shape[0], "samples.")

    def condition_split(
            self,
            val_size: float = 0.2):
        class ACDCSubset(Dataset):
            def __init__(
                self,
                input,
                target,
            ):
                self.input = input
                self.target = target

            def __len__(self):
                return self.input.shape[0]

            def __getitem__(self, idx):
                return {
                    "input": self.input[idx],
                    "target": self.target[idx],
                    "index": idx
                }

        condition_combinations = [("rain", True), ("rain", False),
                                  ("night", True), ("night", False),
                                  ("fog", True), ("fog", False),
                                  ("snow", True), ("snow", False)]

        random.seed(0)
        indices_train = []
        indices_val = []

        for combination in condition_combinations:
            print("combination", combination)
            condition_indices = [self.sample_path.index(
                path) for path in self.sample_path if combination[0] == path.split('/', 1)[0] and combination[1] == ("ref" in path)]
            print("condition indices", condition_indices)
            random.shuffle(condition_indices)
            indices_train.extend(
                condition_indices[int(val_size * len(condition_indices)):])
            indices_val.extend(condition_indices[:int(
                val_size * len(condition_indices))])

        print("indices train", indices_train)
        print("indices val", indices_val)

        mnmv2_train = ACDCSubset(
            input=self.input[indices_train],
            target=self.target[indices_train],
        )

        mnmv2_val = ACDCSubset(
            input=self.input[indices_val],
            target=self.target[indices_val],
        )

        return mnmv2_train, mnmv2_val

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, np.ndarray):
            return {
                "input": self.input[idx],
                "target": self.target[idx],
                "index": idx,
                "sample_path": [self.sample_path[i] for i in idx],
            }
        return {
            "input": self.input[idx],
            "target": self.target[idx],
            "index": idx,
            "sample_path": self.sample_path[idx],
        }
