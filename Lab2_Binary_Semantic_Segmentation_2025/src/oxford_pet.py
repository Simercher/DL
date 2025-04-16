import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import random
from tqdm import tqdm
import cv2
from torchvision import transforms
from torchvision.transforms import functional as F

# def resize_all_fields(sample):
#     sample["image"] = cv2.resize(sample["image"], (384, 384), interpolation=cv2.INTER_LINEAR)
#     sample["mask"] = cv2.resize(sample["mask"], (192, 192), interpolation=cv2.INTER_NEAREST)
#     sample["trimap"] = cv2.resize(sample["trimap"], (192, 192), interpolation=cv2.INTER_NEAREST)
#     return sample

# transform = transforms.Compose([
#     # transforms.HorizontalFlip(p=0.5),
#     # A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=15, p=0.5),
#     # A.RandomBrightnessContrast(p=0.3),
#     transforms.Normalize(mean=(0.48011755, 0.44633889, 0.3943648),
#                 std=(0.22629277, 0.22372609, 0.2258771)),
#     # ToTensorV2()
# ])#, additional_targets={"mask": "mask", "trimap": "mask"}, is_check_shapes=False)

def compute_dataset_mean_std(dataset):
    """
    dataset: 必須是回傳 image 為 numpy array 或 tensor 的 dataset
    """
    mean = 0.0
    std = 0.0
    n_samples = 0

    for sample in tqdm(dataset, desc="Computing mean/std"):
        image = sample["image"]  # (H, W, C)
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if image.shape[0] == 3:  # (C, H, W)，要轉回 (H, W, C)
            image = np.moveaxis(image, 0, -1)

        image = image / 255.0  # normalize 到 0~1

        mean += image.mean(axis=(0, 1))
        std += image.std(axis=(0, 1))
        n_samples += 1

    mean /= n_samples
    std /= n_samples

    print(f"Dataset mean  = {mean}")
    print(f"Dataset std   = {std}")
    return mean, std

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", model_type='unet'):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.model_type = model_type
        self.apply_aug_prob = 0.7
        self.crop_size = 384

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)
        # resize images
        image = np.array(Image.fromarray(image).resize((640, 640), Image.BILINEAR))
        mask = np.array(Image.fromarray(mask).resize((640, 640), Image.NEAREST))
        trimap = np.array(Image.fromarray(trimap).resize((192, 192), Image.NEAREST))
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        trimap = transforms.ToTensor()(trimap)
        # resize_image = transforms.Resize((384, 384))
        resize_mask = None
        if random.random() < self.apply_aug_prob:
            # 同步翻轉
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # 同步旋轉（±30度內）
            if random.random() < 0.5:
                angle = random.uniform(-30, 30)
                image = F.rotate(image, angle)
                mask = F.rotate(mask, angle)

            # 同步裁切
            if random.random() < 0.5:
                i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.crop_size, self.crop_size))
                image = F.crop(image, i, j, h, w)
                mask = F.crop(mask, i, j, h, w)
            else:
                # 沒套用增強，直接中心裁切統一尺寸
                image = F.center_crop(image, (self.crop_size, self.crop_size))
                mask = F.center_crop(mask, (self.crop_size, self.crop_size))

            # 只對 image 做對比度調整
            if random.random() < 0.5:
                contrast_factor = random.uniform(0.8, 1.5)
                image = F.adjust_contrast(image, contrast_factor)
        else:
            # 沒套用增強，直接中心裁切統一尺寸
            image = F.center_crop(image, (self.crop_size, self.crop_size))
            mask = F.center_crop(mask, (self.crop_size, self.crop_size))
        # if self.model_type != 'unet':
        resize_mask = transforms.Resize((192, 192), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        # else:
            # resize_mask = mask
        image = transforms.Normalize(mean=(0.48011755, 0.44633889, 0.3943648),std=(0.22629277, 0.22372609, 0.2258771))(image)
        # print(image.shape)
        # print(resize_mask.shape)
        # mask = transforms.Normalize(mean=(0.48011755, 0.44633889, 0.3943648),std=(0.22629277, 0.22372609, 0.2258771))(mask)

        sample = dict(image=image, mask=resize_mask, trimap=trimap)
        

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    # def _read_split(self):
    #     print("Reading")
    #     split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
    #     split_filepath = os.path.join(self.root, "annotations", split_filename)
    #     with open(split_filepath) as f:
    #         split_data = f.read().strip("\n").split("\n")
    #     filenames = [x.split(" ")[0] for x in split_data]
    #     if self.mode == "train":  # 90% for train
    #         filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
    #     elif self.mode == "valid":  # 10% for validation
    #         filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        
        # return filenames

    def _read_split(self):
        # 讀取 trainval + test 全部資料
        trainval_path = os.path.join(self.root, "annotations", "trainval.txt")
        test_path = os.path.join(self.root, "annotations", "test.txt")

        with open(trainval_path) as f:
            trainval_data = f.read().strip().split("\n")
        with open(test_path) as f:
            test_data = f.read().strip().split("\n")

        all_data = trainval_data + test_data
        filenames = [x.split(" ")[0] for x in all_data]
        random.shuffle(filenames)

        # 為了讓切分固定不亂，每次排序
        # filenames = sorted(filenames)

        total = len(filenames)
        train_end = int(total * 0.8)
        valid_end = int(total * 0.9)

        if self.mode == "train":
            print("Training")
            return filenames[:train_end]
        elif self.mode == "valid":
            print("Validating")
            return filenames[train_end:valid_end]
        elif self.mode == "test":
            print("Testing")
            return filenames[valid_end:]


    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # # resize images
        # image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        # mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        # trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # # convert to other format HWC -> CHW
        # sample["image"] = np.moveaxis(image, -1, 0)
        # sample["mask"] = np.expand_dims(mask, 0)
        # sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, model_type):
    # implement the load dataset function here
    dataset = SimpleOxfordPetDataset(data_path, mode, model_type)
    return dataset

    assert False, "Not implemented yet!"



if __name__ == "__main__":
    # raw_dataset = load_dataset("../dataset", mode="valid")
    # compute_dataset_mean_std(raw_dataset)
    root = "../dataset/oxford-iiil-pet"
    OxfordPetDataset.download(root)
    # train_dataset = SimpleOxfordPetDataset(root, mode="train")
    # print(len(train_dataset))
    # print(train_dataset[0])