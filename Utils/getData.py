import os
import torch
import numpy as np
from os import listdir
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Normalisasi (mean/std ImageNet), sama dengan yang dipakai saat training Model 2 sebelumnya
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


class getImageLabel(Dataset):
    """
    Dataset gambar ngengat (50 kelas).
    folder  : './Dataset/train', './Dataset/valid', atau './Dataset/test'
    img_size: ukuran gambar setelah di-resize (img_size x img_size)
    augment : True hanya untuk data train (flip, rotasi, color jitter)

    Gambar tidak langsung dimuat semua ke RAM (dataset 7000+ gambar bisa > 2 GB),
    jadi yang disimpan hanya path + label one-hot, gambarnya dibaca di __getitem__.
    """

    def __init__(self, folder='./Dataset/train', img_size=128, augment=False):
        self.img_size = img_size
        self.classes = sorted(listdir(folder))
        self.dataset = []

        # inisialisasi one hot encoding untuk kelas
        to_one_hot = np.eye(len(self.classes))
        for i, moth in enumerate(self.classes):
            for image_name in sorted(listdir(os.path.join(folder, moth))):
                if image_name.lower().endswith(IMG_EXT):
                    self.dataset.append([os.path.join(folder, moth, image_name), to_one_hot[i]])

        # label dalam bentuk angka (dipakai untuk membagi fold secara stratified)
        self.labels = np.array([np.argmax(label) for _, label in self.dataset])

        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]) if augment else None

    def __getitem__(self, item):
        path, label = self.dataset[item]

        # baca gambar, resize, (augmentasi), normalisasi ke 0-1 lalu standarisasi mean/std
        image = Image.open(path).convert('RGB').resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.augment is not None:
            image = self.augment(image)
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - MEAN) / STD

        # bentuk feature: (H, W, C) -> nanti di-permute ke (C, H, W) di Train/Test
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)
