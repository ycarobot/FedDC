import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os


def dict_max(d):
    return np.max([np.max(v) for v in d.values()])


class create_digits_dataset(Dataset):
    def __init__(self, data_path, channels, is_test, data_holdout):
        self.client_sample_id_cur_dataset = {}
        for part in range(10):
            images, labels = np.load(os.path.join(data_path, 'partitions/train_ls_part{}.pkl'.format(part)),
                                     allow_pickle=True)
            if part == 0:
                self.images, self.labels = images, labels
            else:
                self.images = np.concatenate([self.images, images], axis=0)
                self.labels = np.concatenate([self.labels, labels], axis=0)
            if is_test:
                sids = np.array(list(range(len(self.images) - len(images), len(self.images))))
                self.client_sample_id_cur_dataset[part] = {"test": sids}
            else:
                sids = np.array(list(range(len(self.images) - len(images), len(self.images))))
                np.random.shuffle(sids)
                split_thresh = int((1 - data_holdout) * len(images))
                train_sids, test_sids = sids[:split_thresh], sids[split_thresh:]
                self.client_sample_id_cur_dataset[part] = {"train": train_sids, "test": test_sids}

        self.n_samples_cur_dataset = len(self.images)
        self.n_clients_cur_dataset = len(self.client_sample_id_cur_dataset)
        self.channels = channels
        if self.channels == 1:
            self.transform = transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        elif self.channels == 3:
            self.transform = transforms.Compose([
                transforms.Resize([28, 28]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            raise ValueError("{} channel is not allowed.".format(self.channels))

        # 使用 np.int64 替代 np.long
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 如果图像是展平的，依据 size 判断正确的重塑方式
        if image.ndim == 1:
            if image.size == 784 and self.channels == 1:
                # 28x28 灰度图
                image = image.reshape(28, 28)
            elif image.size == 3072 and self.channels == 3:
                # 32x32x3 彩色图
                image = image.reshape(32, 32, 3)
            elif image.size == 768:
                # 假设 768 元素为 16x16x3
                image = image.reshape(16, 16, 3)
            elif image.size == 2352:
                # 2352 元素，假设为 28x28x3 图像
                image = image.reshape(28, 28, 3)
            else:
                raise ValueError(f"Unexpected flattened image size: {image.size}")
        elif image.ndim == 2:
            # 2D 图像，例如灰度图
            if self.channels == 3:
                image = np.stack([image] * 3, axis=-1)  # 转换为彩色图
        elif image.ndim == 3:
            # 如果已有3D图像
            if self.channels == 1:
                # 如果数据本身为彩色但请求灰度，不做重塑，后续转换会处理
                pass

        # 转换为 PIL 图像
        image = Image.fromarray(image, mode='RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label