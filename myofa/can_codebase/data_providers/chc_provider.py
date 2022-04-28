# dataset: Car_Hacking_Challenge_Dataset_rev20Mar2021
import warnings
import os
import math
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from myofa.can_codebase.data_providers.base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler
from myofa.can_codebase.data_providers.prepare_loader import prepare_data


class ChcDataProvider:

    def __init__(self, args=None, graphs=None, graphs_list=None, seed=1):

        warnings.filterwarnings('ignore')
        # self._save_path = save_path

        self.sample_trainx0, self.train, self.valid, self.test, max_num_nodes, input_dim = prepare_data(graphs, graphs_list, args, test_graphs=None, max_nodes=args.max_nodes, seed=seed)

        # self.image_size = image_size  # int or list of int
        # self.distort_color = distort_color
        # self.resize_scale = resize_scale
        #
        # self._valid_transform_dict = {}
        # if not isinstance(self.image_size, int):
        #     assert isinstance(self.image_size, list)
        #     from myofa.can_codebase.data_providers.my_data_loader import MyDataLoader
        #     self.image_size.sort()  # e.g., 160 -> 224
        #     MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
        #     MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)
        #
        #
        #     self.active_img_size = max(self.image_size)
        #     valid_transforms = self._valid_transform_dict[self.active_img_size]
        #     train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        # else:
        #     self.active_img_size = self.image_size
        #     train_loader_class = torch.utils.data.DataLoader
        #
        #
        # if valid_size is not None:
        #     if not isinstance(valid_size, int):
        #         assert isinstance(valid_size, float) and 0 < valid_size < 1
        #         valid_size = int(len(train_dataset.samples) * valid_size)
        #
        #     train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.samples), valid_size)
        #
        #     if num_replicas is not None:
        #         train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
        #         valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
        #     else:
        #         train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
        #         valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)
        #
        #     self.train = train_loader_class(
        #         train_dataset, batch_size=train_batch_size, sampler=train_sampler,
        #         num_workers=n_worker, pin_memory=True,
        #     )
        #     self.valid = torch.utils.data.DataLoader(
        #         valid_dataset, batch_size=test_batch_size, sampler=valid_sampler,
        #         num_workers=n_worker, pin_memory=True,
        #     )
        # else:
        #     if num_replicas is not None:
        #         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas, rank)
        #         self.train = train_loader_class(
        #             train_dataset, batch_size=train_batch_size, sampler=train_sampler,
        #             num_workers=n_worker, pin_memory=True
        #         )
        #     else:
        #         self.train = train_loader_class(
        #             train_dataset, batch_size=train_batch_size, shuffle=True,
        #             num_workers=n_worker, pin_memory=True,
        #         )
        #     self.valid = None
        #
        # test_dataset = self.test_dataset(valid_transforms)
        # if num_replicas is not None:
        #     test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas, rank)
        #     self.test = torch.utils.data.DataLoader(
        #         test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=n_worker, pin_memory=True,
        #     )
        # else:
        #     self.test = torch.utils.data.DataLoader(
        #         test_dataset, batch_size=test_batch_size, shuffle=True, num_workers=n_worker, pin_memory=True,
        #     )
        #
        # if self.valid is None:
        #     self.valid = self.test

    @staticmethod
    def name():
        return 'chc'  # CarHackingChallenge

    # @property
    # def data_shape(self):
    #     return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 2

    # @property
    # def save_path(self):
    #     if self._save_path is None:
    #         # self._save_path = '/dataset/imagenet'
    #         # self._save_path = '/usr/local/soft/temp-datastore/ILSVRC2012'  # servers
    #         self._save_path = '../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/'  # home server
    #
    #         if not os.path.exists(self._save_path):
    #             # self._save_path = os.path.expanduser('~/dataset/imagenet')
    #             # self._save_path = os.path.expanduser('/usr/local/soft/temp-datastore/ILSVRC2012')
    #             self._save_path = '../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/'  # home server
    #     return self._save_path

    # @property
    # def data_url(self):
    #     raise ValueError('unable to download %s' % self.name())

    # def train_dataset(self, _transforms):
    #     dataset = datasets.ImageFolder(self.train_path, _transforms)
    #     return dataset

    # def test_dataset(self, _transforms):
    #     dataset = datasets.ImageFolder(self.valid_path, _transforms)
    #     return dataset

    # @property
    # def train_path(self):
    #     return os.path.join(self.save_path, 'train')
    #
    # @property
    # def valid_path(self):
    #     return os.path.join(self.save_path, 'val')

    # @property
    # def normalize(self):
    #     return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])