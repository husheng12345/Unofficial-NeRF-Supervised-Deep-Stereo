
import numpy as np
import torch
import torch.utils.data as data
import cv2
from core.utils import frame_utils

import os.path as osp
from glob import glob


class Middlebury(data.Dataset):
    def __init__(self, datapath, version="training", occ=False, test=True):
        self.is_test = test
        self.disp_list = []
        self.image_list = []

        self.version = version
        self.occ = occ

        self.gt_name = "disp0" if "2021" in self.version else "disp0GT"
        self.mask_name = "mask0nocc"

        image_list = sorted(glob(osp.join(datapath, version, '*/im0.png')))
        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace('im0', 'im1')] ]

    def __getitem__(self, index):

        data = {}
        if self.is_test:
            im2_path, im3_path = self.image_list[index][1], self.image_list[index][2]
            data['im2'] = np.array(frame_utils.read_gen(im2_path), dtype=np.uint8)
            data['im3'] = np.array(frame_utils.read_gen(im3_path), dtype=np.uint8)

            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][..., None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][..., None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            gt_path = self.image_list[index][0].replace('im0.png', f"{self.gt_name}.pfm")
            data['gt'] = np.expand_dims(frame_utils.readPFM(gt_path), -1)
            data['validgt'] = data['gt'] < 5000

            if not self.occ:
                mask_path = self.image_list[index][0].replace('im0.png', f"{self.mask_name}.png")
                mask = np.expand_dims(cv2.imread(mask_path, -1), -1)
                mask = (mask == 255).astype(np.float32)
                data['gt'] *= mask

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()

            return data

    def __len__(self):
        return len(self.image_list)


class KITTI(data.Dataset):
    def __init__(self, datapath, version="KITTI/training/", occ=False, test=True):
        self.is_test = test
        self.version = version
        self.occ = occ

        self.disp_list = []
        self.image_list = []

        self.gt_name = "disp_occ_0" if self.occ else "disp_noc_0"
        self.im0 = "image_2"
        self.im1 = "image_3"

        image_list = sorted(glob(osp.join(datapath, version, self.im0, '*_10.png')))

        for i in range(len(image_list)):
            self.image_list += [ [image_list[i], image_list[i], image_list[i].replace(self.im0, self.im1)] ]

    def __getitem__(self, index):

        data = {}
        if self.is_test:

            im2_path, im3_path = self.image_list[index][1], self.image_list[index][2]
            data['im2'] = np.array(frame_utils.read_gen(im2_path), dtype=np.uint8)
            data['im3'] = np.array(frame_utils.read_gen(im3_path), dtype=np.uint8)


            if len(data['im2'].shape) == 2:
                data['im2'] = np.tile(data['im2'][..., None], (1, 1, 3))
            else:
                data['im2'] = data['im2'][..., :3]

            if len(data['im3'].shape) == 2:
                data['im3'] = np.tile(data['im3'][..., None], (1, 1, 3))
            else:
                data['im3'] = data['im3'][..., :3]

            data['gt'], data['validgt'] = [None]*2
            frm = self.image_list[index][0].split('.')[-1]
            data['gt'], data['validgt'] = frame_utils.readDispKITTI(self.image_list[index][0].replace(self.im0, self.gt_name))

            data['gt'] = np.array(data['gt']).astype(np.float32)
            data['validgt'] = np.array(data['validgt']).astype(np.float32)
            
            data['gt'] = torch.from_numpy(data['gt']).unsqueeze(-1).permute(2, 0, 1).float()
            data['validgt'] = torch.from_numpy(data['validgt']).unsqueeze(-1).permute(2, 0, 1).float()

            data['im2'] = torch.from_numpy(data['im2']).permute(2, 0, 1).float()
            data['im3'] = torch.from_numpy(data['im3']).permute(2, 0, 1).float()
            
            return data

    def __len__(self):
        return len(self.image_list)


def fetch_dataloader(args):

    if args.dataset == 'kitti':
        if args.test:
            dataset = KITTI(args.datapath, version=args.version, occ=args.occ, test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=4, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))

    elif args.dataset == 'middlebury':
        if args.test:
            dataset = Middlebury(args.datapath, version=args.version, occ=args.occ, test=True)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, persistent_workers=True,
                pin_memory=False, shuffle=False, num_workers=4, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))

    return loader

