# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import cv2

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor, TripletFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', dstype='frames_cleanpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa()
            self._add_driving()

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'FlyingThings3D')
        left_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/left/*.png')) )
        right_images = [ im.replace('left', 'right') for im in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        val_idxs = set(np.random.permutation(len(left_images))[:400])
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Monkaa')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = osp.join(self.root, 'Driving')
        left_images = sorted( glob(osp.join(root, self.dstype, '*/*/*/left/*.png')) )
        right_images = [ image_file.replace('left', 'right') for image_file in left_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')]*len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/SintelStereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/FallingThings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        with open(os.path.join(root, 'filenames.txt'), 'r') as f:
            filenames = sorted(f.read().splitlines())

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('left.jpg', 'right.jpg')) for e in filenames]
        disp_list = [osp.join(root, e.replace('left.jpg', 'left.depth.png')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='datasets', keywords=[]):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        with open(os.path.join(root, 'tartanair_filenames.txt'), 'r') as f:
            filenames = sorted(list(filter(lambda s: 'seasonsforest_winter/Easy' not in s, f.read().splitlines())))
            for kw in keywords:
                filenames = sorted(list(filter(lambda s: kw in s.lower(), filenames)))

        image1_list = [osp.join(root, e) for e in filenames]
        image2_list = [osp.join(root, e.replace('_left', '_right')) for e in filenames]
        disp_list = [osp.join(root, e.replace('image_left', 'depth_left').replace('left.png', 'left_depth.npy')) for e in filenames]

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/KITTI', image_set='training'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='datasets/Middlebury', split='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["F", "H", "Q", "2014"]
        if split == "2014": # datasets/Middlebury/2014/Pipes-perfect/im0.png
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E","L",""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        else:
            lines = list(map(osp.basename, glob(os.path.join(root, "MiddEval3/trainingF/*"))))
            lines = list(filter(lambda p: any(s in p.split('/') for s in Path(os.path.join(root, "MiddEval3/official_train.txt")).read_text().splitlines()), lines))
            image1_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im0.png') for name in lines])
            image2_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/im1.png') for name in lines])
            disp_list = sorted([os.path.join(root, "MiddEval3", f'training{split}', f'{name}/disp0GT.pfm') for name in lines])
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]


class NS_Data(data.Dataset):
    def __init__(self, datapath, training_file, conf_threshold=0.5, disp_threshold=512., aug_params=None, scale=1):
        self.augmentor = TripletFlowAugmentor(**aug_params)
        self.scale=scale
        self.disp_threshold = disp_threshold
        self.conf_threshold = conf_threshold
        self.image_list = []

        training_file = open(training_file, 'r')

        for line in training_file.readlines():
            left, center, right, disp, confidence = line.split()
            self.image_list += [[os.path.join(datapath,left), 
                                 os.path.join(datapath,center), 
                                 os.path.join(datapath,right), 
                                 os.path.join(datapath,disp),
                                 os.path.join(datapath,confidence)]]

    def __getitem__(self, index):

        data = {}

        index = index % len(self.image_list)

        data['im0'] = frame_utils.read_gen(self.image_list[index][0])
        data['im1'] = frame_utils.read_gen(self.image_list[index][1])
        data['im2'] = frame_utils.read_gen(self.image_list[index][2])
        data['disp'] = cv2.imread(self.image_list[index][3], -1) / 64.
        data['conf'] = cv2.imread(self.image_list[index][4], -1) / 65536.

        data['im0'] = np.array(data['im0']).astype(np.uint8)
        data['im1'] = np.array(data['im1']).astype(np.uint8)
        data['im2'] = np.array(data['im2']).astype(np.uint8)
        
        data['disp'] = np.squeeze(np.array(data['disp']).astype(np.float32))
        data['conf'] = np.squeeze(np.array(data['conf']).astype(np.float32))
        data['disp'] = data['disp'] * np.squeeze((data['conf'] > self.conf_threshold))
        data['disp'][np.isinf(data['disp'])] = 0  
        data['disp'][data['disp']> self.disp_threshold] = 0
        
        if self.scale != 1:
            h, w = data['im2'].shape[0]//self.scale, data['im2'].shape[1]//self.scale
            data['im0'] = cv2.resize(data['im0'], (w, h), interpolation=cv2.INTER_NEAREST)
            data['im1'] = cv2.resize(data['im1'], (w, h), interpolation=cv2.INTER_NEAREST)
            data['im2'] = cv2.resize(data['im2'], (w, h), interpolation=cv2.INTER_NEAREST)
            data['disp'] = cv2.resize(data['disp'], (w, h), interpolation=cv2.INTER_NEAREST)
            data['conf'] = cv2.resize(data['conf'], (w, h), interpolation=cv2.INTER_NEAREST)

        # grayscale images
        if len(data['im1'].shape) == 2:
            data['im0'] = np.tile(data['im0'][...,None], (1, 1, 3))
            data['im1'] = np.tile(data['im1'][...,None], (1, 1, 3))
            data['im2'] = np.tile(data['im2'][...,None], (1, 1, 3))
        else:
            data['im0'] = data['im0'][..., :3]
            data['im1'] = data['im1'][..., :3]
            data['im2'] = data['im2'][..., :3]

        augm_data = self.augmentor(data['im0'], data['im1'], data['im2'], data['disp'], data['conf'])

        for k in augm_data:
            if augm_data[k] is not None:
                if len(augm_data[k].shape) == 3:
                    augm_data[k] = torch.from_numpy(augm_data[k]).permute(2, 0, 1).float() 
                else:
                    augm_data[k] = torch.from_numpy(augm_data[k].copy()).float() 

        # augm_data['im0_aug'] is unused
        return self.image_list[index][0], augm_data['im1_aug'], augm_data['im2_aug'], -augm_data['disp'], augm_data['conf'], \
                    augm_data['im0'], augm_data['im1'], augm_data['im2']

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        return copy_of_self    

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def collate_fn(batch):
        '''This function is used to enable joint training with binocular and trinocular data.'''

        tri_idx = [i for i in range(len(batch)) if len(batch[i]) == 8]
        bi_idx = [i for i in range(len(batch)) if len(batch[i]) == 5]
        nt, nb = len(tri_idx), len(bi_idx)
        assert nt + nb == len(batch)

        data = {'im1_forward': None, 'im2_forward': None, 'bi': {}, 'tri': {}}
        if nb:
            _, im1_forward, im2_forward, flow, valid = zip(*[batch[i] for i in bi_idx])
            data['im1_forward'] = torch.stack(im1_forward, dim=0)
            data['im2_forward'] = torch.stack(im2_forward, dim=0)

            data['bi']['flow'] = torch.stack(flow, dim=0)
            data['bi']['valid'] = torch.stack(valid, dim=0)
        if nt:
            _, im1_forward, im2_forward, flow, conf, im0, im1, im2 =  zip(*[batch[i] for i in tri_idx])
            if data['im1_forward'] is None:
                data['im1_forward'] = torch.stack(im1_forward, dim=0)
                data['im2_forward'] = torch.stack(im2_forward, dim=0)
            else:
                data['im1_forward'] = torch.cat((data['im1_forward'], torch.stack(im1_forward, dim=0)), dim=0)
                data['im2_forward'] = torch.cat((data['im2_forward'], torch.stack(im2_forward, dim=0)), dim=0)
            data['tri']['flow'] = torch.stack(flow, dim=0)
            data['tri']['conf'] = torch.stack(conf, dim=0)
            data['tri']['im0'] = torch.stack(im0, dim=0)
            data['tri']['im1'] = torch.stack(im1, dim=0)
            data['tri']['im2'] = torch.stack(im2, dim=0)

        return data, nb, nt


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("middlebury_"):
            new_dataset = Middlebury(aug_params, split=dataset_name.replace('middlebury_',''))
        elif dataset_name == 'sceneflow':
            clean_dataset = SceneFlowDatasets(aug_params, dstype='frames_cleanpass')
            final_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            new_dataset = (clean_dataset*4) + (final_dataset*4)
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif 'kitti' in dataset_name:
            new_dataset = KITTI(aug_params, split=dataset_name)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif dataset_name == '3nerf':
            ns_aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.5, 'do_flip': True}
            new_dataset = NS_Data(
                args.datapath, args.training_file, conf_threshold=args.conf_threshold, disp_threshold=args.disp_threshold,
                aug_params=ns_aug_params
            )
            logging.info(f"Adding {len(new_dataset)} samples from 3nerf")
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, collate_fn=NS_Data.collate_fn, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

