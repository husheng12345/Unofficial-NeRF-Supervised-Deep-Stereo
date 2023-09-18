# Unofficial NeRF-Supervised Deep Stereo
This repository is an unofficial implementation of [NeRF-Supervised Deep Stereo (CVPR 2023)](https://arxiv.org/abs/2303.17603).<br/>

## Features
* Integration of [NeRF-Supervised Deep Stereo code snippets](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) for easy training and evaluation.
* Enabling co-training on the NeRF-Stereo dataset combined with other datasets (e.g. SceneFlow), an advancement absent from initial code snippets.

## Required Data
You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder. 
```Shell
├── datasets
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Monkaa
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── Driving
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── disparity
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── Middlebury
        ├── MiddEval3
    ├── NeRF-Stereo
        ├── training_set
            ├── 0000
            ├── 0001
            ...
            ├── 0269
        ├── trainingQ.txt
```
NeRF-Stereo `training_set` can be downloaded [here](https://amsacta.unibo.it/id/eprint/7218/), `trainingQ.txt` can be found [here](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo/blob/main/filenames/trainingQ.txt).<br/>
KITTI 2015 and Middlebury v3 are used for evaluation.

## Training
To train RAFT-Stereo on NeRF-Stereo dataset, run
```Shell
python train_stereo.py --batch_size 2 --train_datasets 3nerf --train_iters 22 --valid_iters 32 --n_downsample 2 --num_steps 200000
```
Hyperparameters are chosen following Section 4.1 of the paper. <br> To train models on multiple datasets, modify `train_datasets` argument, for example `--train_datasets 3nerf sceneflow`. Please note that the data augmentation parameters for NeRF-Stereo dataset are separate from those of other datasets.

## Test
To evaluate the [official pretrained weights](https://drive.google.com/file/d/1zAX2q1Tr9EOypXv5kwkI4a_YTravdtsS/view?usp=sharing), download it to the `models` folder. <br> For KITTI, run
```Shell
CUDA_VISIBLE_DEVICES=0 python test.py --datapath ./datasets --dataset kitti --version KITTI/training/ --model raft-stereo --loadmodel ./models/raftstereo-NS.tar --outdir ./test_output --occ
``` 
For Middlebury, run
```Shell
CUDA_VISIBLE_DEVICES=0 python test.py --datapath ./datasets --dataset middlebury --version Middlebury/MiddEval3/trainingF/ --model raft-stereo --loadmodel ./models/raftstereo-NS.tar --outdir ./test_output --occ
```

## Results

Performance on KITTI 2015 and Middlebury v3:

| Model                         | KITTI-15<br/>(>3px All)   | Midd-T Full<br/>(>2px All)    |
| :---                          | :---:                     | :---:                         |
| Official pretrained weights   | 5.41                      | 16.38                         |
| Trained with this repository  | 6.12                      | 20.94                         |

Despite our best efforts to replicate the experimental setup as delineated in the paper, there exists a discrepancy between the model trained with this repository and the official pretrained weights.

## Acknowledgements

This repository is mainly based on [NeRF-Supervised Deep Stereo code snippets](https://github.com/fabiotosi92/NeRF-Supervised-Deep-Stereo) and [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo). Appreciation also goes to [Fabio Tosi](https://github.com/fabiotosi92) for his valuable instructions.