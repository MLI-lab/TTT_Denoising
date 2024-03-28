# TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts

This repository contains code to illustrate test-time adaptation to single images and reproduce the test results in Table 1 of the paper: TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts.

## Preperation
The code is written in python and heavily depends on Pytorch. It has been developed and tested with following packages which can be installed with: 
```
pip install requirements.txt
```
* scikit-image==0.19.2
* scikit-learn==1.0.2
* scipy==1.9.1
* torch==1.13.1
* torchvision==0.14.1

## Usage Modes
### Joint Training
Distributed data paralleling is adopted here.
```
python main_joint_train.py \
--dataset imagenet --noise-mode gaussian --noise-var 0.005 \ 
--gpu 0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
[your imagenet-mini folder with train.csv and val.csv]
```

### Test-Time Adaptation to Batches of Images
An example to apply our method on SIDD with distributed data paralleling.
```
python ttt_mim.py \
--dataset sidd\
--pretrained [path of pretrained model] \
--nepochs 20 --lr 1e-4 --batch-adapt 20 --mask-ratio 0.3 --mask-patch-size 14\
--gpu 0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
[SIDD dataset folder]
```

### Test-Time Adaptation to Single Images
An example to apply our method on fastMRI with simulated noise on single GPU can be seen below. Test-time adaptation is evalauated on natural and synthetic noise with different noise levels. Test results are obtained by running the method on selected 10 images from each dataset. Test images can be found under 'test_time_training/testset/', and the path to the pretrained model that is adapted during the test-time adaptation is under 'test_time_training/model/'.
```
python ttt_mim_online.py \
--dataset fastmri --noise-mode gaussian --noise-var 0.005\
--pretrained [path of pretrained model] \
--niters 8 --lr 1e-5 --mask-ratio 0.01 --mask-patch-size 1 --denoise-loss pd\
--gpu 0\
[fastMRI dataset folder]
```
