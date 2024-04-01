# TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts

This repository contains code to illustrate test-time adaptation to single images and reproduce the results of the paper: TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts.

## Preparation
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
We provide the pretrained model here
Alternatively, you can pretrain your own model using distributed data parallel using:
```
python main_joint_train.py \
--dataset imagenet --noise-mode gaussian --noise-var 0.005 \ 
--gpu 0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
[your imagenet-mini folder with train.csv and val.csv]
```


### Test-Time Adaptation to Single Images
An example to apply our method on fastMRI with simulated noise on single GPU can be seen below. Test-time adaptation is evalauated on natural and synthetic noise with different noise levels. Test results are obtained by running the method on selected 10 images from each dataset. Test images can be found under 'test_time_training/testset/', and the pretrained model that is adapted during the test-time adaptation is 'test_time_training/model/0715_ttt_mim_unet_gn_0.005.pth.tar'. Run 'test_time_training/TTT_MIM_TestTimeTraining.ipynb' for an example of test-time adaptation to single image, and reproduction of the test results. 
```
python ttt_mim_online.py \
--dataset fastmri --noise-mode gaussian --noise-var 0.005\
--pretrained [path of pretrained model] \
--niters 8 --lr 1e-5 --mask-ratio 0.01 --mask-patch-size 1 --denoise-loss pd\
--gpu 0\
[fastMRI dataset folder]
```
## Parameters

### Hyperparameters

* --lr: Learning rate for the gradient updates during the Test-Time Training.
* --niters: Number of gradient update iterations during the Test-Time Training
* --mask-ratio: Mask ratio determines the fraction of the masked-off area.
* --mask-patch-size: Mask patch size determines the size of each square masked patch.

### Options

* --dataset: Selects the dataset among alternatives; 'sidd', 'polyu', 'fmdd', 'ct', 'fastmri', 'imagenet'
* --noise-mode: Selects the noise mode for simulated noise scenarios. Alternatives are 'gaussian', 'sp', 'poisson'. Not needed for natural noise case.
* --noise-var: Selects the noise variance for simulated noise. Not needed for natural noise case.
* --pretrained: Gives the path to the pretrained model that will be used during the Test-Time Training.
* --ImageNum: Selects the image to be illustrated among the different images in testset. If there are 10 images, it can be set to 1-10. Set to 0 for no illustration.
* --gpu 0: Selects the single gpu for applying method. Distributed data paralleling is not used in this case.
* --denoise-loss: Selects the denoising loss. Set 'pd' for the default version of our method.

## Reproduction of the results in Table 1

The test results can be reproduced by running 'test_time_training/TTT_MIM_TestTimeTraining.ipynb'. The exact parameters to get results in Table 1 are given as,

| | SIDD | PolyU | FMDD | CT | FastMRI | G0.01 | G0.02 | SP | Poisson |
|----------|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|:-----------:|
| iteration number | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| learning rate | 1e-4 | 5e-5 | 5e-6 | 1e-5 | 1e-4 | 1e-5 | 5e-5 | 5e-5 | 1e-6 |
| mask ratio | 0.3 | 0.4 | 0.5 | 0.4 | 0.01 | 0.4 | 0.4 | 0.4 | 0.1 |
| mask patch size| 14 | 4 | 4 | 4 | 1 | 4 | 4 | 4 | 4 |

### Test-Time Adaptation to Batches of Images
This section is for adapting the method to a batch of images instead of a single one.
Here is an example to apply our method on SIDD with distributed data parallel.
```
python ttt_mim.py \
--dataset sidd\
--pretrained [path of pretrained model] \
--nepochs 20 --lr 1e-4 --batch-adapt 20 --mask-ratio 0.3 --mask-patch-size 14\
--gpu 0 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
[SIDD dataset folder]
```
