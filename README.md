# Introduction
************************************************************
**Glasssix-SphereFace2** project aims to train better feature extraction models or explore new hypersphere face recognition frameworks based on the **SphereFace2** framework. We made some modifications based on the opensphere source code to better adapt to existing environments and reduce device requirements. In this framework, the loss function and other parts (such as network structure, optimizer, data augmentation) are separated, which makes it easy to compare the effects of different loss functions.

### Supported Projects
* https://wyliu.com/papers/LiuCVPR17v3.pdf
* https://wyliu.com/papers/LiuNIPS18_MHE.pdf
* https://wyliu.com/papers/sphereface2_ICLR22.pdf
* https://github.com/ydwen/opensphere
* https://www.overleaf.com/8278323646wddwqvrmqngs

# Get Started
************************************************************
### Train a model
In this section, we will introduce how to train a model and where to view your training logs and models.

1.We train models on different datasets and backbone architectures through different yml files. Please refer to the comments in **_config/train/example.yml_** for the specific parameter annotations.
* To train SphereFace2 with UnicornNet256,run the following commend(with 4 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 train.py --config config/train/unicorn256_sphereface2.yml

```
* To train SphereFace with UnicornNet256,run the following commend(with 2 GPUs)
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=2 train.py --config config/train/unicorn256_sphereface.yml

```
2.We provide backbone architectures such as UnicornNet256 and UnicornNet512, but if you want to train other networks, you can add them yourself.

3.After finishing training a model, you will see a project folder under root. The trained model and log is saved in the folder named by the job starting time,eg,20230508_094803 for 09:48:03 on 2023-05-08.
### Test a model
During the testing phase, we provide code that can test multiple models and datasets simultaneously to facilitate comparisons between different models.
Additionally, we provide different code for conducting face recognition tests in a 1:1 or 1:N scenario.After finishing training a model, you will see a val folder under root.The test 
result is saved in the folder as one txt named by the job starting time.
* To test 1:1 ,simply run
```
CUDA_VISIBLE_DEVICES=0,1 python test.py --config config/val/val_unicorn256_1800.yml
```
* To test 1:N ,simply run
```
CUDA_VISIBLE_DEVICES=0,1 python test_1_n.py --config config/val/val_cc_10w_1_n.yml
```














