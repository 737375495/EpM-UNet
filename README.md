# EpM-UNet
This is the official code repository for "EpM-UNet: A Mamba U-shaped network for the detection of Enteromorpha prolifera".

## Abstract
Enteromorpha prolifera is a rapidly spreading marine green algae that threatens coastal ecosystems. Traditional detection methods often fail under complex remote sensing conditions, while existing deep learning models struggle with small, blurred, and scattered patches, particularly under cloud cover or background noise. To address these challenges, we propose EpM-UNet, a multi-scale semantic segmentation network tailored for high spatial resolution remote sensing images. EpM-UNet integrates three key modules: the Backbone feature extraction block for combining local and global features, the Gradient and enhancement detection module for enhancing sensitivity to subtle targets using gradients and attention, and Adaptive state space scanning mechanism for direction-aware spatial modeling. Experiments on the FIO-EP dataset demonstrate that EpM-UNet achieves superior performance, with an IoU of 0.8519 and F1-score of 0.9201. Furthermore, by correlating E. prolifera distribution with environmental factors such as temperature and currents, the method improves ecological interpretability. Overall, EpM-UNet enhances detection accuracy and offers a practical method for remote sensing–based monitoring and early warning of green tide events.

## 0. Main Environments
```bash
conda create -n epmunet python=3.8
conda activate epmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

## 1. Prepare the dataset
### FIO-EP datasets
-The FIO-EP datasets can be found here {Enteromorpha Prolifera Detection in High-Resolution Remote Sensing Imagery Based on Boundary-Assisted Dual-Path Convolutional Neural Networks(https://ieeexplore.ieee.org/abstract/document/10291028)}
- After downloading the datasets, you are supposed to put them into './data/FIO-EP/'
- './data/FIO-EP/'
  - train
    - images
      - .tif
    - labels
      - .tif
  - val
    - images
      - .tif
    - labels
      - .tif

### Synapse datasets
- For the Synapse dataset, you could use dataprepare to make the dataset.
- './data/FIO-EP/'
  - data_test.npy
  - data_train.npy
  - data_val.npy
  - mask_test.npy
  - mask_train.npy
  - mask_val.npy
 
## 2. Train the EpM-UNet
```bash
cd EpM-UNet
python train.py  # Train and test EpM-UNet on the FIO-EP dataset.
```


## 3. Obtain the outputs
- After trianing, you could obtain the results in './results/'

## 4. Acknowledgments
- We thank the authors of [VM-UNet](https://github.com/JCruan519/VM-UNet) and [H-vmunet](https://github.com/wurenkai/H-vmunet) for their open-source codes.
