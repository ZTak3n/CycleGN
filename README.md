
### Overview
This repository contains the official code for CycleGN, as presented in the paper: [Variational Inference for Cyclic Learning (ICLR 2026)](https://openreview.net/forum?id=c1jWNZ1Zqg).

CycleGN provides an alternative to CycleGAN that does not rely on adversarial networks.
This repository is primarily intended to illustrate the differences between training with the EM method (CycleGN) and training with the single-step method (CycleGAN).
To facilitate a fair comparison, only minimal changes have been made to the official CycleGAN codebase to derive CycleGN.

### Clone repositories:
```
git clone https://github.com/ZTak3n/CycleGN
cd CycleGN

git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix

cp ../cycle_gn_model.py models/
cp ../environment.yml ./
sed -i 's/, weights_only=True//g' models/base_model.py 
```

### Prepare the environment:
Set up the environment according to CycleGAN's requirements and activate it.
```
conda env create -f environment.yml
conda activate pytorch-CycleGAN-and-pix2pix
```

### Prepare the dataset:
The cityscape dataset can be downloaded from https://cityscapes-dataset.com.
 Please download the datasets [gtFine_trainvaltest.zip] and [leftImg8bit_trainvaltest.zip] and unzip them.
 Then run:
```
python datasets/prepare_cityscapes_dataset.py --gtFine_dir ./gtFine/ --leftImg8bit_dir ./leftImg8bit --output_dir ./datasets/cityscapes/
```

### Train:

```
python train.py  --dataroot ./datasets/cityscapes  --name cityscapes_cyclegn  --model cycle_gn  --cycle_step 200  --no_html  --lr 0.002 --n_epochs_decay 0 --lambda_identity 0 --checkpoints_dir cycleGN_ckpt
```

### Test:
```
python test.py --dataroot ./datasets/cityscapes  --name cityscapes_cyclegn --model cycle_gn  --num_test 500 --checkpoints_dir cycleGN_ckpt --direction BtoA
python test.py --dataroot ./datasets/cityscapes  --name cityscapes_cyclegn --model cycle_gn  --num_test 500 --checkpoints_dir cycleGN_ckpt --direction AtoB
```
More quantitative evaluation tools can be found in [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).


### 💡Training Tips:

1. Adjust your learning rate and cycle steps to ensure convergence in both forward and backward processes.

2. Neural networks are generally more sensitive to noisy labels than to noisy inputs. Therefore, during training, maintain a cyclic backward phase for the network you are training. Specifically, when training `netA`, use:
   ```python
   loss = idt_loss(netA(netB(x).detach()), x)
   ```
   instead of:
   ```python
   loss = idt_loss(frozen_netB(netA(y)), y)
   ```
3. Pay attention to the bijective assumption. In reality, most generation tasks do not satisfy the bijective assumption. Many-to-many relationships increase the difficulty of learning. For example, mapping from a horse to a zebra can result in multiple possible stripe patterns, which may break cycle consistency.
   
---

If you find this work helpful, please use the following BibTeX code to cite the paper.
```
@inproceedings{
zou2026variational,
title={Variational Inference for Cyclic Learning},
author={Zhuojun Zou and Jie Hao},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=c1jWNZ1Zqg}
}
```
