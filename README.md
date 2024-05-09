[![License](https://img.shields.io/badge/license-red.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv.2312.06709-blue.svg)](https://arxiv.org/abs/2110.09983)
[![Paper](https://img.shields.io/badge/paper-MICCAI-AMAI-2022-green.svg)](https://link.springer.com/chapter/10.1007/978-3-031-17721-7_8)

# \[MICCAI-AMAI 2022\] ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks


Official PyTorch implementation of \[MICCAI 2023\] [**ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**](https://arxiv.org/abs/2110.09983).

\[[Paper](https://arxiv.org/abs/2110.09983)\]\[[BibTex](#citing-ECG-ATK-GAN)\]

<br clear="left"/>

---

## Abstract
Our proposed Conditional Generative Adversarial Network (GAN) represents a significant advancement in the field of arrhythmia detection from ECG signals, addressing critical issues of robustness and security. By integrating a novel class-weighted objective function for identifying adversarial perturbations and incorporating specialized blocks for discerning and combining out-of-distribution shifts in signals, our architecture stands as the first of its kind to withstand adversarial attacks while maintaining high accuracy in classification. Through extensive benchmarking against six different white and black-box attacks, our model showcases superior robustness compared to existing arrhythmia classification models. Tested on two publicly available ECG arrhythmia datasets, our model consistently demonstrates its ability to accurately classify various arrhythmia types, even under adversarial conditions. This groundbreaking approach not only enhances the reliability of automated arrhythmia detection systems but also addresses security concerns associated with the potential misuse of adversarial attacks in medical contexts.

<div align="left">
  <img src="Figure/Fig1(4).png" width="1000"/>
</div>

## Training

_Detail_Coming Soon_


## Results
### Visualize-Output:
Qualitative comparison of (×2) image reconstruction using different SR methods on AMD, PALM, G1020 and SANS dataset. The green rectangle is the zoomed-in region. The rows are for the AMD, PALM and SANS datasets. Whereas, the column is for each different models: SwinFSR, SwinIR, RCAN and ELAN.
<div align="left">
  <img src="Figure/Fig2(4).png" width="700"/>
</div>

### Model stats:
(a) and (b) Effects of the numbers of iRSTB Blocks on the PSNR and SSIM, and (c) and (d) the numbers of DCA Blocks on the PSNR and SSIM for *2 images.
<div align="left">
  <img src="Figure/Fig4(2).png" width="700"/>
</div>


### Clinical Assessment:
We carried out a diagnostic assessment with two expert ophthalmologists and test samples of 80 fundus images (20 fundus images per disease classes: AMD, Glaucoma, Pathological Myopia and SANS for both original x2 and x4 images, and super-resolution enhanced images). Half of the 20 fundus images were control patients without disease pathologies; the other half contained disease pathologies. The clinical experts were not provided any prior pathology information regarding the images. Each of the experts was given 10 images with equally distributed control and diseased images for each disease category.
<div align="left">
  <img src="Figure/clinicalAssessment.jpg" width="900"/>
</div>

## Citing SwinFSR

If you find this repository useful, please consider giving a star and citation:

#### MICCAI 2023 Reference:
```bibtex
@inproceedings{hossain2023revolutionizing,
  title={Revolutionizing space health (Swin-FSR): advancing super-resolution of fundus images for SANS visual assessment technology},
  author={Hossain, Khondker Fariha and Kamran, Sharif Amit and Ong, Joshua and Lee, Andrew G and Tavakkoli, Alireza},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={693--703},
  year={2023},
  organization={Springer}
}
```

#### ArXiv Reference:
```bibtex
@article{hossain2023revolutionizing,
  title={Revolutionizing Space Health (Swin-FSR): Advancing Super-Resolution of Fundus Images for SANS Visual Assessment Technology},
  author={Hossain, Khondker Fariha and Kamran, Sharif Amit and Ong, Joshua and Lee, Andrew G and Tavakkoli, Alireza},
  journal={arXiv preprint arXiv:2308.06332},
  year={2023}
}
```
