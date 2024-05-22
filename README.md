# [IPCAI'2024] Surgical-DINO: Adapter Learning of Foundation Models for Depth Estimation in Endoscopic Surgery

![Image](https://github.com/BeileiCui/SurgicalDINO/blob/main/main_figure.jpg)

### [__[arxiv]__](https://arxiv.org/abs/2401.06013) [__[paper]__](https://links.springernature.com/f/a/Oq3zaVfPV2ar861oFWImTA~~/AABE5gA~/RgRnzg_NP0SiaHR0cHM6Ly9saW5rLnNwcmluZ2VyLmNvbS8xMC4xMDA3L3MxMTU0OC0wMjQtMDMwODMtNT91dG1fc291cmNlPXJjdF9jb25ncmF0ZW1haWx0JnV0bV9tZWRpdW09ZW1haWwmdXRtX2NhbXBhaWduPW9hXzIwMjQwMzA4JnV0bV9jb250ZW50PTEwLjEwMDcvczExNTQ4LTAyNC0wMzA4My01VwNzcGNCCmXqzYrrZQcskYRSGmJlaWxlaWN1aUBsaW5rLmN1aGsuZWR1LmhrWAQAAAct)

* 2024-05-14 Our new paper for adapting the foundation model for self-supervised depth estimation [__EndoDAC__](https://github.com/BeileiCui/EndoDAC) has been provisional accepted for MICCAI 2024!
## Introduction
This is the implementation for our __Surgical-DINO__. 

__Purpose:__ Depth estimation in robotic surgery is vital in 3D reconstruction, surgical navigation and augmented reality visualization. Although the foundation model exhibits outstanding performance in many vision tasks, including depth estimation (e.g., DINOv2), recent works observed its limitations in medical and surgical domain-specific applications. This work presents a low-ranked adaptation (LoRA) of the foundation model for surgical depth estimation.

__Methods:__ We design a foundation model-based depth estimation method, referred to as Surgical-DINO, a low-rank adaptation of the DINOv2 for depth estimation in endoscopic surgery. We build LoRA layers and integrate them into DINO to adapt with surgery-specific domain knowledge instead of conventional fine-tuning. During training, we freeze the DINO image encoder, which shows excellent visual representation capacity, and only optimize the LoRA layers and depth decoder to integrate features from the surgical scene. 

__Results:__ Our model is extensively validated on a MICCAI challenge dataset of SCARED, which is collected from da Vinci Xi endoscope surgery. We empirically show that Surgical-DINO significantly outperforms all the state-of-the-art models in endoscopic depth estimation tasks. The analysis with ablation studies has shown evidence of the remarkable effect of our LoRA layers and adaptation.

__Conclusion:__ Surgical-DINO shed some light on the successful adaptation of the foundation models into the surgical domain for depth estimation. There is clear evidence in the results that zero-shot prediction on pre-trained weights in computer vision datasets or naive fine-tuning is not sufficient to use the foundation model in the surgical domain directly.

## Notes
We are focusing on self-supervised learning now so we only provide the __Surgical-DINO model__ and __corresponding losses__ in ```./surgicaldino.py```. We provide an example on how to use it so you can easily adapt our model to your current or other baselines to start training. We also provide the results on [SCARED Dataset](https://endovissub2019-scared.grand-challenge.org/) with the splits in [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) below.

## Results

| Method | Abs Rel | Sq Rel | RMSE | RMSE log | &delta; |
|  :----:  | :----:  | :----:   |  :----:  | :----:  | :----:  |
| SfMLearner | 0.079 |	0.879 |	6.896 |	0.110 |	0.947 |
| Fang et al. | 0.078 |	0.794 |	6.794 |	0.109 |	0.946 |
| Defeat-Net | 0.077 |	0.792 |	6.688 |	0.108 |	0.941 |
| SC-SfMLeaner |0.068 |	0.645 |	5.988 |	0.097 |	0.957 |
| Monodepth2 | 0.071 |	0.590 |	5.606 |	0.094 |	0.953 |
| Endo-SfM | 0.062 |	0.606 |	5.726 |	0.093 |	0.957 |
| AF-SfMLeaner | 0.059 |	0.435 |	4.925 |	0.082 |	0.974 |
| DINOv2 (zero-shot) | 0.088 | 0.963 | 7.447 | 0.120 | 0.933 |
| DINOv2 (fine-tuned) | 0.060 | 0.459 | 4.692 | 0.081 | 0.963 |
|__Surgical-DINO SSL (Ours)__ |__0.059__ |	__0.427__ |	__4.904__ |	__0.081__ |	__0.974__ | 
|__Surgical-DINO Supervised (Ours)__ | __0.053__ | __0.377__ | __4.296__ | __0.074__ | __0.975__ |

Note: SSL refers to Self-Supervised Learning method.
## Initialization
You can use your own environment cause Surgical-DINO only requires torch and transformer. 

You can also build up the environment as instructed by [dinov2](https://github.com/facebookresearch/dinov2) and install transformer by ```pip install transformer```.

## Citation
If you found our paper our results helpful, please consider citing our paper as follows:
```
@article{beilei2024surgical,
  title={Surgical-DINO: Adapter Learning of Foundation Models for Depth Estimation in Endoscopic Surgery},
  author={Cui, Beilei and Islam, Mobarakol and Bai, Long and Ren, Hongliang},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2024}
}
```
