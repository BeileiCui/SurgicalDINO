# [IPCAI'2024] Surgical-DINO: Adapter Learning of Foundation Model for Depth Estimation in Endoscopic Surgery

![Image](https://github.com/BeileiCui/SurgicalDINO/blob/main/main_figure.jpg)

### [__[arxiv]__](https://arxiv.org/abs/2401.06013)
## Introduction
This is the implementation for our __Surgical-DINO__. 

__Purpose:__ Depth estimation in robotic surgery is vital in 3D reconstruction, surgical navigation and augmented reality visualization. Although the foundation model exhibits outstanding performance in many vision tasks, including depth estimation (e.g., DINOv2), recent works observed its limitations in medical and surgical domain-specific applications. This work presents a low-ranked adaptation (LoRA) of the foundation model for surgical depth estimation.

__Methods:__ We design a foundation model-based depth estimation method, referred to as Surgical-DINO, a low-rank adaptation of the DINOv2 for depth estimation in endoscopic surgery. We build LoRA layers and integrate them into DINO to adapt with surgery-specific domain knowledge instead of conventional fine-tuning. During training, we freeze the DINO image encoder, which shows excellent visual representation capacity, and only optimize the LoRA layers and depth decoder to integrate features from the surgical scene. 

__Results:__ Our model is extensively validated on a MICCAI challenge dataset of SCARED, which is collected from da Vinci Xi endoscope surgery. We empirically show that Surgical-DINO significantly outperforms all the state-of-the-art models in endoscopic depth estimation tasks. The analysis with ablation studies has shown evidence of the remarkable effect of our LoRA layers and adaptation.

__Conclusion:__ Surgical-DINO shed some light on the successful adaptation of the foundation models into the surgical domain for depth estimation. There is clear evidence in the results that zero-shot prediction on pre-trained weights in computer vision datasets or naive fine-tuning is not sufficient to use the foundation model in the surgical domain directly.

## Notes
We are focusing on self-supervised learning now so we only provide the Surgical-DINO model and corresponding losses here. You can easily adapt our model to your current or other baselines to start training. We also provide the results on [SCARED Dataset](https://endovissub2019-scared.grand-challenge.org/) with the splits in [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) shown below.

## Results

| Method | Abs Rel | Sq Rel | RMSE | RMSE log | &delta |
|  :----:  | :----:  | :----:   |  :----:  | :----:  | :----:  |
| SfMLearner | 0.079 |	0.879 |	6.896 |	0.110 |	0.947 |


## Initialization
