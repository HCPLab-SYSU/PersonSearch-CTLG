# Contrastive Transformer Learning with Proximity Data Generation for Text-Based Person Search

Official code for "Contrastive Transformer Learning with Proximity Data Generation for Text-Based Person Search" <br />
IEEE Transactions on Circuits and Systems for Video Technology (T-CSVT), 2023. <br />
Hefeng Wu, Weifeng Chen, Zhibin Liu, Tianshui Chen, Zhiguang Chen, Liang Lin. <br />

## Pretrained Model
Put VIT `vit_base_patch16_384.pth` into `models/`.
Download from [Google Drive](https://drive.google.com/file/d/1b7p3SsG7L7LPFrChc1Jx8X94YnLIH00X/view?usp=sharing).

## Dataset
```
dataset/
└── CUHK-PEDES
    ├── annotations
    └── imgs
        ├── cam_a
        ├── cam_b
        ├── CUHK01
        ├── CUHK03
        ├── Market
        ├── test_query
        └── train_query
```

## Training + Testing 
```
bash run.sh
```

## Checkpoints
The trained checkpoint and log can be found here: [Google Drive](https://drive.google.com/drive/folders/17ny2ZPPRHiEoON6NxLK38AlgyDkfIuIH?usp=sharing).

## Citation
```
@article{Wu2023PersonSearch,
  title={Contrastive Transformer Learning with Proximity Data Generation for Text-Based Person Search},
  author={Hefeng Wu and Weifeng Chen and Zhibin Liu and Tianshui Chen and Zhiguang Chen and Liang Lin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023}
}
```
