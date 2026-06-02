# SAM2-Based Ultrasound Ablation Monitoring for Robotic HIFU Experiments

This repository provides the main graphical user interface and processing pipeline used for SAM2-based ultrasound ablation monitoring in simulation-informed robotic high-intensity focused ultrasound (HIFU) experiments.

The code was developed for sequential Verasonics B-mode ultrasound frames acquired during ex vivo HIFU ablation experiments. The pipeline uses a prompt-initialized SAM2 video predictor to segment the evolving ablation region, applies post-processing to refine the predicted mask, exports mask contours and focal-point CSV files, and computes segmentation metrics when ground-truth masks are available.

## Repository contents

```text
sam2-us-hifu-ablation-monitoring/
│
├── app.py
├── requirements.txt
├── README.md
├── weights/
│   └── README.md
├── checkpoints/
│   └── README.md
└── data_sample/
    ├── images/
    └── masks/
