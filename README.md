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


````

## Main features

The main application is implemented in `app.py` and includes:

- Gradio-based graphical interface for loading the SAM2 model and selecting ultrasound frames.
- Prompt-based segmentation initialization using a point or bounding box on the first frame.
- SAM2 video mask propagation for sequential ultrasound ablation monitoring.
- Optional loading of fine-tuned SAM2 prompt encoder and mask decoder weights.
- Mask thresholding using the selected or stored best threshold.
- Mask refinement using median filtering, contour smoothing, morphological closing, and largest connected-component selection.
- Export of predicted mask contours as JSON and CSV files.
- Export of scaled `x`, `y`, and fixed `z` coordinates for downstream robotic planning.
- Generation of focal-point CSV files from predicted masks when the focal-point helper script is available.
- Nearest-neighbour reordering of focal points for path execution.
- Motion-aware update logic for shifting or re-masking the current treatment plan based on measured motion.
- Segmentation metric calculation against ground-truth masks, including IoU, Dice, accuracy, precision, recall, IoU loss, and Dice loss.
- Overlay visualization of true positives, false positives, and false negatives.

## Relationship to the official SAM2 repository

This repository does not redistribute the official SAM2 source code. Users should install SAM2 from the official Meta/Facebook Research repository:

https://github.com/facebookresearch/sam2

The official SAM2 repository provides the model implementation, installation instructions, pretrained checkpoints, and citation information. This repository provides the application-level code used to apply SAM2 to ultrasound-guided HIFU ablation monitoring.

## Installation

First, create and activate a Python environment. Python 3.10 or newer is recommended.

Then install PyTorch and TorchVision according to your operating system and CUDA version. After that, install the official SAM2 package by following the instructions in the official SAM2 repository.

A typical installation workflow is:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e ".[notebooks]"
````

Then clone this repository and install the additional packages:

```bash
git clone https://github.com/Alaleh-Taghipour/sam2-us-hifu-ablation-monitoring.git
cd sam2-us-hifu-ablation-monitoring
pip install -r requirements.txt
```

## Checkpoints

This code uses the SAM2.1 large backbone in the default configuration:

```text
sam2.1_hiera_large.pt
configs/sam2.1/sam2.1_hiera_l.yaml
```

The official SAM2 checkpoint should be downloaded from the official SAM2 repository and placed in the local SAM2 checkpoint folder.

The fine-tuned ultrasound head weights used in this project are expected to be placed in the `weights/` folder or another user-defined local path. The default code expects a fine-tuned checkpoint with a corresponding metadata file:

```text
decoder_ultrasound.pt
decoder_ultrasound_meta.json
```

The metadata file may include the model configuration and the selected best threshold used for mask binarization.

## Data organization

Example input ultrasound frames should be placed in:

```text
data_sample/images/
```

Ground-truth masks, when available for metric calculation, should be placed in:

```text
data_sample/masks/
```

The current implementation expects a sequence of image frames in `.jpg`, `.jpeg`, or `.png` format. The frames are sorted numerically when possible.

## Running the application

After installing the dependencies and preparing the SAM2 checkpoint, run:

```bash
python app.py
```

The Gradio interface will open locally. In the interface, provide:

1. The path to the SAM2 checkpoint.
2. The path to the SAM2 model configuration file.
3. The path to the ultrasound image sequence.
4. Whether to use the base SAM2 weights or the fine-tuned head weights.

After initialization, click on the first frame to provide a segmentation prompt. Then click **Start Prediction** to propagate the mask through the image sequence.

## Outputs

The pipeline saves outputs to the configured prediction directory, including:

* Predicted mask contour JSON files.
* Predicted contour CSV files.
* Live treatment-plan CSV file.
* Focal-point CSV files, when focal-point generation is enabled.
* Nearest-neighbour ordered focal-point CSV files.
* Segmentation overlays.
* Ground-truth comparison overlays.
* Metric JSON files containing IoU, Dice, accuracy, precision, recall, IoU loss, and Dice loss.

## Motion-aware update logic

The GUI includes optional motion inputs:

```text
d, dx, dy
```

The update logic uses three motion-response levels:

* If the measured motion is small, the current boundary and focal-point plan are shifted by the measured displacement.
* If the measured motion is moderate, the first frame is re-masked and the live plan is refreshed.
* If the measured motion is large, the same re-masking pathway is used as a conservative re-planning fallback.

This logic was included to support motion-aware updates during ultrasound-guided HIFU monitoring.

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{taghipour2026hifu_sam2_monitoring,
  title   = {Simulation-Informed Ultrasound-Guided HIFU Planning and Ablation Monitoring with Robotic Focal Targeting},
  author  = {Taghipour, Alaleh and Lari, Salman and Rajabzadeh, Hossein and Han, Jeong-woo and Kwon, Hyock Ju},
  journal = {Ultrasonics},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please also cite the official SAM2 paper:

```bibtex
@article{ravi2024sam2,
  title   = {SAM 2: Segment Anything in Images and Videos},
  author  = {Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal = {arXiv preprint arXiv:2408.00714},
  url     = {https://arxiv.org/abs/2408.00714},
  year    = {2024}
}
```

## License

Please check the license of the official SAM2 repository before redistributing any SAM2-related files. This repository only provides the application-level code developed for ultrasound ablation monitoring.

A license for this repository should be selected after confirming with all co-authors and the supervising research group.

## Contact

For questions about this repository or the associated ultrasound-guided HIFU monitoring workflow, please contact the corresponding author listed in the manuscript.

