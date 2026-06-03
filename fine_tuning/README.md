# SAM2 Fine-Tuning Script

This folder contains the fine-tuning script used for ultrasound-domain adaptation of SAM2 for ablation monitoring.

The script fine-tunes SAM2 using paired ultrasound images and binary annotation masks. The image encoder/backbone is frozen, while prompt- and mask-related components are trained using simulated point prompts generated from the ground-truth masks.

## Script

```\
finetune_decoder.py
```

## Expected data structure

The script expects image and mask folders with matching relative paths and file names (but with all data):

```
data/
└── Annotations/
    ├── Images/
    │   └── image_seq_0/
    └── Annotations/
        └── image_seq_0/
```

## Main steps

The fine-tuning script performs the following steps:

* Loads paired ultrasound images and binary annotation masks.
* Applies resizing and light ultrasound-style augmentation.
* Simulates one positive and one negative point prompt for each mask.
* Builds the SAM2 video predictor.
* Freezes image encoder/backbone parameters.
* Selects prompt- and mask-related trainable parameters.
* Runs a one-frame SAM2 training pass using the video predictor interface.
* Optimizes a combined BCE and Dice loss.
* Tracks validation IoU during training.
* Saves the best checkpoint.

## Important notes

The script contains example local paths in the `if __name__ == "__main__":` section. These paths must be updated before running the script on another machine.

For reproducibility, the SAM2 configuration used for fine-tuning should match the SAM2 configuration used in the main monitoring application. If the monitoring application uses the SAM2.1 large model, the fine-tuning script should also use the corresponding SAM2.1 large configuration and checkpoint.

The full fine-tuned model weights are not included directly in this GitHub repository. They should be deposited in a persistent data/model repository if approved by the authors and research group.
