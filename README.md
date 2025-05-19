# Conditioned LSTM Image Captioning Model

This project implements an image captioning model using a conditioned LSTM decoder and ResNet-based image encoder, trained on the Flickr8k dataset.

## Overview

This project follows a multi-stage pipeline:

1. **Image Encoding**: Extracts image features using a pre-trained ResNet-18.
2. **Text Preparation**: Tokenizes and structures captions from Flickr8k.
3. **LSTM Decoder**: Trains a language model on captions.
4. **Conditioned Decoder**: Feeds image encodings at each LSTM timestep.
5. **Beam Search**: Implements a beam search strategy for improved caption generation.


## Files

- `caption_model.ipynb`: Main development notebook
    - This Jupyter Notebook contains:
      - Explanatory Markdown cells with comments and instructions.
      - Visual outputs from training and evaluation (e.g. plots, printed examples).
      - A clean, linear presentation of the full training + evaluation workflow.
    - Use this if you want to understand, interactively explore, or reproduce the results.
- `caption_model.py`: Script version
  - This is an auto-exported version of the notebook using Jupyter’s "Export as Python" feature.
  - Contains the same code, but without the Markdown explanations or output.
- `requirements.txt`: Required packages (e.g., torch, torchvision, PIL).
- `data/`: Folder expected to contain the Flickr8k dataset.

## Dataset

To run this project, download the Flickr8k dataset:
- [Flickr8k Dataset (Google Drive Link)](https://drive.google.com/drive/folders/1sXWOLkmhpA1KFjVR0VjxGUtzAImIvU39?usp=sharing)

Set the `MY_DATA_DIR` variable in the notebook or script accordingly.

## Setup

```bash
pip install -r requirements.txt
```

## Usage
Run the notebook interactively:
```bash
jupyter notebook caption_model.ipynb
```
Or run the script (modify to fit command-line use):
```bash
python caption_model.py
```

## Notes
- This model requires access to a GPU.
- Some preprocessing (like image encoding) is done once and stored to disk to save time during training.

