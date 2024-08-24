# ISAR Image Registration using a CNN-Bi-LSTM Architecture in PyTorch
PyTorch implementation of this [paper](https://ieeexplore.ieee.org/document/9971386)

## Notes/Todo
- Currently processing 120x120 images. Original image sizes are around 250x170

## Installation/Running
Create conda environment:
```bash
conda env create -f environment.yml
```

Running gradio dashboard using pre-trained weights:
```bash
python dashboard.py
```

## Training
Extract data and place into a directory called `data/`. Modify `utils/constants.py` for a different location.
The data has already been labeled in `labels.csv`.

To train, run: 
```bash
python train.py --train --test <directory_to_store_weights>
```

Modify the weights directory in `dashboard.py` for visualization.

## Hyperparams and Performance
200 epochs, learning rate 0.0001, batch size 128:
- Sequence length 3: .7306
- Sequence length 5: Test acc .7753
- Sequence length 10: Test acc .7593


## Data Specifications
```
GEOEYE: Images #1-17630
--Range Resolution (y-axis): 0.0047 m
--Cross-Range Resolution (x-axis): 0.0047 m
SPASE: Images #17631-32840
-Images #17631-21509,#25083-32840
--Range Resolution (y-axis): 0.0062 m
--Cross-Range Resolution (x-axis): 0.0062 m
-Images #21510-25082
--Range Resolution (y-axis): 0.0057 m
--Cross-Range Resolution (x-axis): 0.0058 m

Images are shown in log-scale and are thresholded below at 60 dB below the peak.
```