# ISAR Image Registration using a CNN-Bi-LSTM Architecture in PyTorch

## Notes/Todo
- Currently processing 120x120 images. Original image sizes are around 250x170


## Performance
- Seq length 5, Batch size 128, 200 epochs
    - Test accuracy: .9881


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