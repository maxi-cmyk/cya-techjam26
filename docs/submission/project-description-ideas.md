# Written Project Description Ideas

- Why we chose frozen CLIP over frequency-artifact detection
- Why we apply a matched JPEG re-encoding pass
- Why we chose RINE over UnivFD
- Assumption: bilinear interpolation is used for both the downsampling and upsampling steps of the resize round trip
- Why downsample-and-restore counts as one compound transform
- Why resize outputs are stored losslessly
- Why the model combines global context with pre-resize local crops
- Why interpolation artifacts are treated as an authentic false-positive risk
- Why matched JPEG normalization is offline rather than an inference step
