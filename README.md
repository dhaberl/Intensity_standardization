# Intensity standardization
Scaling intensities of images to a common standard scale. The intensity standardization method by Nyul et al. should 
*theoretically* work with all types of images (not only medical images). This implementation was built for 
bone scintigraphy scans, which are basically grayscale images.

## How to use
Load your data and store it in a list:
```python
data = [img_1, img_2, ..., img_N]
```
where `data` is a list of your images as `numpy.ndarrays` (height, width).

Compute the standard scale for a given intensity-landmark configuration:
```python
standard_scale, percs = learn_standard_scale(data, i_min=1, i_max=99, i_s_min=1, i_s_max=100, l_percentile=10, u_percentile=90, step=10)
```
Apply the standard scale to a target image, i.e. normalize the image intensities of the target image to the standard 
scale:
```python
img_norm = apply_standard_scale(input_image, standard_scale, percs, interp_type='linear')

input_image (numpy.ndarray): Input image to normalize. Shape: (height, width)
```
Check the jupyter notebook for a detailed explanation on the fundamental problem and the usage: 
```python
demo.ipynb
```
Check the docstrings for a detailed explanation of the functions and its parameters.

## Acknowledgements
Implementation based on:
- https://github.com/sergivalverde/MRI_intensity_normalization
- https://github.com/jcreinhold/intensity-normalization
- https://gitlab.com/eferrante/nyul/-/tree/c01ad9afc89e9de18cd5ff911877bcbf49777476

It is suggested to use one of those packages for (brain) MRI scans, since they support NIfTI file format.

## References
[1] Nyúl, László G., and Jayaram K. Udupa. "On standardizing the MR image intensity scale" Magn Reson Med. (1999) 
Dec;42(6):1072-81.

[2] Shah, Mohak, et al. "Evaluating intensity normalization on MRIs of human brain with multiple sclerosis" Medical 
image analysis 15.2 (2011): 267-282.
