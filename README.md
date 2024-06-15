### Image-Watermark-Removal

This repository contains a Python script (`remove_watermark.py`) that removes watermarks from images using image processing techniques with OpenCV and NumPy libraries.

### Features

- **Watermark Removal**: Uses keypoint matching and homography to identify and inpaint watermarked areas.
- **Adjustable Threshold**: Allows customization of the threshold for matching keypoint descriptors.
- **Visualization**: Displays the original image and the image with the watermark removed.

### Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)

### Usage

To remove watermarks from an image, run the script with the following command:

```bash
python remove_watermark.py
```

### Example

Ensure that your script loads images from the specified paths (`input_images/water.jpg` and `input_images/Rattan Garden Lounge.png`) and performs watermark removal as demonstrated in the script.

### Notes

- Make sure to have the required libraries installed (`opencv-python`, `numpy`) before running the script.
- Adjust paths and filenames according to your specific file locations and naming conventions.
- Experiment with the threshold for matching keypoints (`good_matches = matches[:int(len(matches) * 0.15)]`) to optimize watermark removal based on your images.
