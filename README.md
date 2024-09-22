# Zoning Segmentation and Denoising Pipeline

This project implements a zoning segmentation and denoising pipeline for processing images, specifically for zoning maps or related data. It uses **Mean Shift Clustering** for segmentation and **parallelized contour operations** to denoise zones within the image.

## Features
- **Mean Shift Clustering** to segment the image into different zones based on pixel color.
- **Parallel Processing** for efficient denoising of zone masks using multiple CPU cores.
- **Contour Detection** and **Polygon Intersection** to isolate valid zones.
- **Logging** of image processing time and details.

## Requirements

Before running the code, make sure the following dependencies are installed:

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Scikit-learn (`scikit-learn`)
- Shapely

You can install all dependencies using `pip`:

```bash
pip install opencv-python numpy scikit-learn shapely
```

## Code Structure

1. **MeanShift_Zoning_Segmenter**
    - Segments the image into zones using Mean Shift clustering and generates mask images for each cluster.
    - **Input**: Image and output directory for storing the masks.
    - **Output**: Paths to the mask images.

2. **retrieve_poly**
    - Validates and compares the polygons of the original contours and filled contours to check for intersections.
    - **Input**: Original contours and filled contours.
    - **Output**: Returns the original contour if an intersection is found, otherwise returns `None`.

3. **process_denoise_zone_mask_image**
    - Processes the image mask to denoise it by filling valid contours.
    - **Input**: Mask image path and output directory.
    - **Output**: Denoised mask saved in the output directory.

4. **process_ori_image**
    - Processes an entire original image: first by applying Mean Shift clustering to segment the zones and then by denoising each segmentation mask.
    - **Input**: Image path and output directory.
    - **Output**: Processed masks and log of processing details.

5. **process_images**
    - Iterates over all images in the input directory, calling `process_ori_image` for each image.
    - **Input**: Input and output directory paths.
    - **Output**: Segmented and denoised zone masks.

## Usage

1. Set the paths for the input and output directories in the script:

    ```python
    input_directory = "/path/to/your/input_images/"
    output_directory = "/path/to/output_results/"
    ```

2. Run the script to process all images in the `input_directory`:

    ```bash
    python zoning_segmenter.py
    ```

3. The output will be saved in the `output_directory`, with each mask saved as a separate image. A log of the processing times and details is saved in `im_process3.log`.

## Example

If the input directory contains a set of zoning map images, the script will:

1. Apply Mean Shift clustering to group pixels by their color.
2. Create a mask for each cluster.
3. Denoise the mask by validating the contours.
4. Save the denoised mask images to the output directory.

## Logging

All processed image details, including the resolution and processing time, are logged in the `im_process3.log` file. This file is automatically generated and updated as the script processes each image.

## License

This project is licensed under the MIT License.
