# PLISM-registration

`smartphone_wsi_registration.py` is a Python script for image registration between Whole Slide Images (WSI) and smartphone captured images.

## Installation

To install the necessary dependencies, run the following command:
`pip install -r requirements.txt`

## Usage

To execute the script, use the following command format;

Here is the explanation of the command line arguments:
- `--q` : Query image filename (e.g., `smartphone.png`)
- `--path` : Path to the Whole Slide Image (WSI) file (e.g., `/path/to/wsi.svs`)
- `--result_dir` : Directory to save the results (e.g., `/save_dir`)
- `--crop_size` : Crop size (default is `512`)
- `--window_size` : Window size for tiling (default is `4096`)
- `--save_folder_png` : Folder to save PNG patches (e.g., `/path/to/save_folder_png`)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
