# Recipe Image Downloader

This project downloads images for various recipes based on dish names from a dataset. It uses the `icrawler` library to fetch images from Google Images and organizes them in a designated output folder.

## How It Works

1. **Load the Dataset:**
   - Reads a CSV file named `recipes_data.csv` that contains recipe data.
   - Extracts unique dish names from the column `title`.

2. **Setup Output Directory:**
   - Creates an output folder (`recipe_images`) where all downloaded images will be saved.
   - Builds a set of already downloaded dish names from the filenames in the output directory to avoid duplicate downloads.

3. **Image Download Process:**
   - Iterates over each dish name from the dataset.
   - Constructs a search query for Google Images by appending "recipe" to the dish name.
   - Checks if an image for the dish already exists; if so, it skips downloading.
   - Uses `GoogleImageCrawler` to download one image per dish into a temporary directory.
   - Renames and moves the downloaded image from the temporary directory to the main output directory. If a file with the same name exists, it appends a counter to the filename.

4. **Cleanup:**
   - After processing each dish, the script cleans up the temporary directory to prepare for the next download.

## Requirements

- Python 3.x
- Pandas
- icrawler

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install pandas icrawler

2. **Prepare Your Dataset:**

- Ensure the CSV file recipes_data.csv is in the working directory.
- The CSV should include a column named title with dish names.

3. **Output:**

- Downloaded images will be saved in the recipe_images folder.
- The script prints messages indicating whether an image was downloaded or skipped due to an existing file.

## Conclusion

This tool automates the process of gathering recipe images for further use, such as in culinary projects or recipe recommendation systems.