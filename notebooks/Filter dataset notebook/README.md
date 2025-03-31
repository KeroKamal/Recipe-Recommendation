# Recipe Data Filtering Script

This script processes a recipe dataset and filters it based on the availability of corresponding recipe images. It performs the following tasks:

1. **Load the Dataset:**
   - Reads the CSV file `recipes_data.csv` into a Pandas DataFrame.

2. **Remove Duplicates:**
   - Eliminates duplicate rows based on the `title` column.

3. **Select Relevant Columns:**
   - Keeps only the columns `title`, `NER`, and `ingredients`.

4. **Filter Based on Available Images:**
   - Retrieves a list of image files from the `recipe_images` folder.
   - Extracts the base names of the images (without file extensions).
   - Filters the DataFrame to include only rows where the `title` matches one of the image base names.

5. **Save the Filtered Data:**
   - Saves the filtered DataFrame to a new CSV file (`filtered_recipes_data2.csv`).

## Requirements

- Python 3.x
- Pandas

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install pandas

2. **Prepare Your Files:**

- Place recipes_data.csv in your working directory.
- Ensure that the folder recipe_images contains the recipe image files (with file names that correspond to recipe titles).

3. **Output:**

- The script generates a filtered CSV file named filtered_recipes_data2.csv that includes only the recipes for which corresponding images are available.

## Conclusion

This tool is useful for ensuring consistency between your dataset and available image resources.