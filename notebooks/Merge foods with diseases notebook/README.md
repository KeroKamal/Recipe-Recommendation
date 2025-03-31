# Food Dataset Merging Script

This script is intended to read two CSV files, restrict the merged data to the first 9882 rows, and save the result to a new CSV file. The CSV files used are:

- `food_dataset_with_nutriition.csv`
- `disease_food_nutrition_mapping.csv`

**Steps:**
1. Load the two CSV files into separate DataFrames.
2. Restrict the merged DataFrame to the first 9882 rows.
3. Save the final DataFrame as `food_dataset_with_nutriition Merged.csv`.

> **Note:** The variable `merged_df` is used to represent the merged DataFrame, but the merge operation is not explicitly defined in the code.
