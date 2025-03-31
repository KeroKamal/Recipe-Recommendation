# CSV List String Formatter

This project processes a CSV file (`filtered_recipes_data.csv`) and updates specific columns by adding quotes around the items within list-like strings. The transformation is applied to columns such as "NER" and "ingredients" to ensure that each item in the list is properly quoted.

## How It Works

1. **Function Definition:**
   - The function `add_quotes_to_list_string` takes a string that represents a list without quotes on the items (e.g., `[chicken gravy, cream of mushroom soup, chicken, shredded cheese]`).
   - It removes any leading and trailing whitespace, checks if the string is enclosed in square brackets, splits the inner string by commas, and adds double quotes around each item.
   - The function then reconstructs the string with the quoted items.

2. **Load CSV Data:**
   - The script loads the CSV file `filtered_recipes_data.csv` into a Pandas DataFrame.

3. **Update Specific Columns:**
   - The script defines the columns to be processed (e.g., `NER` and `ingredients`).
   - For each specified column that exists in the DataFrame, the transformation function is applied to update the string format.

4. **Save Updated CSV:**
   - The modified DataFrame is saved to a new CSV file named `data_with_quotes.csv`.
   - A message is printed confirming the successful creation of the updated CSV.

## Requirements

- Python 3.x
- Pandas

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install pandas

2. **Prepare Your CSV File:**
   - Ensure your CSV file (filtered_recipes_data.csv) is in the working directory.
   - The CSV should contain the columns that require processing, such as NER and ingredients.

3. **Output:**
   - The script processes the specified columns in the CSV file, adding quotes around items in list strings.
   - An updated CSV file named data_with_quotes.csv is generated in the same directory.

## Conclusion

This tool is useful for standardizing the formatting of list-like strings in CSV files, ensuring consistency for further data processing or analysis.