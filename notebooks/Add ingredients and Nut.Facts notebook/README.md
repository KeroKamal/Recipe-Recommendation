# Disease-Food-Nutrition Mapping Generator

This project generates a CSV file that maps real disease names to recommended best and worst foods and nutrition attributes. The recommendations are based on a food dataset with nutritional information and a mapping of nutrition facts to descriptive phrases.

## How It Works

1. **Load and Process the Food Dataset:**
   - The script reads a CSV file (`food_dataset_with_nutriition.csv`) containing meal data.
   - It converts the string representation of the "NER" column (which contains ingredient lists) into actual Python lists.
   - The total number of meals is printed.

2. **Load Real Disease Names:**
   - A second CSV file (`real_disease_names.csv`) is read to obtain a list of real disease names from the "Disease" column.
   - The number of diseases loaded is printed.

3. **Define Nutrition Mapping:**
   - A dictionary (`nutrition_mapping`) maps various nutrition facts (e.g., calories, protein) to tuples containing a "best" phrase and a "worst" phrase.
   - A list of nutrition fact names is extracted from the keys of this mapping.

4. **Generate Disease-to-Food and Nutrition Mapping:**
   - For each disease:
     - **Best Foods:**  
       A random meal is selected from the food dataset. A random sample of 2 or 3 ingredients is chosen from the meal's ingredient list.
     - **Worst Foods:**  
       A different meal is randomly selected (ensuring it differs from the best food meal), and a sample of 2 or 3 ingredients is chosen.
     - **Best Nutrition:**  
       A random sample of 2 to 3 nutrition facts is selected, and their "best" descriptive phrases are retrieved.
     - **Worst Nutrition:**  
       From the remaining nutrition facts (not selected for best nutrition), a random sample is taken, and their "worst" descriptive phrases are retrieved.
   - The best and worst foods, as well as the nutrition recommendations, are converted to string representations and stored in a mapping entry.

5. **Save the Merged Mapping to a CSV File:**
   - All mapping entries are compiled into a DataFrame and saved as `disease_food_nutrition_mapping.csv`.
   - The script prints a message indicating the successful creation of the CSV file along with the number of rows.

## Requirements

- Python 3.x
- Pandas
- ast (part of the Python Standard Library)
- random (part of the Python Standard Library)

## Setup and Usage

1. **Install Dependencies:**
   - Ensure you have Python 3.x installed.
   - Install Pandas if not already installed:
     ```bash
     pip install pandas
     ```

2. **Prepare the Datasets:**
   - Place `food_dataset_with_nutriition.csv` in the working directory. This file should contain a column "NER" with stringified lists of ingredients.
   - Place `real_disease_names.csv` in the working directory. This file should contain a column "Disease" with real disease names.

3. **Run the Script:**
   - Save the script to a file (e.g., `generate_mapping.py`).
   - Just run in case of notebook but adjust paths for your dataets..
   - Run the script:
     ```bash
     python generate_mapping.py
     ```

4. **Output:**
   - The script generates a CSV file named `disease_food_nutrition_mapping.csv` containing the mapping data.
   - The output CSV includes columns for "Disease", "Best_Foods", "Worst_Foods", "Best_Nutrition", and "Worst_Nutrition".

This tool is useful for generating automated dietary recommendations linked to various diseases based on nutritional properties.
