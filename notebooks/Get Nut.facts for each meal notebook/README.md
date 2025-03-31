# Food Dataset Nutrition Aggregator

This project aggregates nutrition data for ingredients listed in a recipe dataset using the USDA FoodData Central API. It prefetches nutrition information for each unique ingredient, aggregates selected nutrient values for each recipe, and outputs a final CSV file with the additional nutrition columns.

## How It Works

1. **USDA API Configuration:**
   - The script is configured with a USDA API key and the search endpoint from the USDA FoodData Central.
   - A mapping is defined to convert USDA nutrient names to desired output names (e.g., "Energy" to "calories").

2. **Fetching Nutrition Data:**
   - The function `fetch_nutrition_data` queries the USDA API for a given ingredient.
   - It uses retry logic with exponential backoff to handle rate limits and network issues.
   - Results are cached to avoid duplicate API calls.

3. **Extracting Unique Ingredients:**
   - The function `extract_unique_ingredients` parses the "NER" column from a CSV file (e.g., `filtered_recipes_data.csv`) and extracts a set of unique ingredients across all recipes.

4. **Prefetching Nutrition Data:**
   - The function `prefetch_ingredients` uses a thread pool to fetch nutrition data in parallel for all unique ingredients.
   - Fetched data is stored in a global cache (`nutrition_cache`).

5. **Aggregating Nutrition for Recipes:**
   - The function `aggregate_nutrition` processes each recipe's list of ingredients, retrieves cached nutrition data, and aggregates values for the selected nutrients.
   - The result is a dictionary of aggregated nutrient values for each recipe.

6. **Batch Processing:**
   - The function `process_in_batches` processes the dataset in batches (default 500 rows per batch) to conserve memory.
   - For each batch, it aggregates nutrition data for each recipe and expands the aggregated data into separate columns.
   - Intermediate results are saved to partial CSV files to safeguard against data loss.

7. **Final Output:**
   - The final DataFrame with aggregated nutrition facts is saved as `food_dataset_with_nutriition.csv`.
   - The script prints overall execution time and average processing time per row.

## Requirements

- Python 3.x
- Pandas
- requests
- json (Standard Library)
- time (Standard Library)
- concurrent.futures (Standard Library)

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install pandas requests

2. **Prepare Your Dataset:**

- Ensure you have a CSV file named filtered_recipes_data.csv with a column "NER" containing JSON-style lists of ingredients.

3. **Configure the USDA API Key:**

- Replace the API_KEY variable in the script with your USDA FoodData Central API key.

4. **Output:**

- The script outputs intermediate CSV files (e.g., food_dataset_with_nutrition_partial_500.csv) during processing.
- The final output is saved as food_dataset_with_nutriition.csv containing the original data along with the aggregated nutrition columns.
- Execution time and processing statistics are printed to the console.

## Conclusion

This tool automates the process of enhancing a recipe dataset with detailed nutrition facts, which can be used for nutritional analysis, recipe recommendations, or dietary planning.