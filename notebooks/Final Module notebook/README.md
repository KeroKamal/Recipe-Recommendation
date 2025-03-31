# Recipe Recommendation and Nutrition Analyzer

This project is an intelligent recipe recommendation system that uses natural language processing (NLP) and nutritional filtering to provide meal suggestions based on user input. It reads a recipe dataset and a disease-to-food mapping file to filter and rank recipes according to included/excluded ingredients and nutritional conditions, while also taking into account dietary recommendations for specific diseases.

## Overview

The code integrates several NLP models and libraries to:
- **Process user queries:** Summarizes and cleans long inputs, corrects spelling (while preserving domain-specific terms), and tokenizes the query.
- **Extract and classify ingredients:** Identifies ingredients to include or exclude using question-answering and zero-shot classification pipelines.
- **Analyze nutrition conditions:** Extracts nutritional conditions (e.g., “low calories,” “high fat”) from the query and applies corresponding numeric thresholds.
- **Provide disease-specific recommendations:** Matches disease names from the query against a disease-food mapping CSV file to suggest best/worst foods and nutrition conditions.
- **Filter and rank recipes:** Filters a recipe dataset based on the identified include/exclude ingredients and nutritional conditions, scoring recipes by how many desired ingredients they contain, and providing fallback options if no exact match is found.
- **Display recipe details:** Prints recipe details and, if available, displays an image for each recommended meal.

## Features

- **NLP Pipelines:** Utilizes the Hugging Face Transformers library for question-answering, zero-shot classification, and summarization.
- **Spelling Correction:** Uses TextBlob for correcting general spelling errors while preserving key nutritional and domain-specific terms.
- **Ingredient Extraction & Correction:** Employs regular expressions and fuzzy matching (using `difflib.get_close_matches`) to normalize ingredient names and update a misspellings dictionary.
- **Nutrition Filtering:** Filters recipes based on nutritional values such as calories, fats, carbohydrates, protein, etc., using preset thresholds and user-specified conditions.
- **Disease Recommendations:** Reads a CSV mapping diseases to food recommendations to further guide recipe suggestions.
- **Dynamic Fallback:** If no recipes match all the criteria, the code implements a fallback mechanism by relaxing ingredient requirements or randomly sampling recipes.
- **Recipe Display:** Shows recipe details (including a nutritional breakdown) and displays an associated image if available.

## Setup

### Prerequisites

- **Python 3.6+**
- **Pandas:** For handling CSV data.
- **NLTK:** For tokenization (ensure to download required data such as `punkt`).
- **TextBlob:** For spelling correction.
- **Transformers:** For leveraging pretrained NLP models.
- **Other dependencies:** `re`, `json`, `os`, `difflib`, `ast`, and IPython for image display.

### Installation

1. **Clone the repository or copy the script.**

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate

3. **Install required packages:**

- pip install pandas nltk textblob transformers ipython

4. **Download Necessary NLTK Data:**

- The code automatically downloads the punkt tokenizer data using nltk.download('punkt').

5. **Prepare Data Files:**

- Recipe Dataset: Place the food_dataset_with_nutriition.csv file in the specified directory.
- Disease Mapping CSV: Place the disease_food_nutrition_mapping.csv file in the corresponding folder.
- JSON Files: The code uses common_misspellings.json and common_ingredients.json (if not found, they will be created).

6. **Images:**

- Ensure that the recipe images are stored in the designated folder (e.g., E:\Project\AI\Nutrition\data\recipe_images).

## Acknowledgments

- Hugging Face Transformers: For providing easy-to-use NLP pipelines.
- NLTK & TextBlob: For text processing and spelling correction functionalities.
- Pandas: For robust CSV data manipulation.