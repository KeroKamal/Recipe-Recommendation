# 🍽️ Recipe Recommendation API

This project is a Flask-based API for generating personalized recipe recommendations based on user queries. The system processes natural language input and applies various Natural Language Processing (NLP) techniques to extract ingredients, nutrition conditions, and disease-related recommendations. It then filters and ranks recipes from a dataset and returns a structured JSON response with detailed information about the recommended meal and additional options.

---

## 📌 Table of Contents

- [📖 Introduction](#-introduction)
- [✨ Features](#-features)
- [⚙️ Installation](#-installation)
- [🚀 Usage](#-usage)
- [📂 Project Structure](#-project-structure)
- [🛠 Technologies Used](#-technologies-used)
- [🔍 How It Works](#-how-it-works)
  - [📥 Data Loading and Preprocessing](#-data-loading-and-preprocessing)
  - [🧠 Natural Language Processing](#-natural-language-processing)
  - [🥗 Ingredient and Nutrition Extraction](#-ingredient-and-nutrition-extraction)
  - [⚕️ Disease Recommendations](#-disease-recommendations)
  - [📊 Recipe Filtering and Ranking](#-recipe-filtering-and-ranking)
  - [📌 NER Processing](#-ner-processing)
- [🔗 API Endpoints](#-api-endpoints)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 📖 Introduction

This project builds an intelligent recipe recommendation system that accepts plain text input through an API endpoint. It uses state-of-the-art NLP models (like transformers for question-answering, summarization, and zero-shot classification) to analyze the input, extract relevant information, and filter recipes from a large dataset. The API outputs recommendations in a clear and professional JSON format, listing a **Recommended Meal** along with several additional options.

---

## ✨ Features

✅ **Text Summarization:** Summarizes long input texts to focus on the most relevant parts.  
✅ **Spelling Correction:** Uses NLP (TextBlob) to correct spelling mistakes while preserving domain-specific terms.  
✅ **Ingredient Extraction:** Extracts and normalizes ingredients from both the user query and recipe datasets.  
✅ **Nutrition Analysis:** Identifies nutrition-related conditions (e.g., high/low calorie).  
✅ **Disease Recommendations:** Matches queries against disease-related recommendations and adjusts filtering criteria.  
✅ **Recipe Filtering and Ranking:** Filters recipes based on included/excluded ingredients and ranks them by relevance.  
✅ **NER Cleanup:** Processes NER fields from recipes to output a clean, comma-separated string.  
✅ **Flask API Endpoint:** Accepts plain text input and returns structured JSON with recipe recommendations.  

---

## ⚙️ Installation

1️⃣ **Clone the Repository:**  
```bash
 git clone https://github.com/KeroKamal/Recipe-Recommendation.git
 cd Recipe-Recommendation.git
```

2️⃣ **Create a Virtual Environment (Recommended):**  
```bash
 python -m venv env
 source env/bin/activate  # On Windows: env\Scripts\activate
```

3️⃣ **Install Dependencies:**  
```bash
 pip install -r requirements.txt
```

4️⃣ **Download Necessary NLTK Data:**  
The script automatically downloads the `punkt` tokenizer data using:
```python
 import nltk
 nltk.download('punkt')
```

---

## 🚀 Usage

1️⃣ **Run the Flask Application:**  
```bash
 python src/app.py
```

2️⃣ **Test the API Endpoint:**  
- Use Postman or `curl` to send a `POST` request to:  
  `http://127.0.0.1:5000/process`
- The request body should be plain text (not JSON). Example input:  
  **"I love milk"**

3️⃣ **Example Output Format:**  
```json
{
    "Recommended Meal": {
        "Meal name": "Hollywood Chicken",
        "NER": "onion, flour, milk, ground beef, potato chips, chicken",
        "Nutrition details": { ... }
    },
    "Option 1": { ... },
    "Option 2": { ... },
    "Option 3": { ... },
    "Option 4": { ... },
    "Option 5": { ... }
}
```

---

## 🛠 Technologies Used

🔹 Python 3  
🔹 Flask (for web API development)  
🔹 Pandas (for data manipulation and CSV handling)  
🔹 Transformers (Hugging Face models for NLP tasks)  
🔹 TextBlob (for spelling correction)  
🔹 NLTK (for natural language tokenization)  
🔹 Regular Expressions (for text cleaning and extraction)  
🔹 JSON (for configuration and output formatting)  

---

## 🔍 How It Works

### 📥 Data Loading and Preprocessing
- Loads recipe data and disease-food mappings from CSV files.
- Reads configurations from JSON files, creating defaults if necessary.
- Cleans data by converting nutrition columns to numeric types.

### 🧠 Natural Language Processing
- **Summarization:** Uses a pre-trained model to reduce long queries.
- **Spelling Correction:** Fixes typos while keeping domain-specific terms.
- **Tokenization:** Splits text into sentences and words using NLTK.

### 🥗 Ingredient and Nutrition Extraction
- Extracts ingredients from queries and normalizes them using fuzzy matching.
- Identifies nutrition-related terms (e.g., "low fat", "high protein").
- Uses zero-shot classification to determine inclusion/exclusion criteria.

### ⚕️ Disease Recommendations
- Matches queries against a disease mapping dataset.
- Adjusts recommendations based on best/worst foods for the condition.

### 📊 Recipe Filtering and Ranking
- Filters recipes based on ingredients and nutrition conditions.
- Implements a fallback strategy if no exact match is found.
- Ranks recipes by how well they match the input criteria.

### 📌 NER Processing
- Cleans up the NER (Named Entity Recognition) field in recipes.
- Outputs a structured, comma-separated list of ingredients.

---

## 🔗 API Endpoints

📌 **POST /process**  
Accepts plain text input and returns JSON-formatted recipe recommendations.

---

## 🤝 Contributing

We welcome contributions! To contribute:
1️⃣ Fork the repository.  
2️⃣ Create a feature branch (`git checkout -b feature-name`).  
3️⃣ Commit changes (`git commit -m "Added new feature"`).  
4️⃣ Push to the branch (`git push origin feature-name`).  
5️⃣ Open a Pull Request. 🎉

---

## 📜 License

This project is licensed under the MIT License. 📄

---

💡 **Enjoy using the Recipe Recommendation API!** 🎉
Happy coding! 😎🎉