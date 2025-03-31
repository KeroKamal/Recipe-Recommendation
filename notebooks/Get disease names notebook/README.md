# Disease Ontology CSV Generator

This script downloads the Disease Ontology OBO file from the OBO Foundry, extracts unique disease names from it, and saves them in a CSV file.

## How It Works

1. **Download OBO File:**
   - The script fetches the OBO file from the provided URL (`http://purl.obolibrary.org/obo/doid.obo`).

2. **Parse the OBO File:**
   - It uses the `obonet` library to read the OBO file into a graph structure.
  
3. **Extract Disease Names:**
   - Iterates through the nodes in the graph, selecting nodes that have a "name" attribute and a node ID that starts with "DOID:".
   - Collects all unique disease names.

4. **Create and Save CSV:**
   - Converts the list of disease names into a Pandas DataFrame.
   - Saves the DataFrame to a CSV file named `real_disease_names.csv`.

## Requirements

- Python 3.x
- obonet
- pandas

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install obonet pandas

2. **Output:**

- The script prints the number of unique disease names extracted.
- A CSV file named real_disease_names.csv is created containing the disease names.

## Conclusion

This tool is useful for creating a curated list of disease names for further use in data analysis, research, or integration with other systems.