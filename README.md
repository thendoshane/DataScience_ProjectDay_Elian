# Data Science Project Day - Elian
# Project Overview
This repository serves as the submission for the Data Science Project Day. It contains data analysis pipelines and a web application designed to derive insights from customer, health, and product datasets. The project is modularized into specific analytical questions, with a central application interface for visualization.

Student: Elian [Insert Surname]

Supervisor: [Insert Supervisor Name]

# Repository Structure
The project is organized into the following key components:

1. app.py: The main entry point for the Streamlit web application.

2. pages/: Directory containing additional pages for the multi-page Streamlit app structure.

3. customers.csv, health_data.csv, products.csv: The raw datasets used for analysis.

4. q1_functions.py - q4_functions.py: Python modules containing specific functions and logic for answering project Questions 1 through 4.

5. requirements.txt: A list of Python dependencies required to run the project.

6. ITPFA0-44-Project-Bedfordview-EDUV8558526/: Folder containing additional project resources or documentation.

# Installation & Setup
To set up the project locally, please follow these instructions:

1. Clone the Repository

- git clone https://github.com/thendoshane/Data_Science_Project_By_Elian.git
- cd DataScience_ProjectDay_Elian

2. Create a Virtual Environment (Recommended)

# Windows
- python -m venv venv
- venv\Scripts\activate

# macOS/Linux
- python3 -m venv venv
- source venv/bin/activate

3. Install Dependencies
- Install the required libraries using the provided requirements.txt file:

- pip install -r requirements.txt
# Usage
- Running the Web Application
- This project is built to be viewed as an interactive application. To launch it:


- streamlit run app.py
Note: This will open the application in your default web browser.

# Data Analysis Modules
The logic for the analysis is separated into individual function files (q1_functions.py, etc.). You can import these into other scripts or notebooks to inspect the raw analysis logic:

# Python

from q1_functions import *
# Example usage of functions defined in the file
# Features
- Multi-Dataset Integration: Combines insights from customers.csv, health_data.csv, and products.csv.

- Modular Codebase: Analysis logic is separated from the presentation layer, ensuring clean and maintainable code.

- Interactive Dashboard: A multi-page Streamlit app allows users to navigate through different sections of the analysis.

Acknowledgements
This project was developed by Elian under the supervision of [Insert Supervisor Name].
