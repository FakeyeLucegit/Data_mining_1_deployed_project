Description
loading_page.py is a Streamlit application designed for data analysis and visualization. This script allows users to upload CSV files, perform data preprocessing, visualize data, apply clustering algorithms, and evaluate machine learning models.

Prerequisites
Before running this script, ensure you have the following installed:

Python 3.x
Required Python libraries (listed in requirements.txt)
You can install the dependencies using pip:


pip install -r requirements.txt
Or install individual libraries:


pip install streamlit numpy pandas scikit-learn matplotlib seaborn plotly
Usage
To run loading_page.py, use the following command in your terminal:


streamlit run loading_page.py
Features
1. Initial Data
Upload CSV files and display an overview of the data including the first and last rows, summary statistics, and missing values.
2. Data Preprocessing
Handle missing values with options to drop rows/columns, replace with mean, median, mode, or use KNN imputation.
3. Visualization
Univariate, bivariate, and multivariate visualizations using histograms, boxplots, and correlation heatmaps.
4. Clustering
Apply K-Means or DBSCAN clustering on selected features and visualize the clusters.
5. Learning Evaluation
Evaluate machine learning models (Decision Tree for classification and Linear Regression for regression) and display relevant metrics and visualizations.
6. Objective
Display the objective of the project and information about the authors.
Example
Here is an example of how to use this script in a real-world scenario:


import streamlit as st

# Run the Streamlit application
st.run('loading_page.py')
Contributions
Contributions are welcome! Please submit a pull request or open an issue to discuss your changes.

License
This project is licensed under the MIT License.

Author
Luce Fakeye and Oriane Assale, Master's students at Efrei Paris.

Github link: https://github.com/FakeyeLucegit/Data_mining_1_deployed_project.git
app deploy link: https://luceorianechocolate.streamlit.app/

