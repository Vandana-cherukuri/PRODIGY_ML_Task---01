# PRODIGY_ML_Task---01README for "aaat - Colab"
Overview

This project demonstrates the application of K-Means Clustering on a dataset to identify distinct customer segments. The dataset (mall.csv) contains information about customers, including their demographics and purchasing behavior. The project utilizes Python and various data visualization and machine learning libraries to preprocess data, explore patterns, and cluster the data.

Dataset

The dataset contains 200 rows and 5 columns:

CustomerID: Unique identifier for each customer.

Gender: Gender of the customer (Male/Female).

Age: Age of the customer.

Annual Income (k$): Annual income in thousands of dollars.

Spending Score (1-100): Score assigned based on customer behavior and spending nature.


Tools and Libraries Used

Pandas: Data manipulation and analysis.

NumPy: Numerical computing.

Matplotlib & Seaborn: Data visualization.

Plotly: Interactive 3D plots.

Scikit-learn: Machine learning library, particularly for K-Means Clustering.


Project Features

1. Data Preprocessing:

Imported and cleaned data.

Checked for null values and data types.

Performed descriptive analysis to understand the distribution of the data.



2. Exploratory Data Analysis (EDA):

Visualized distributions using distplots.

Explored relationships between variables using scatter plots and count plots.

Examined patterns between Age, Annual Income, and Spending Score, stratified by Gender.



3. K-Means Clustering:

Applied K-Means clustering to various subsets of the data:

Age vs Spending Score

Annual Income vs Spending Score

Age, Annual Income, and Spending Score combined.


Visualized clusters using 2D and 3D scatter plots.

Determined the optimal number of clusters using the elbow method.



4. Cluster Visualization:

Used 3D scatter plots to visualize clusters, leveraging Plotly for interactivity.

Highlighted centroids of clusters.




File Details

mall.csv: Input dataset for the project.

Python script includes:

Importing required libraries.

Exploratory Data Analysis (EDA).

Implementation of K-Means clustering.

Visualization of results.



Instructions for Use

1. Ensure you have Python and required libraries installed.


2. Place the mall.csv file in the working directory.


3. Run the script in a Python environment (e.g., Jupyter Notebook, Google Colab).



Dependencies

Install the following Python libraries:

pip install numpy pandas matplotlib seaborn plotly scikit-learn

Results and Insights

Identified distinct customer clusters based on age, income, and spending habits.

Observed patterns in customer segmentation that can guide targeted marketing strategies.


Future Scope

Incorporate additional datasets for better customer profiling.

Apply hierarchical clustering or other advanced clustering methods.

Extend the project to include predictive modeling based on clusters.


Contact

For questions or collaboration, feel free to reach out.

