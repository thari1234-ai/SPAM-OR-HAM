📧 SMS Spam Classification & Feature Selection
📌 Project Overview
As part of a Data Science initiative for an email service provider, this project focuses on classifying incoming SMS messages as Spam or Ham (not spam).

The primary challenge in text classification is the high dimensionality of the data—TF-IDF vectorization creates thousands of features (words). This project implements Lasso Regression (L1 Regularization) to perform automated feature selection, shrinking irrelevant word coefficients to zero and identifying the most critical "Spam" indicators.

🗂️ Dataset
The project uses the SMS Spam Collection Dataset from the UCI Machine Learning Repository.

Total Messages: 5,574

Columns: v1 (Label: Ham/Spam), v2 (Raw Text)

Source: Kaggle Dataset Link

🛠️ Technical Workflow
Data Cleaning: Handling encoding issues (Latin-1) and mapping labels to binary integers (0 for Ham, 1 for Spam).

Vectorization: Applying TfidfVectorizer to convert text into a numerical matrix, removing English stop words.

Lasso Regularization: Utilizing the L1 penalty to minimize the objective function:

w
min
​
 ( 
2n
1
​
 ∥y−Xw∥ 
2
2
​
 +α∥w∥ 
1
​
 )
Feature Impact Analysis: Comparing model sparsity across different α values (0.01,0.1,1.0).

📈 Results & Key Findings
Lasso Regression proves highly effective for text data. Below is the expected behavior of the model:

Alpha (α)	Feature Retention	Description
0.01	~5-10%	Keeps many features; identifies subtle spam patterns.
0.1	~0.5-1%	Sweet spot. Retains only "heavy hitter" words like free, claim, win.
1.0	< 0.1%	Extreme sparsity; likely eliminates almost all predictive features.
Percentage Reduction
By using α=0.1, the model typically achieves a feature reduction of over 99%, significantly simplifying the model without losing the ability to detect common spam triggers.

🧪 Requirements
pandas

numpy

scikit-learn

AUTHOR: Tharini P
