import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder

# 1. Load and Clean Data
# The SMS dataset often has encoding issues; 'latin-1' is the standard fix.
df = pd.read_csv('data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Encode labels: ham = 0, spam = 1
le = LabelEncoder()
y = le.fit_transform(df['label'])

# 2. TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['message'])
total_features = X.shape[1]

print(f"--- Feature Extraction ---")
print(f"Total number of features created: {total_features}")
print("-" * 30)

# 3. Lasso Regression Analysis Function
def analyze_lasso(alpha_val, X_data, y_data, total_feat_count):
    # Lasso is typically for regression, but we use it here 
    # as a feature selector for the binary targets.
    lasso = Lasso(alpha=alpha_val)
    lasso.fit(X_data, y_data)
    
    non_zero = np.sum(lasso.coef_ != 0)
    eliminated = total_feat_count - non_zero
    reduction_pct = (eliminated / total_feat_count) * 100
    
    return non_zero, eliminated, reduction_pct

# 4. Compare Different Alpha Values
alphas = [0.1, 0.01, 1.0]
results = []

print(f"{'Alpha':<10} | {'Non-Zero':<10} | {'Eliminated':<10} | {'% Reduction':<15}")
print("-" * 55)

for a in alphas:
    nz, elim, red = analyze_lasso(a, X, y, total_features)
    print(f"{a:<10} | {nz:<10} | {elim:<10} | {red:<14.2f}%")