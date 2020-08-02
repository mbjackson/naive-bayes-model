# Import SciKit-Learn Modules
from sklearn import metrics
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# Load Wine Quality Dataset
wine = datasets.load_wine()

# Print Out Some Info to User
print("Dataset Features:", wine.feature_names)
print("Dataset Labels:", wine.target_names)
print("Dataset Size:", wine.data.shape)

# Split Data into Test and Train Sets
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=92)

# Initialize Naive Bayes Model
gnb = GaussianNB()

# Fit Model
gnb.fit(X_train, y_train)

# Make Predictions
y_pred = gnb.predict(X_test)

# Print Accuracy of Model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
