# Flood_Prediction_System_with_Voting_Classifier

# Import necessary libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

from sklearn.metrics import plot_confusion_matrix

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


# Load your data

df = pd.read_csv("FannoCreekDurham.1.csv")


# Drop rows with missing values

df.dropna(inplace=True)


# Preprocess the data

X = df[['Gage Height [ft]', 'Discharge [cfs]', 'Temperature [degrees C]', 'Turbidity [FNU]']]

y = df['Flooding']


# Select relevant features

selected_features = ['Gage Height [ft]', 'Discharge [cfs]', 'Temperature [degrees C]', 'Turbidity [FNU]']

selected_df = df[selected_features]


# Create a correlation matrix

correlation_matrix = selected_df.corr()


# Visualize the correlation matrix using a heatmap

plt.figure(figsize=(8, 6))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title("Correlation Matrix")

plt.show()


# Pairplot for selected features

sns.pairplot(selected_df)

plt.suptitle("Pairplot of Selected Features", y=1.02)

plt.show()


# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)


# Clean feature names

clean_feature_names = [col.replace("[", "_").replace("]", "_").replace("<", "_") for col in X.columns]

X_train.columns = clean_feature_names

X_test.columns = clean_feature_names


# Create individual classifiers

xgb_clf = XGBClassifier(

    n_estimators=50,       # Reduce the number of boosting rounds
    
    learning_rate=0.05,    # Lower the learning rate for slower convergence
    
    max_depth=2,           # Decrease the maximum depth of the decision tree
    
    min_child_weight=5,    # Increase the minimum sum of instance weight needed in a child
    
    gamma=0.5,             # Increase the minimum loss reduction required for a split
    
    subsample=0.6,         # Reduce the fraction of observations sampled for each boosting round
    
    colsample_bytree=0.6   # Reduce the fraction of features sampled for each boosting round

)


rf_clf = RandomForestClassifier(

    n_estimators=100,       # Number of trees in the forest
    
    max_depth=3,           # Decrease the maximum depth of the tree
    
    min_samples_split=15,  # Increase the minimum number of samples required to split an internal node
    
    min_samples_leaf=10    # Increase the minimum number of samples required to be at a leaf node

)


# Create a voting classifier

voting_clf = VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf)], voting='soft')



# Train individual classifiers

xgb_clf.fit(X_train, y_train)

rf_clf.fit(X_train, y_train)

voting_clf.fit(X_train, y_train)



# Make predictions

xgb_pred = xgb_clf.predict(X_test)

rf_pred = rf_clf.predict(X_test)

voting_pred = voting_clf.predict(X_test)



# Evaluate the models

print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("Voting Classifier Accuracy:", accuracy_score(y_test, voting_pred))



# Print classification report

print("\nClassification Report for XGBoost:\n", classification_report(y_test, xgb_pred))

print("\nClassification Report for Random Forest:\n", classification_report(y_test, rf_pred))

print("\nClassification Report for Voting Classifier:\n", classification_report(y_test, voting_pred))



# Plot confusion matrix

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))



# XGBoost Confusion Matrix

plot_confusion_matrix(xgb_clf, X_test, y_test, ax=axes[0], cmap='Blues', display_labels=['No Flood', 'Flood'])

axes[0].set_title('XGBoost Confusion Matrix')



# Random Forest Confusion Matrix

plot_confusion_matrix(rf_clf, X_test, y_test, ax=axes[1], cmap='Blues', display_labels=['No Flood', 'Flood'])

axes[1].set_title('Random Forest Confusion Matrix')



# Voting Classifier Confusion Matrix

plot_confusion_matrix(voting_clf, X_test, y_test, ax=axes[2], cmap='Blues', display_labels=['No Flood', 'Flood'])

axes[2].set_title('Voting Classifier Confusion Matrix')

plt.tight_layout()
plt.show()

