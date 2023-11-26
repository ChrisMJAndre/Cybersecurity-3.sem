# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 11:33:06 2023

@author: chris
"""
############################# PRE-PROCESSING #############################

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
dataset_path = 'dataset_full.csv'
df = pd.read_csv(dataset_path)

# Separating out the features and the target variable
X = df.drop('phishing', axis=1)  # Replace 'phishing' with your actual target column name
y = df['phishing']

# Remove constant features
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(X)
non_constant_columns = [column for column in X.columns
                        if column in X.columns[constant_filter.get_support()]]

X_filtered = constant_filter.transform(X)
 
# Convert filtered features back to a DataFrame
X_filtered_df = pd.DataFrame(X_filtered, columns=non_constant_columns)

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered_df)

# Balancing the dataset using SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X_scaled, y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=0)

# X_train, X_test, y_train, y_test are now ready for model training and evaluation



#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
############################# Logistic Regression ############################# - 95.75%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ... [previous code for loading, preprocessing, balancing, and splitting the dataset]

# Initialize the Logistic Regression classifier
log_reg = LogisticRegression()

# Train the classifier on the training data
log_reg.fit(X_train, y_train)

# Predict on the test data
y_pred = log_reg.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of Logistic Regression model: {accuracy * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the additional performance metrics
print(f"Precision of Logistic Regression model: {precision * 100:.2f}%")
print(f"Recall of Logistic Regression model: {recall * 100:.2f}%")
print(f"F1 Score of Logistic Regression model: {f1 * 100:.2f}%")


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
############################# Random Forest ############################# - 99.50%

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



# Initialize the Random Forest classifier
random_forest = RandomForestClassifier()

# Train the classifier on the training data
random_forest.fit(X_train, y_train)

# Predict on the test data
y_pred_rf = random_forest.predict(X_test)

# Calculate the accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"Accuracy of Random Forest model: {accuracy_rf * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the additional performance metrics
print(f"Precision of Logistic Regression model: {precision * 100:.2f}%")
print(f"Recall of Logistic Regression model: {recall * 100:.2f}%")
print(f"F1 Score of Logistic Regression model: {f1 * 100:.2f}%")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot the confusion matrix using Seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.title('Confusion Matrix for Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()




######################## PLOTS ###########################

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np

# Initialize the Random Forest classifier
random_forest = RandomForestClassifier()

# Train the classifier on the training data
random_forest.fit(X_train, y_train)

# Get the predicted probabilities for the test set
probabilities = random_forest.predict_proba(X_test)

# Extract probabilities for the positive class (phishing)
phishing_probabilities = probabilities[:, 1]

# Create a scatter plot of the predicted probabilities
plt.figure(figsize=(10, 6))
plt.scatter(range(len(phishing_probabilities)), phishing_probabilities, c=phishing_probabilities, cmap='viridis', alpha=0.6, edgecolors='w')
plt.title('Prediction Probabilities for Phishing Detection')
plt.xlabel('Instance number')
plt.ylabel('Predicted Probability of Phishing')
plt.colorbar(label='Probability')
plt.show()





import matplotlib.pyplot as plt
import pandas as pd


# Create a DataFrame with the predicted probabilities and the actual labels
plot_df = pd.DataFrame({
    'Predicted_Probability': probabilities[:, 1],  # Probability of being phishing
    'True_Label': y_test  # The true label
})

# Generate plot
plt.figure(figsize=(10, 6))
# For non-phishing instances (True_Label == 0)
plt.scatter(plot_df[plot_df['True_Label'] == 0].index, 
            plot_df[plot_df['True_Label'] == 0]['Predicted_Probability'], 
            label='Non-Phishing', alpha=0.5, cmap='viridis')

# For phishing instances (True_Label == 1)
plt.scatter(plot_df[plot_df['True_Label'] == 1].index, 
            plot_df[plot_df['True_Label'] == 1]['Predicted_Probability'], 
            label='Phishing', alpha=0.5, cmap='viridis')

plt.title('Prediction Probabilities for Phishing vs. Non-Phishing')
plt.xlabel('Instance number')
plt.ylabel('Predicted Probability of Phishing')
plt.legend()
plt.show()




#################### PROB #######################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Create a DataFrame with the actual labels and the predicted probabilities
plot_df = pd.DataFrame({
    'True_Label': y_test.reset_index(drop=True),  # Reset index to ensure proper alignment
    'Predicted_Probability': probabilities[:, 1]  # Probability of being phishing
})

# Generate the scatter plot with jitter
plt.figure(figsize=(10, 6))
# Define jitter amount
jitter_amount = 0.25
# Apply jitter and plot for each class
for label, grp in plot_df.groupby('True_Label'):
    # Adding jitter: a random number between -jitter_amount and jitter_amount
    jitter = np.random.uniform(-jitter_amount, jitter_amount, size=grp.shape[0])
    plt.scatter(grp['True_Label'] + jitter, grp['Predicted_Probability'], 
                c=grp['Predicted_Probability'],  # Color by probability
                cmap='viridis',  # Colormap from blue to green to yellow
                alpha=0.5, 
                edgecolor='w')

plt.colorbar(label='Predicted Probability')  # Show color scale
plt.title('Prediction Probabilities for Phishing vs. Non-Phishing with Jitter')
plt.xlabel('Actual Label (0=Non-Phishing, 1=Phishing)')
plt.ylabel('Predicted Probability of Phishing')
plt.xticks([0, 1])  # Show only 0 and 1 on the x-axis
plt.xlim(-0.5, 1.5)  # Adjust limits to include jitter
plt.legend()
plt.show()






#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
############################# K-Nearest Neighbors ############################# - 99.39%

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Initialize  K-Nearest Neighbors classifier
# n = 5, can be changed
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on the training data
knn.fit(X_train, y_train)

# Predict on the test data
y_pred_knn = knn.predict(X_test)

# Calculate the accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy of K-Nearest Neighbors model: {accuracy_knn * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the additional performance metrics
print(f"Precision of Logistic Regression model: {precision * 100:.2f}%")
print(f"Recall of Logistic Regression model: {recall * 100:.2f}%")
print(f"F1 Score of Logistic Regression model: {f1 * 100:.2f}%")


#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
############################# Support Vector Machines ############################# - 97.15%

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize the Support Vector Machine classifier
# The default kernel is 'rbf', can be changed to 'linear'
svm_model = SVC(kernel='rbf')

# Train the classifier on the training data
svm_model.fit(X_train, y_train)

# Predict on the test data
y_pred_svm = svm_model.predict(X_test)

# Calculate the accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print(f"Accuracy of SVM model: {accuracy_svm * 100:.2f}%")

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the additional performance metrics
print(f"Precision of Logistic Regression model: {precision * 100:.2f}%")
print(f"Recall of Logistic Regression model: {recall * 100:.2f}%")
print(f"F1 Score of Logistic Regression model: {f1 * 100:.2f}%")

#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
############################# Feature Importance - Random Forest ############################# 

import matplotlib.pyplot as plt
import numpy as np

# Get feature importances
feature_importances = random_forest.feature_importances_

# Sort the feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]

# Create a bar chart to display the importance of each feature in descending order
plt.figure(figsize=(12, 8))
plt.title('Feature Importances in Random Forest Classifier')
plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], color='b', align='center')
plt.yticks(range(len(sorted_indices)), [non_constant_columns[i] for i in sorted_indices])
plt.xlabel('Relative Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()


# TOP 10

# Get feature importances
feature_importances = random_forest.feature_importances_

# Sort the feature importances in descending order and select the top 10
sorted_indices = np.argsort(feature_importances)[::-1][:10]

# Create a bar chart to display the importance of the top 10 features
plt.figure(figsize=(12, 8))
plt.title('Top 10 Feature Importances in Random Forest Classifier')
plt.barh(range(len(sorted_indices)), feature_importances[sorted_indices], color='b', align='center')
plt.yticks(range(len(sorted_indices)), [non_constant_columns[i] for i in sorted_indices])
plt.xlabel('Relative Importance')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.show()




#------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------#
############################# Visualizations ############################# 




############################# T-SNE ############################# 

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Use t-SNE to project the dataset (which has been balanced and split) into 2 dimensions
tsne = TSNE(n_components=2, random_state=0)
X_reduced = tsne.fit_transform(X_train)  # Use X_train to fit the TSNE model

# Plot the reduced data
plt.figure(figsize=(12, 8))
plt.scatter(X_reduced[y_train == 0][:, 0], X_reduced[y_train == 0][:, 1], label='Non-Phishing', alpha=0.7)
plt.scatter(X_reduced[y_train == 1][:, 0], X_reduced[y_train == 1][:, 1], label='Phishing', alpha=0.7)
plt.legend()
plt.title('2D t-SNE visualization of Phishing vs. Non-Phishing')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()




############################# PCA 2 ############################# 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Use PCA to project the dataset (which has been balanced and split) into 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)  # Use X_train to fit the PCA model

# Plot the reduced data
plt.figure(figsize=(12, 8))
plt.scatter(X_reduced[y_train == 0][:, 0], X_reduced[y_train == 0][:, 1], label='Non-Phishing', alpha=0.7)
plt.scatter(X_reduced[y_train == 1][:, 0], X_reduced[y_train == 1][:, 1], label='Phishing', alpha=0.7)
plt.legend()
plt.title('2D PCA visualization of Phishing vs. Non-Phishing')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


############################# PCA 2 zoomed with xlim and ylim ############################# 
# Use PCA to project the dataset (which has been balanced and split) into 2 dimensions
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_train)  # Use X_train to fit the PCA model

# Plot the reduced data with adjusted point size and transparency
plt.figure(figsize=(12, 8))
plt.scatter(X_reduced[y_train == 0][:, 0], X_reduced[y_train == 0][:, 1], label='Non-Phishing', alpha=0.5, s=10)
plt.scatter(X_reduced[y_train == 1][:, 0], X_reduced[y_train == 1][:, 1], label='Phishing', alpha=0.5, s=10)
plt.legend()
plt.title('2D PCA visualization of Phishing vs. Non-Phishing')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.xlim([0, 30])
plt.ylim([-10, 20])

plt.show()



############################# PCA 3 ############################# 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Use PCA to project the dataset (which has been balanced and split) into 3 dimensions
pca = PCA(n_components=3)
X_reduced_3d = pca.fit_transform(X_train)  # Use X_train to fit the PCA model

# Plot the reduced data in 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for 3D visualization
ax.scatter(X_reduced_3d[y_train == 0][:, 0], X_reduced_3d[y_train == 0][:, 1], X_reduced_3d[y_train == 0][:, 2], label='Non-Phishing', alpha=0.7)
ax.scatter(X_reduced_3d[y_train == 1][:, 0], X_reduced_3d[y_train == 1][:, 1], X_reduced_3d[y_train == 1][:, 2], label='Phishing', alpha=0.7)

# Set rotation angle
ax.view_init(azim=-120, elev=10)

ax.set_title('3D PCA visualization of Phishing vs. Non-Phishing')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()

plt.show()



from sklearn.decomposition import PCA

# Use PCA to project the dataset (which has been balanced and split) into 3 dimensions
pca = PCA(n_components=3)
X_reduced_3d = pca.fit_transform(X_train)  # Use X_train to fit the PCA model

# Print the explained variance ratio for each component
explained_variances = pca.explained_variance_ratio_
print(f"Variance captured by each component: {explained_variances}")
print(f"Cumulative variance captured: {explained_variances.sum() * 100:.2f}%")















##################  RANDOM FOREST THRESHHOLD MODELLER #####################


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Initialize the Random Forest classifier
random_forest = RandomForestClassifier()

# Train the classifier on the training data
random_forest.fit(X_train, y_train)

# Predict probabilities on the test data
y_probs_rf = random_forest.predict_proba(X_test)

# Apply the threshold to the probabilities to create class predictions
y_pred_rf_threshold = (y_probs_rf[:, 1] >= 0.6).astype(int)

# Calculate the accuracy using the new predictions
accuracy_rf_threshold = accuracy_score(y_test, y_pred_rf_threshold)

print(f"Accuracy of Random Forest model with 0.6 threshold: {accuracy_rf_threshold * 100:.2f}%")

# Generate the confusion matrix using the new predictions
cm_threshold = confusion_matrix(y_test, y_pred_rf_threshold)

# Print the confusion matrix
print("Confusion Matrix with 0.6 threshold:")
print(cm_threshold)

# Plot the confusion matrix using Seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_threshold, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.title('Confusion Matrix for Random Forest Classifier with 0.6 threshold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()






from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Initialize the Random Forest classifier
random_forest = RandomForestClassifier()

# Train the classifier on the training data
random_forest.fit(X_train, y_train)

# Predict probabilities on the test data
y_probs_rf = random_forest.predict_proba(X_test)

# Apply the threshold to the probabilities to create class predictions
y_pred_rf_threshold = (y_probs_rf[:, 1] >= 0.4).astype(int)

# Calculate the accuracy using the new predictions
accuracy_rf_threshold = accuracy_score(y_test, y_pred_rf_threshold)

print(f"Accuracy of Random Forest model with 0.4 threshold: {accuracy_rf_threshold * 100:.2f}%")

# Generate the confusion matrix using the new predictions
cm_threshold = confusion_matrix(y_test, y_pred_rf_threshold)

# Print the confusion matrix
print("Confusion Matrix with 0.4 threshold:")
print(cm_threshold)

# Plot the confusion matrix using Seaborn's heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_threshold, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Phishing', 'Phishing'], yticklabels=['Non-Phishing', 'Phishing'])
plt.title('Confusion Matrix for Random Forest Classifier with 0.4 threshold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()













