import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA

# Load the dataset
data = pd.read_csv("D:\\DCS\\SEM- 5\\ML\\Mini Project\\data.csv")
print(data.head())
print(data.describe())
print(data.info())

# Missing value checking
missing_val = data.isnull().sum()
print("Missing values:", missing_val)

# Numerical columns alone
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Outlier detection and removal
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Apply outlier removal function to the dataset
data_without_outliers = remove_outliers(data, numeric_columns)

# Display the updated dataset after outlier removal
print(data_without_outliers.head())

# ------------------------ SIMPLE LINEAR REGRESSION ------------------------

X = data[['radius_mean']]      # Independent variable (X)
y = data['area_mean']          # Dependent variable (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split the data into training and test sets 

model = LinearRegression()  # Model building
model.fit(X_train, y_train)
predictions = model.predict(X_test)  # Prediction test

# Printing the model evaluation metrics
print('Coefficients:', model.coef_) 
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f', mean_squared_error(y_test, predictions))
print('Coefficient of determination (R^2): %.2f', r2_score(y_test, predictions))

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression')
plt.xlabel('Radius Mean')
plt.ylabel('Area Mean')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------ LOGISTIC REGRESSION ------------------------

# Assuming 'diagnosis' is the target variable and other columns are predictors
X = data.drop(['id', 'diagnosis'], axis=1)  # Exclude 'id' column and 'diagnosis' as predictors
y = data['diagnosis']  # Target variable

# Split the data into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
predictions = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Confusion Matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# ------------------------ RANDOM FOREST ------------------------s
# Create a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100)  # You can adjust the number of estimators
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Model evaluation for Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# Feature importances plot for Random Forest
feature_importance = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importance)[::-1]
sorted_importance = feature_importance[sorted_indices]
sorted_columns = X.columns[sorted_indices]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importance, y=sorted_columns, palette='viridis')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Random Forest Feature Importances')
plt.show()

# Get the first tree from the Random Forest model
first_tree = rf_model.estimators_[0]  # Change the index to view different trees

plt.figure(figsize=(12, 8))
plot_tree(first_tree, feature_names=X.columns, filled=True, rounded=True)
plt.title('Random Forest Decision Tree Visualization')
plt.show()


# ------------------------ SVM ------------------------

# Create an SVM Classifier
svm_model = SVC(kernel='linear')  # You can choose different kernels like 'rbf', 'poly', etc.
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Model evaluation for SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))

#---- Accuracy comparison graph
models = ['SVM', 'Random Forest']
accuracies = [svm_accuracy, rf_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, accuracies, color=['orange', 'green'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0.85, 1.0)  # Adjust the y-axis limits for better visualization
plt.show()

# Data preprocessing for KNN
X_knn = data.drop(['id', 'diagnosis'], axis=1)  # Features for KNN
y_knn = data['diagnosis']  # Target variable for KNN

label_encoder = LabelEncoder()
y_knn = label_encoder.fit_transform(y_knn)

# Scale features for KNN
scaler_knn = StandardScaler()
X_knn_scaled = scaler_knn.fit_transform(X_knn)

# Perform PCA for KNN
pca_knn = PCA(n_components=2)
X_knn_pca = pca_knn.fit_transform(X_knn_scaled)

# Split data for KNN
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn_pca, y_knn, test_size=0.2, random_state=42)

# Initialize the KNN classifier
k = 5  # Number of neighbors (can be adjusted)
knn = KNeighborsClassifier(n_neighbors=k)

# Train the KNN model
knn.fit(X_train_knn, y_train_knn)

# Predictions using KNN on the test set
y_pred_knn = knn.predict(X_test_knn)

# Evaluate the KNN model
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
print(f"KNN Accuracy: {accuracy_knn:.2f}")

print("\nKNN Classification Report:")
print(classification_report(y_test_knn, y_pred_knn))

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test_knn, y_pred_knn))

# Plotting for KNN with decision boundary
plt.figure(figsize=(6, 4))

# Plotting decision boundary by creating a mesh grid
h = .02  # Step size in the mesh
x_min, x_max = X_knn_pca[:, 0].min() - 1, X_knn_pca[:, 0].max() + 1
y_min, y_max = X_knn_pca[:, 1].min() - 1, X_knn_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot also the test points
plt.scatter(X_test_knn[:, 0], X_test_knn[:, 1], c=y_test_knn, cmap=plt.cm.coolwarm)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('KNN Classification')
plt.show()

X = data.drop(['id', 'diagnosis'], axis=1)  # Features
y = data['diagnosis']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


# Initialize the ANN model
model = Sequential()

# Add input layer and hidden layers
model.add(Dense(units=32, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=16, activation='relu'))

# Add output layer
# Adjust units based on your target variable type (e.g., binary classification)
model.add(Dense(units=1, activation='sigmoid'))  # For binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()

