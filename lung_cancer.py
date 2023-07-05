import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras


# Load the dataset
data = pd.read_csv('lung_cancer.csv')

gender_encoded = pd.get_dummies(data['GENDER'], prefix='GENDER')

# Drop the original 'GENDER' column from the data
data = data.drop('GENDER', axis=1)

# Concatenate the encoded gender columns to the data
data = pd.concat([data, gender_encoded], axis=1)

# Separate the features (X) from the target variable (y)
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the decision tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Evaluate the decision tree model
y_pred_tree = decision_tree.predict(X_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)
print("Decision Tree Accuracy:", accuracy_tree)

# Convert decision tree to a neural network
model = keras.models.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

# Evaluate the model
_, accuracy_nn = model.evaluate(X_test_scaled, y_test)
print("Neural Network Accuracy:", accuracy_nn)

# Make predictions using the neural network
y_pred_nn = model.predict_classes(X_test_scaled)