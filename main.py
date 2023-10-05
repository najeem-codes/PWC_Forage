import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load data from file path
data = pd.read_csv('C:/Users/pc/OneDrive - FHNW/PWC_Forage/archive/bank-additional-full.csv', sep=';')

# Convert target variable to binary (1 for 'yes', 0 for 'no')
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), data.select_dtypes(include=['float64', 'int64']).columns),
        ('cat', OneHotEncoder(), data.select_dtypes(include=['object']).drop(columns=['y']).columns)
    ]
)

# Define pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['y']), data['y'], test_size=0.2, random_state=42
)

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
predictions = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

# Display results
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
