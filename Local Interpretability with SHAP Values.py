import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data
data = pd.read_csv('C:/Users/pc/OneDrive - FHNW/PWC_Forage/archive/bank-additional-full.csv', sep=';')

# Convert target variable to binary (1 for 'yes', 0 for 'no')
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=['y']), data['y'], test_size=0.2, random_state=42
)

# Define preprocessor
numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Fit the preprocessor and transform the training data
X_train_transformed = preprocessor.fit_transform(X_train)

# Get feature names
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
feature_names = numeric_features + cat_features.tolist()

# Initialize and train the classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_transformed, y_train)

# Extract feature importances
feature_importances = classifier.feature_importances_

# Check lengths of feature names and importances
print("Length of feature names: ", len(feature_names))
print("Length of feature importances: ", len(feature_importances))

# Create DataFrame to hold feature importances
feature_imp_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importances
}).sort_values(by='importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['feature'][:10], feature_imp_df['importance'][:10], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Important Features')
plt.gca().invert_yaxis()
plt.show()
