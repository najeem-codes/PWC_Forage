import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import lime
import lime.lime_tabular

# Load data
data = pd.read_csv('C:/Users/pc/OneDrive - FHNW/PWC_Forage/archive/bank-additional-full.csv', sep=';')

# Preprocess data
X = data.drop(columns=['y'])
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify feature types
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

# Define transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Bundle preprocessing and modeling code in a pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)
                          ])

# Preprocessing of training data, fit model
pipeline.fit(X_train, y_train)

# Manually preprocess the data for LIME explanations
X_train_transformed = preprocessor.transform(X_train)

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_transformed,
    feature_names=np.concatenate([numerical_cols, preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)]),
    class_names=['no', 'yes'],
    categorical_features=[False]*len(numerical_cols) + [True]*len(categorical_cols)*len(data[categorical_cols].nunique()),
    mode='classification'
)

# Define a predict_proba function that manually applies the model to preprocessed input data
def predict_proba_lime(data):
    # Ensure the input data is 2D
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    return pipeline.named_steps['model'].predict_proba(data)

# Explain instance #4 and save the explanation to an image file
instance_4 = preprocessor.transform(X.iloc[[3]])[0]
exp_4 = explainer.explain_instance(instance_4, predict_proba_lime, num_features=5)
exp_4.save_to_file('explanation_4.html')

# Explain instance #20 and save the explanation to an image file
instance_20 = preprocessor.transform(X.iloc[[19]])[0]
exp_20 = explainer.explain_instance(instance_20, predict_proba_lime, num_features=5)
exp_20.save_to_file('explanation_20.html')

