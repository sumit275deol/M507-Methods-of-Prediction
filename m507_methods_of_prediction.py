#!/usr/bin/env python
# coding: utf-8

# # M507 Methods of Prediction
# 

# ## Introduction

# ### Problem Statement

# The difficulty of this work is the proper forecasting of the median house values in California districts. The estimation of the price of the housing is a complicated regression issue that is threatened by various geographic, demographic, and property-related factors. The conventional valuation techniques can hardly reflect the complex interrelationship among these variables. The given project applies the use of deep learning to create the predictive models, suggesting ideas that could predict housing prices depending on such attributes as the location coordinates, property features, population density, median income index, and closeness to the ocean. The aim is to develop strong neural network architectures that are able to train non-linear patterns of the data to give sound price predictions.

# ### Overview of Business Problem

# The government and other stakeholders of real estate such as investors, developers, insurance agencies and other government agencies need precise predictions of housing prices to make an informed decision. The property valuation influences the choices of investments, risk evaluation, formation of tax policy, and the prospective projects in the city. The existing appraisal procedures could be lengthy, biased, and uncertain when it comes to various appraisers. The company issue involves the development of a data-autonomized solution to the business problem that enables a steady and precise housing valuation at scale. It allows the stakeholders to evaluate the values of thousands of districts within very limited time, detect undervalued markets, allocate resources effectively, and make strategic decisions not only based on subjective judgment but also supported by quantitative analysis.

# ## Importance of Solving Problem
# 

# The economic and social implications of addressing this issue in predicting the housing prices are massive. Real valuations will aid in avoiding market bubbles with the forecast of realistic prices and the occurrence of overpriced properties. In the case of homebuyers, it is the valid predictions that guarantee the fair pricing and eliminate overpayment. The financial institutions will gain by having a better mortgage risk evaluation and management of the loan portfolio. Urban development planning and property tax assessment of the government agencies can be optimized. Data insights help real estate professionals to develop competitive advantages. As well, this predictive power is used to facilitate low-cost housing since it detects price trends and accessible areas. The solution illustrates how machine learning can be practically applied to finance and creates the methodologies that can be applied to other real estate markets worldwide.

# ## Data Collection Strategy

# https://www.kaggle.com/datasets/camnugent/california-housing-prices

# The dataset is taken out of the California census of 1990, which was realized on the Kaggles deposit of California Housing Prices. This source was chosen as it is fully district-level housing information that has 20,640 observations with 10 variables and the sample size is sufficient to train a deep learning model. The census material contains validated geographic coordinates, property features, demographic records of individuals and real median house values as ground truth classifications. Data collection strategy is highly marked by completeness and reliability because the official census data is used, as opposed to crowdsourced or scraped information. Although the 1990 period restricts interest to the present day markets, it offers a clean and well documented dataset that can be used effectively in illustrating the machine learning methodologies and formulating baseline prediction frameworks that can be used to draw conclusions of the present day housing information.

# 

# ## Data Exploration

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)



data = pd.read_csv("/content/housing.csv")
print("Dataset loaded successfully!")
print(f"Shape: {data.shape}")
data.head()


# The script will be reading housing.csv file in pandas and it will be read successfully with a total of 20,640 records with 10 columns. The head () command would show the top five rows of geographic coordinates (longitude, latitude) housing data (medianage, totalrooms, totalbedrooms), demographic (population, households, medianincome) and the target variable (medianhousevalue), as well as the target variable (oceanproximity) is a categorical variable. The data is all of the float64 type, which defines the format of the dataset used in further analysis.

# In[5]:


data.info()
data.describe()


# The info() method also shows that there are 20,640 records with 10 different columns with totalbedrooms showing 207 missing values (1.00 percent of data value). The describe() drylands that give a detailed statistical overview of all numerical features (mean, standard deviation, and quartiles). It is important to note that the median house values are between 14, 999 to 500,001, median income falls between 0.499 to 15.000 and housing age falls within 1-52 years, which offer the basic knowledge of data distributions and scales.
# 

# In[7]:


# Check for missing values

missing_values = data.isnull().sum()
missing_percent = (missing_values / len(data)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_values,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Missing_Count'] > 0])
duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
print(data['ocean_proximity'].value_counts())
print("\n")
print(data['ocean_proximity'].value_counts(normalize=True) * 100)


# The total number of missing values (1.00) in totalbedrooms is identified in the analysis with 0 as a duplicate row, which enforces the uniqueness of data in the table, indicating that the analysis was completed. The oceanproximity categorical variable has five different categories; <1H OCEAN (44.26, 9,136 districts), INLAND (31.74, 6,551), NEAR OCEAN (12.88, 2,658), NEAR BAY (11.09, 2,290), and ISLAND (0.02, 5). The proximity to the coastline is found to be a strong location attribute that needs to be encoded as one-hot in order to integrate it into the model.

# In[9]:


# Scatter plot showing geographic distribution with house values
plt.figure(figsize=(12, 8))
scatter = plt.scatter(data['longitude'], data['latitude'],
                     c=data['median_house_value'],
                     cmap='viridis',
                     alpha=0.4,
                     s=10)
plt.colorbar(scatter, label='Median House Value')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of California Housing Prices')
plt.grid(True, alpha=0.3)
plt.show()


# The scatter plot indicates the housing districts in California in terms of longitude (x-axis) and latitude (y-axis) with the color intensity showing the median house values (purple=low, yellow=high, $50,000500,000 range). The geography of California is clearly identified in the visualization with an agglomeration of the high-value areas (yellow/green) in the San Francisco Bay Area and Coastal Los Angeles/San Diego. There is strong geographic relationship between location and housing prices in that lower value properties (purple) take the lead in inland areas.

# In[14]:


correlations = data.select_dtypes(include=[np.number]).corr()['median_house_value'].sort_values(ascending=False)
print("Correlation with Median House Value:")
# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = data.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()


# The Pearson correlation table measures relationships between nine numerical features with colors ( -1 +1). Median income has the highest p value of positive correlation with median house value (0.69), whereas houses characteristics (totalrooms, totalbedrooms, population, households) depict the high multicollinearity (0.86-0.98 intercorrelations). Geographic positions are only weakly associated with the price, which are intricate non-linear geographical associations. It is the feature engineering decision that is guided by the heatmap and might show redundancy that needs the dimensionality of consideration.

# In[16]:


# Scatter plot showing house locations with price
plt.figure(figsize=(12, 8))
scatter = plt.scatter(data['longitude'], data['latitude'],
                     c=data['median_house_value'],
                     cmap='YlOrRd',
                     alpha=0.4,
                     s=data['population']/100)
plt.colorbar(scatter, label='Median House Value')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('California Housing Prices - Geographic Distribution', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# It is a fine-tuning scatterplot on a color gradient of red-yellow (YlOrRd colormap) and the size of the points represents the population/100, which produces a bubble-chart effect. The analysis focuses on premium coastal markets, such as San Francisco Bay Area, Los Angeles Basin, and San Diego County, that have concentric circles of red, which represents areas of high value. The inland areas were seen with lighter yellow colors to depict low valuation. It is a multi dimensional representation, which at the same time captures location, price, and density of populations to understand the spatial pattern comprehensively.

# ## Data Preprocessing and Feature Engineering

# In[20]:


# Create a copy for preprocessing
df = data.copy()
print(f"Working with a copy of the dataset")
print(f"Shape: {df.shape}")
# Check missing values again
print("Missing values before handling:")
print("="*50)
print(df.isnull().sum())
print("\n")

# Strategy: Fill total_bedrooms with median value
print("Strategy: Fill 'total_bedrooms' missing values with median")


# In[21]:


# Fill missing values in total_bedrooms with median
median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)

print("Missing values after handling:")
print("="*50)
print(df.isnull().sum())


# In[24]:


# Feature Engineering: Create rooms per household
df['rooms_per_household'] = df['total_rooms'] / df['households']
print("Created feature: rooms_per_household")
df[['total_rooms', 'households', 'rooms_per_household']].head()
# Feature Engineering: Create bedrooms per room
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
print("Created feature: bedrooms_per_room")
df[['total_bedrooms', 'total_rooms', 'bedrooms_per_room']].head()
# Feature Engineering: Create population per household
df['population_per_household'] = df['population'] / df['households']
print("Created feature: population_per_household")
df[['population', 'households', 'population_per_household']].head()


# Key three engineered features are prepared as ratio calculations: the number of rooms in the house (roomsperhousehold), number of bedrooms in the house (bedroomsperroom) and the number of people in the house (populationperhousehold). The output presents the sample calculations of values such as 2.56 rooms per household that give normalized metrics that reflect property density and household composition features. These derived features minimize multicollinearity and construct more valuable predictors besides raw counts that represent housing quality and occupancy patterns.

# In[25]:


# Display all new features
print("Dataset with new features:")
print("="*50)
print(f"Shape: {df.shape}")
print("\nNew columns:")
print(df.columns.tolist())


# In[ ]:





# ## Model Training and Evaluation

# In[28]:


# Separate features and target variable
X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
from sklearn.model_selection import train_test_split

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"\nTraining set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# In[29]:


# Identify numerical and categorical columns
numerical_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print("Numerical Features:")
print(numerical_features)
print("\nCategorical Features:")
print(categorical_features)


# In[30]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# In[33]:


# Numerical pipeline: impute missing values and scale
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

print("Numerical pipeline created:")
print(numerical_pipeline)

# Categorical pipeline: impute missing values and one-hot encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

print("Categorical pipeline created:")
print(categorical_pipeline)
# Combine both pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

print("Full preprocessor created:")
print(preprocessor)


# Scikit-learn ColumnTransformer is an algorithm that preprocesses numeric and categorical features in parallel. SimpleImputer (median strategy) is used before StandardScaler in the numerical pipeline to impute the data and make them normal. Encoding of the categorical pipeline is done through SimpleImputer (mostfrequent strategy) and OneHotEncoder (handleunknown= ignore, sparseoutput=False). Transformer uses nine numerical features (including engineered variables) and oceanproximity independently, thus making sure that the right transformations are done before model training to avoid data leakage by using fittransform with training data only.

# In[ ]:


# Fit on training data and transform
X_train_processed = preprocessor.fit_transform(X_train)
print(f"Processed training data shape: {X_train_processed.shape}")
print(f"Original training data shape: {X_train.shape}")


# In[40]:


# Get feature names after preprocessing
try:
    # Get numerical feature names
    num_feature_names = numerical_features

    # Get categorical feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)

    # Combine all feature names
    all_feature_names = num_feature_names + list(cat_feature_names)

    print(f"Total features after preprocessing: {len(all_feature_names)}")
    print("\nFeature names:")
    for i, name in enumerate(all_feature_names, 1):
        print(f"{i}. {name}")
except:
    print("Feature names extracted")


# In[41]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time


# In[44]:


# Split training data into train and validation sets
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42
)

print(f"Final training set: {X_train_final.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# In[49]:


def build_model1(layers, optimizer, input_dim):
    """
    Build a simple Deep Neural Network
    """
    model = Sequential()

    # First hidden layer
    model.add(Dense(layers[0], activation='relu', input_dim=input_dim))

    # Additional hidden layers if specified
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))

    # Output layer
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# This role forms a fundamental deep neural network through Keras sequential API controlled architecture. Parameters incorporated are layers list (defined number of neurons per layer), optimizer object, and input dimension. Dense layers having ReLU activation are progressively added to the architecture, whose initial layer is defined the inputdim. Single-neuron output layer (zero-activation) is utilized in regression prediction. Mean squared error (MSE) loss and mean absolute error (MAE) metric are used as model compilation metrics resulting in a trained-ready architecture to predict housing prices.

# In[50]:


def build_model2(layers, optimizer, input_dim, dropout_rate=0.2):
    """
    Build a Deep Neural Network with Dropout regularization
    """
    model = Sequential()

    # First hidden layer
    model.add(Dense(layers[0], activation='relu', input_dim=input_dim))
    model.add(Dropout(dropout_rate))

    # Additional hidden layers if specified
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# This improved capability uses dropout regularization to avoid overfitting of deeper networks. Between each Dense layer followed by ReLU activation, there is a Dropout layer (default rate=0.2) that stochastically does nothing to 1 in 5 neurons during training, which is a forced distributed form of feature learning. The architecture is the same as that of buildmodel1 but includes dropout in between hidden layers but not the output layer. The structures of compilation are the same (MSE loss, MAE metric), which develops a stronger architecture to make generalization on out-of-view validation data.

# In[51]:


# Model 1 configurations (Simple DNN) - Optimized
model1_configs = [
    {"layers": [32], "optimizer": Adam(0.01)},
    {"layers": [64], "optimizer": Adam(0.01)},
    {"layers": [32, 16], "optimizer": Adam(0.01)},
    {"layers": [64, 32], "optimizer": Adam(0.01)},
    {"layers": [128], "optimizer": Adam(0.005)},
    {"layers": [128, 64], "optimizer": Adam(0.005)},
    {"layers": [64], "optimizer": RMSprop(0.01)},
    {"layers": [64, 32], "optimizer": RMSprop(0.01)},
    {"layers": [128, 64, 32], "optimizer": Adam(0.01)},
    {"layers": [256], "optimizer": Adam(0.005)},
]

# Model 2 configurations (DNN with Dropout) - Optimized
model2_configs = [
    {"layers": [32], "optimizer": Adam(0.01), "dropout": 0.2},
    {"layers": [64], "optimizer": Adam(0.01), "dropout": 0.2},
    {"layers": [32, 16], "optimizer": Adam(0.01), "dropout": 0.2},
    {"layers": [64, 32], "optimizer": Adam(0.01), "dropout": 0.2},
    {"layers": [128], "optimizer": Adam(0.005), "dropout": 0.3},
    {"layers": [128, 64], "optimizer": Adam(0.005), "dropout": 0.3},
    {"layers": [64], "optimizer": RMSprop(0.01), "dropout": 0.2},
    {"layers": [64, 32], "optimizer": RMSprop(0.01), "dropout": 0.2},
    {"layers": [128, 64, 32], "optimizer": Adam(0.01), "dropout": 0.2},
    {"layers": [256], "optimizer": Adam(0.005), "dropout": 0.25},
]

print(f"Model 1 configurations: {len(model1_configs)}")
print(f"Model 2 configurations: {len(model2_configs)}")


# In[52]:


# Store results for Model 1
results_model1 = []
input_dim = X_train_final.shape[1]

print("Training Model 1 (Simple DNN) - All Configurations")
print("="*70)

for idx, config in enumerate(model1_configs, 1):
    print(f"\nConfiguration {idx}/{len(model1_configs)}")
    print(f"Layers: {config['layers']}, Optimizer: {config['optimizer'].__class__.__name__}")

    start_time = time.time()

    # Build model
    model = build_model1(config['layers'], config['optimizer'], input_dim)

    # Train model with early stopping
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=30,  # Reduced from 50
        batch_size=64,  # Increased from 32 for faster training
        verbose=0
    )

    # Predict on validation set
    y_val_pred = model.predict(X_val, verbose=0).flatten()

    # Calculate metrics
    r2 = r2_score(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)

    training_time = time.time() - start_time

    # Store results
    results_model1.append({
        'Config': idx,
        'Model': 'Simple DNN',
        'Layers': str(config['layers']),
        'Optimizer': config['optimizer'].__class__.__name__,
        'Learning_Rate': float(config['optimizer'].learning_rate.numpy()),
        'R2_Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Training_Time': training_time
    })

    print(f"R² Score: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, Time: {training_time:.2f}s")

print("\nModel 1 training completed!")


# In[53]:


# Store results for Model 2
results_model2 = []

print("\nTraining Model 2 (DNN with Dropout) - All Configurations")
print("="*70)

for idx, config in enumerate(model2_configs, 1):
    print(f"\nConfiguration {idx}/{len(model2_configs)}")
    print(f"Layers: {config['layers']}, Optimizer: {config['optimizer'].__class__.__name__}, Dropout: {config['dropout']}")

    start_time = time.time()

    # Build model
    model = build_model2(config['layers'], config['optimizer'], input_dim, config['dropout'])

    # Train model with early stopping
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=30,  # Reduced from 50
        batch_size=64,  # Increased from 32 for faster training
        verbose=0
    )

    # Predict on validation set
    y_val_pred = model.predict(X_val, verbose=0).flatten()

    # Calculate metrics
    r2 = r2_score(y_val, y_val_pred)
    mse = mean_squared_error(y_val, y_val_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_val, y_val_pred)

    training_time = time.time() - start_time

    # Store results
    results_model2.append({
        'Config': idx,
        'Model': 'DNN with Dropout',
        'Layers': str(config['layers']),
        'Optimizer': config['optimizer'].__class__.__name__,
        'Learning_Rate': float(config['optimizer'].learning_rate.numpy()),
        'Dropout': config['dropout'],
        'R2_Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'Training_Time': training_time
    })

    print(f"R² Score: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, Time: {training_time:.2f}s")

print("\nModel 2 training completed!")


# ## Model Assessment

# In[54]:


# Combine results from both models
all_results = results_model1 + results_model2

# Create DataFrame
results_df = pd.DataFrame(all_results)

print(f"Total experiments conducted: {len(results_df)}")
print("\nResults DataFrame shape:", results_df.shape)
results_df.head(10)


# In[55]:


# Display all experimental results sorted by R2 Score
results_sorted = results_df.sort_values('R2_Score', ascending=False)

print("EXPERIMENTAL RESULTS - ALL CONFIGURATIONS")
print("="*100)
print(results_sorted.to_string(index=False))


# The table of results that is sorted by the R 2 score (descending) contains all model configurations (20). Configuration 9 (Simple DNN, layers [128,64,32], Adam optimizer, LR=0.01) is the best performing with a R2=0.7498, RMSE=58,755, MAE=40,432, training time=23.13s. The table shows that Simple DNN models tend to be superior in comparison to the models with the addition of Dropout and it is also clear that Adam optimizer is generally better than RMSprop. The configurations vary widely containing R2=0.4154-0.7498 indicating that different architectures and hyperparameter changes yield a significant performance difference.

# In[56]:


# Summary statistics by model type
print("\nSUMMARY STATISTICS BY MODEL TYPE")
print("="*70)

summary = results_df.groupby('Model').agg({
    'R2_Score': ['mean', 'std', 'min', 'max'],
    'RMSE': ['mean', 'std', 'min', 'max'],
    'MAE': ['mean', 'std', 'min', 'max'],
    'Training_Time': ['mean', 'sum']
}).round(4)

print(summary)


# Compared to Simple DNN and DNN with Dropout, aggregated statistics compare these statistical models based on major criteria. Simple DNN demonstrates better mean R 2 (0.6326+-0.1027) than Dropout models (0.6262+-0.0898), and the highest R 2 (0.7498) as compared to 0.6992. As of average RMSE, it stands at 58755 (Simple) and 64426 (Dropout), which means improved predictive power without regularization. Simple DNN (mean 22.44s, total 224.39s) is more likely to be more efficient in training than Dropout models (24.73s mean, 247.27s total). The findings indicate that there is adequate regularization in the dataset (16,512 training samples) even without dropout.

# In[57]:


# Display top 5 configurations
print("\nTOP 5 CONFIGURATIONS (Based on R² Score)")
print("="*100)

top_5 = results_sorted.head(5)
for idx, row in top_5.iterrows():
    print(f"\nRank {list(top_5.index).index(idx) + 1}:")
    print(f"  Model: {row['Model']}")
    print(f"  Layers: {row['Layers']}")
    print(f"  Optimizer: {row['Optimizer']} (LR: {row['Learning_Rate']})")
    if 'Dropout' in row and pd.notna(row['Dropout']):
        print(f"  Dropout: {row['Dropout']}")
    print(f"  R² Score: {row['R2_Score']:.4f}")
    print(f"  RMSE: {row['RMSE']:.2f}")
    print(f"  MAE: {row['MAE']:.2f}")
    print(f"  Training Time: {row['Training_Time']:.2f}s")


# In[58]:


# Get the best configuration
best_config = results_sorted.iloc[0]

print("BEST MODEL CONFIGURATION")
print("="*70)
print(f"Model Type: {best_config['Model']}")
print(f"Configuration: {best_config['Config']}")
print(f"Layers: {best_config['Layers']}")
print(f"Optimizer: {best_config['Optimizer']}")
print(f"Learning Rate: {best_config['Learning_Rate']}")
if 'Dropout' in best_config and pd.notna(best_config['Dropout']):
    print(f"Dropout Rate: {best_config['Dropout']}")
print(f"\nValidation Performance:")
print(f"  R² Score: {best_config['R2_Score']:.4f}")
print(f"  RMSE: {best_config['RMSE']:.2f}")
print(f"  MAE: {best_config['MAE']:.2f}")
print(f"  Training Time: {best_config['Training_Time']:.2f} seconds")


# The most suitable option is Simple DNN architecture with 3 hidden layers [128, 64, 32 neurons], Adam optimization (LR=0.00999999997764825) where validation R2=0.7498, RMSE=$58,755.36, MAE=$40,432.33 in 23.23 seconds training time are found. Such progressive layer sparseness (128-64-32) introduces a bottleneck in information towards making an efficient feature representation. The configuration balances the model complexity with computational efficiency which explains a price variance of 74.98, with simple training computation speed to practical training speed in cases of production deployment.

# ## Final Discussion

# The results of the proposed deep neural network pipeline shown to be able to predict with high accuracy with the optimal model having variance explanation (R2=0.7498) of 75 percent and the mean absolute error of 40,432 on California housing prices. Major strengths are systematic feature engineering (roomsperhousehold, bedroomsperroom, populationperhousehold) that have eliminated multicollinearity but increased interpretability, thorough preprocessing pipelines, which deal with missing data, and categorical encoding automatically, and full hyperparameter exploration of 20 architecture configurations determined the best architecture. The correlation analysis showed that the most informative feature was median-income (r=0.69) and geographic clustering effects represented in the spatial visualizations. Nevertheless, there are major shortcomings: the RMSE of the model of 58,755 indicates a 28 percent average lack of capture of complex price determinants; the 1990 census data bring serious temporal validity problems to modern market applications; deep neural networks do not inherently have explainability as tree-based models do; and the current market makes no effort to capture post99,000 determinants of price such as improvement in the technology sector, housing policy changes, or recent market dynamics. In business deployment, it has been recommended to: limit it to relative price comparisons other than absolute predictions, use SHAP or LIME explainability structures to comply with regulatory systems, re-train using modern data, which takes into account the economic indicators and neighborhood amenities, and implement monitoring systems that detect prediction drift. The model demonstrates potential to automated valuation models (AVMs) in historical research or scholarly studies but needs to undergo significant improvement, such as the use of up to date data, external economical characteristics and collective approaches, prior to the actual production real estate.

# ## Conclusion

# This project was able to prove the range of deep learning to the prediction of housing prices and the results showed that with 20 configurations of neural networks the accuracy is 75 percent through systematic experimentation. This was further demonstrated by the Simple DNN architecture with progressive parameter layer reduction [128,64,32], which was found to be better than dropout-regularized counterparts, meaning that there was sufficient data in the dataset to generalize. Although median income and engineered density characteristics were the most predictive, the 1990 data vintage and modelopaquity are barriers of critical deployment. The focus of future work should be on the present data collection, incorporating explainable AI, and hybrid methods of integrating neural networks and interpretable models to balance the accuracy with the transparency requirements of stakeholders in respect of production real estate applications.

# ## Bibliography

# Abedi, V., Avula, V., Chaudhary, D., Shahjouei, S., Khan, A., Griessenauer, C.J., Li, J. and Zand, R. (2021). Prediction of long-term stroke recurrence using machine learning models. Journal of Clinical Medicine, 10(6), 1286.
# 
# 
# Fernandez-Lozano, C., Gestal, M., Munteanu, C.R., Dorado, J. and Pazos, A. (2021). Random forest-based prediction of stroke outcome. Scientific Reports, 11, 10071.
# 
# MacEachern, S.J. and Forkert, N.D. (2021). Machine learning for precision medicine. Genome, 64(4), 416-425.
# 
# 
# 
# Sarker, I.H. (2021). Machine learning: Algorithms, real-world applications and research directions. SN Computer Science, 2, 160.
# 

# In[ ]:




