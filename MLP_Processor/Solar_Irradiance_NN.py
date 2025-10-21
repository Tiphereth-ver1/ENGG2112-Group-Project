import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


scope = 25
# Load data
data = pd.read_csv("A.csv")

# create a DataFrame for all lagged features
lagged_cols = {f"lag_{lag}": data["ALLSKY_SFC_SW_DWN"].shift(lag) for lag in range(1, scope)}
lagged_df = pd.DataFrame(lagged_cols, index=data.index)

# concatenate all at once
data = pd.concat([data, lagged_df], axis=1)

# drop NA rows
data = data.dropna()

# Features and target
X = data[[
    "T2M", "T2MDEW", "T2MWET", "RH2M", "PS", "WS2M",
    "Basel Wind Gust", "Basel Wind Direction [10 m]",
    "Basel Precipitation Total", "Basel Wind Speed [10 m]", "Basel Cloud Cover Total",
    "HOUR_SIN", "HOUR_COS"
] + [f"lag_{i}" for i in range(1, scope)]]

y = data["ALLSKY_SFC_SW_DWN"]

Regressor = MLPRegressor(
    hidden_layer_sizes=(50,50),
    max_iter=4000,
    activation='relu',
    solver='adam',
    learning_rate_init=0.006247564316322379,
    learning_rate='adaptive',
    alpha=0.0007174815096277166,
    early_stopping=True,
    random_state=42
)

# Pipeline
pipeline_mlp = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('mlp', Regressor)
])

# TimeSeriesSplit cross-validation
tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []
mae_scores = []
rmse_scores = []
r2_scores = []

# ################### HYPERPARAMETER TESTING ZONE! #######################################################################
# param_dist = {
#     'mlp__hidden_layer_sizes': [(20,20), (50,50), (50,50,20), (100,50), (50,20,20), (60,60),(60,50,20)],
#     'mlp__activation': ['relu', 'tanh'],
#     'mlp__learning_rate_init': uniform(0.001, 0.01),  # samples values between 0.001 and 0.01
#     'mlp__alpha': uniform(0.0001, 0.001),             # L2 regularization
# }

# random_search = RandomizedSearchCV(
#     estimator=pipeline_mlp,
#     param_distributions=param_dist,
#     n_iter=20,             # number of random combinations to try
#     cv=tscv,               # your TimeSeriesSplit
#     scoring='neg_mean_squared_error', # minimize MSE
#     n_jobs=-1,             # parallelize
#     verbose=2,
#     random_state=42
# )
# # Fit
# random_search.fit(X, y)

# # Best parameters & score
# print("Best parameters:", random_search.best_params_)
# print("Best CV score (MSE):", -random_search.best_score_)
# #####################################################################################################################


# Lists to store metrics for all folds
train_metrics = {'MSE': [], 'MAE': [], 'RMSE': [], 'R2': []}
test_metrics  = {'MSE': [], 'MAE': [], 'RMSE': [], 'R2': []}

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit model
    pipeline_mlp.fit(X_train, y_train)

    # Predictions
    y_train_pred = pipeline_mlp.predict(X_train)
    y_test_pred  = pipeline_mlp.predict(X_test)

    # Training metrics
    mse_train = mean_squared_error(y_train, y_train_pred)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = mse_train ** 0.5
    r2_train = r2_score(y_train, y_train_pred)

    # Testing metrics
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = mse_test ** 0.5
    r2_test = r2_score(y_test, y_test_pred)

    # Store metrics
    train_metrics['MSE'].append(mse_train)
    train_metrics['MAE'].append(mae_train)
    train_metrics['RMSE'].append(rmse_train)
    train_metrics['R2'].append(r2_train)

    test_metrics['MSE'].append(mse_test)
    test_metrics['MAE'].append(mae_test)
    test_metrics['RMSE'].append(rmse_test)
    test_metrics['R2'].append(r2_test)

    # Print fold results
    print(f"Fold {fold+1}:")
    print(f"Train -> MSE: {mse_train:.2f}, MAE: {mae_train:.2f}, RMSE: {rmse_train:.2f}, R2: {r2_train:.2f}")
    print(f"Test  -> MSE: {mse_test:.2f}, MAE: {mae_test:.2f}, RMSE: {rmse_test:.2f}, R2: {r2_test:.2f}")
    print("______________")

# Print average metrics
print("Average Training Metrics:")
print({k: sum(v)/len(v) for k, v in train_metrics.items()})
print("Average Testing Metrics:")
print({k: sum(v)/len(v) for k, v in test_metrics.items()})


residuals = y_test - y_test_pred

spearman = stats.spearmanr(y_test, y_test_pred).correlation
kendall = stats.kendalltau(y_test, y_test_pred).correlation
print(f"Spearman ρ: {spearman:.3f}, Kendall τ: {kendall:.3f}")


# Residual vs Predicted
sns.scatterplot(x=y_test_pred, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

# Distribution of residuals
sns.histplot(residuals, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("Residual Distribution")
plt.show()

# Q-Q Plot (normality check)
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

# Select only numeric columns
numeric_X = X.select_dtypes(include=['float64', 'int64'])

# Correlation between each feature and the true target
corr_with_target = numeric_X.corrwith(y).sort_values(ascending=False)
plt.figure(figsize=(6,8))
sns.barplot(x=corr_with_target, y=corr_with_target.index, orient='h')
plt.title("Feature Correlation with Target (y)")
plt.xlabel("Correlation Coefficient (Pearson)")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label="Actual")
plt.plot(y_test_pred, label="Predicted", alpha=0.7)
plt.legend()
plt.title("Actual vs Predicted Demand (Last Fold)")
plt.show()

corrs = data[[f"lag_{i}" for i in range(1, scope)]].corrwith(data["TOTALDEMAND"])
plt.plot(range(1, scope), corrs)
plt.title("Lag Correlation with Target")
plt.show()

plt.plot(residuals)
plt.title("Residuals over Time")
plt.show()