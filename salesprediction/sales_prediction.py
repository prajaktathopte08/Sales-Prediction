# ============================
# Sales Prediction using Python
# ============================

# 📌 Step 1: Import Required Libraries
import sys
sys.stdout.reconfigure(encoding='utf-8')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 📌 Step 2: Load the Dataset
data = pd.read_csv("advertising.csv")
print("Dataset Loaded Successfully\n")
print(data.head())
print("\nDataset Info:\n", data.info())

# 📌 Step 3: Exploratory Data Analysis (EDA)
print("\nMissing Values:\n", data.isnull().sum())
print("\nDataset Summary:\n", data.describe())

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("📊 Correlation Heatmap", fontsize=14)
plt.show()

# Scatter plots for each feature vs Sales
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.suptitle("📈 Relationship between Ad Spend & Sales")#, y=1.02)
plt.show()

# 📌 Step 4: Prepare Data for Modeling
X = data[['TV', 'Radio', 'Newspaper']]  # Features
y = data['Sales']                       # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 5: Build the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📌 Step 6: Print Model Coefficients
print("\nModel Parameters:")
print("Intercept:", model.intercept_)
print("Coefficients for TV, Radio, Newspaper:", model.coef_)

# 📌 Step 7: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f" Mean Absolute Error (MAE): {mae:.2f}")
print(f" Mean Squared Error (MSE): {mse:.2f}")
print(f" Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f" R² Score: {r2:.2f}")

# 📌 Step 8: Visualization - Actual vs Predicted
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color="blue", edgecolor="k")
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.title("🔵 Actual vs Predicted Sales", fontsize=14)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()

# 📌 Step 9: Predict Future Sales
new_data = pd.DataFrame({'TV': [200], 'Radio': [50], 'Newspaper': [20]})
predicted_sales = model.predict(new_data)
print(f"\n Predicted Sales for TV=200, Radio=50, Newspaper=20 ➡ {predicted_sales[0]:.2f} units")


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# ✅ Decision Tree Model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

# ✅ Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# ✅ Polynomial Regression Model
poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

# ✅ Evaluation Function
def evaluate_model(name, y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"\n {name} Performance:")
    print(f" MAE: {mae:.2f}")
    print(f" RMSE: {rmse:.2f}")
    print(f" R² Score: {r2:.2f}")

# ✅ Evaluate All Models
evaluate_model("Linear Regression", y_test, y_pred)
evaluate_model("Decision Tree", y_test, tree_pred)
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Polynomial Regression", y_test, poly_pred)

try:
    # 👇 move all your existing code here (everything below this line)
    print("Script starting...")
    # ...your existing code here...

except Exception:
    traceback.print_exc()

