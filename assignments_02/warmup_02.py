# --- Scikit-Learn API ---
# Q1
import numpy as np
from sklearn.linear_model import LinearRegression

temp_min = np.array([1,4,8,12,16,20]).reshape(-1,1)
temp_max = np.array([8,10,15,18,23,27])

model = LinearRegression()
model.fit(temp_min, temp_max)

predictions = model.predict([[6],[18]])

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for low of 6:", predictions[0])
print("Predicted high for low of 18:", predictions[1])

# Q2
x = np.array([10, 20, 30, 40, 50])
print("Original shape", x.shape)

x_2d = x.reshape(-1, 1)
print("New shape:", x_2d.shape )

# X needs to be 2D so rows represent observations and columns represent features.
# --- Linear Regression ---

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")

plt.xlabel("Daily low temperature (C)")
plt.ylabel("Daily high temperature (C)")
plt.savefig("outputs/high_vs_low.png")
plt.close()

# The red and blue dots overlap near lows of 0, 5, 10, and 15 C.
# Rainy days generally have lower highs than dry days with similar lows.
# This suggests is_rainy could help predict the daily high.

# Q2
X = temp_min.reshape(-1,1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Q3 slope:", model.coef_[0])
print("Q3 intercept:", model.intercept_)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("Q3 RMSE:", rmse)
print("Q3 test R2:", r2)

# When the daily low increases by 1 C, the predicted high goes up by about 0.94 C.

# Q4
X_full = np.column_stack([temp_min, is_rainy])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print("Q3 test R2:", r2)
print("Q4 test R2:", r2_full)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# With the same daily low, a rainy day's predicted high is about 2.92 C lower than a dry day's.
# Adding rain helped because the test R2 increased from about 0.894 to 0.945.


# Q5
y_pred_full = model_full.predict(X_test_full)

plt.scatter(y_pred_full, y_test_full)
plt.title("Predicted vs Actual")
plt.xlabel("Predicted high temperature (C)")
plt.ylabel("Actual high temperature (C)")

lowest = min(y_pred_full.min(), y_test_full.min())
highest = max(y_pred_full.max(), y_test_full.max())

plt.plot([lowest, highest], [lowest, highest], "k--")
plt.savefig("outputs/predicted_vs_actual_warmup.png")
plt.close()

# Above the diagonal, the prediction is too low.
# Below the diagonal, the prediction is too high.
