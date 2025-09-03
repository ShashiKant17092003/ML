import numpy as np
import matplotlib.pyplot as plt

X = np.arange(1, 21, dtype=float)

y = np.array([
    35, 37, 39, 41, 43, 44, 45, 46, 47, 48,
    49, 50, 50, 50, 50, 50, 50, 50, 50, 50
], dtype=float)

sumX = X.sum()
sumY = y.sum()
sumXY = (X*y).sum()
sumX2 = (X**2).sum()

n = len(X)

m = (n*sumXY-sumX*sumY)/(n*sumX2-(sumX)**2)
c = (sumY - m*sumX)/n

y_pred = m*X+c

mse = np.mean((y-y_pred)**2)
rmse = np.sqrt(mse)
ss_res = ((y-y_pred)**2).sum()
ss_tot = ((y-y.mean())**2).sum()
r2 = 1 - ss_res/ss_tot

print(f"Slope(m) = {m:.4f}\nIntercept = {c:.4f}\n\nPredictions = {np.round(y_pred,1)}\n\nMSE = {mse:.4f}\nRMSE = {rmse:.4f}\nR2 = {r2:.4f}")

plt.scatter(X,y,label = "Actual")
plt.scatter(X,y_pred,color="red",label="Predicted")
plt.plot(X,y_pred,color="green")
plt.xlabel("Hours Study")
plt.ylabel("Marks Scored")
plt.legend()
plt.grid(alpha=0.5)
plt.show()