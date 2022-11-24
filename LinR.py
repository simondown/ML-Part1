import numpy as np

class LinearRegression:
  def __init__(self,**kwargs):
    self.coef_ = None

  def fit(self, x: np.array, y: np.array):
   #y =np.append(y, 1)
    b = np.ones((x.shape[0], 1))
    x=np.hstack((x,b))
    X = np.dot(x.transpose(),x)
    X_= np.linalg.inv(X)
    X__ =np.dot(X_, x.transpose())
    self.coef_ = np.dot(X__, y)


  def predict(self, x: np.array):
     b = np.ones((x.shape[0], 1))
     x=np.hstack((x,b))
     y_pred = np.dot(x, self.coef_)
     return y_pred
