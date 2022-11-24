import numpy as np

def distance (x,y):
  dist = np.linalg.norm(x-y)
  return dist

class KNN_classifier:
  def __init__(self, n_neighbors: int, **kwargs):
    self.K = n_neighbors

  def fit(self, x: np.array, y: np.array):
    self.x = x
    self.y = y

  def predict(self, x: np.array):
    predictions = [self._predict(i) for i in x]
    predictions = np.array(predictions)
    return predictions

  def _predict(self, x):

    distances = [distance(x, x_train) for x_train in self.x]

    indices = np.argsort(distances)[:self.K]
    labels = [self.y[i] for i in indices]

    num = set(labels)

    most_common = None
    qty_most_common = 0

    for i in num:
      qty = labels.count(i)
      if qty > qty_most_common:
        qty_most_common = qty
        most_common = i
    return most_common
