import numpy as np
# Функция подсчета градиента
def gradient(y_true: int, y_pred: float, x: np.array) -> np.array:
  #x.append(1)  # добавляю в конец х = 1
  grad =  [ (y_pred - y_true)*i for i in x]
  return grad


# Функция обновления весов
def update(alpha: np.array, gradient: np.array, lr: float):
  grad = [lr*i for i in  gradient]
  alpha_new=[x - y for x, y in zip(alpha, grad)]
  return alpha_new

#функция тренировки модели
def train(alpha0: np.array, x_train: np.array, y_train: np.array, lr: float, num_epoch: int):
  alpha = alpha0.copy()
  b = np.ones((x_train.shape[0], 1))
  x_train=np.hstack((x_train,b))
  for epo in range(num_epoch):
    for i,x in enumerate(x_train):
      y_pred =  1/(1+np.exp(-np.dot(alpha,x))) #обновление предсказаний с учетом новых весов
      grad = gradient(y_train[i], y_pred, x)
      alpha = update(alpha, grad, lr)
  return alpha
