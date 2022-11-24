from sklearn.metrics import r2_score
import numpy as np
# WA
def r2(y_true, y_pred):

    D_f = sum(map(lambda x: x**2, y_pred-y_true))
    y_mean = np.sum(y_true)/len(y_true)
    D_y = sum(map(lambda x: x**2, y_pred-y_mean))

    R2 = 1 - (D_f/D_y)

    return R2

y_true = np.array([1, 0, 2])
y_pred = np.array([0.5, 0, 2.1])

print(r2(y_true, y_pred))
print(r2_score(y_true, y_pred))

#OK
import numpy as np
def r2(y_true, y_pred):
    y = np.sum((y_true)/len(y_true))
    a = np.sum((np.array(y_true) - np.array(y_pred))**2)
    b = np.sum((np.array(y_true) - y)**2)
    return 1 -  a/b
