from engine import err
import numpy as np

#NOTE: Unfinished example

X = np.array([
    [err(0, .01), err(0, .01)],
    [err(1, .01), err(0, .01)],
    [err(0, .01), err(1, .01)],
    [err(1, .01), err(1, .01)],
], dtype=object)


y_true = np.array([
    [err(0)],
    [err(1)],
    [err(1)],
    [err(0)]
], dtype=object)

print(X)
print(y_true)

tanh = np.vectorize(err.tanh)
 
out = np.sum(tanh(X.T@y_true))
out.backward()
print(out)
