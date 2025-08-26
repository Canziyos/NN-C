
def MSE(y_true, y_pred):
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("MSE: y_true and y_pred must be non-empty and equal length.")
    n = len(y_true)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / n



print(MSE([1], [1]))                # expect 0.0
print(MSE([1], [0]))                # expect 1.0
print(MSE([0, 1], [0, 1]))          # expect 0.0
print(MSE([0, 1], [1, 0]))          # expect 1.0
print(MSE([0.5, 0.5], [1, 0]))      # expect 0.25
