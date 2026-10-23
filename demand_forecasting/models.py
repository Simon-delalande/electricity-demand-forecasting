from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge


def make_linear_model():
    return make_pipeline(
        StandardScaler(),
        LinearRegression(),
    )


def make_ridge_model(alpha=1.0):
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )