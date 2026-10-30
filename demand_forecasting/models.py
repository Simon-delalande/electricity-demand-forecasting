"""Forecasts of 48 half-hours from seven historical daily profiles."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, lag=7):
        self.lag = lag

    def fit(self, X, y):
        self.n_outputs_ = y.shape[1]
        if self.lag < 1 or X.shape[1] < self.lag * self.n_outputs_:
            raise ValueError("History does not contain the requested lag.")
        return self

    def predict(self, X):
        start = (self.lag - 1) * self.n_outputs_
        return np.asarray(X)[:, start:start + self.n_outputs_]


def make_linear_model():
    return make_pipeline(StandardScaler(), LinearRegression())


def make_ridge_model(alpha=10.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def make_dense_model(n_features, n_outputs):
    from tensorflow import keras

    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(n_outputs),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4), loss="mse")
    return model


class DenseRegressor(RegressorMixin, BaseEstimator):
    """Choose epochs on an inner chronological holdout, then refit all input days."""

    def __init__(self, epochs=120, patience=12, batch_size=32,
                 validation_fraction=0.15, random_state=42):
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def _dataset(self, X, y, training=False):
        import tensorflow as tf

        dataset = tf.data.Dataset.from_tensor_slices(
            (X.astype("float32"), y.astype("float32")))
        if training:
            # Independent day-ahead samples; the chronological split stays fixed.
            dataset = dataset.shuffle(len(X), seed=self.random_state)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return dataset.batch(self.batch_size).with_options(options)

    def fit(self, X, y):
        from tensorflow import keras

        cut = int(len(X) * (1 - self.validation_fraction))
        if not 0 < cut < len(X):
            raise ValueError("NN needs nonempty inner training and validation sets.")
        keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)
        x_scaler = StandardScaler().fit(X[:cut])
        y_scaler = StandardScaler().fit(y[:cut])
        model = make_dense_model(X.shape[1], y.shape[1])
        history = model.fit(
            self._dataset(x_scaler.transform(X[:cut]), y_scaler.transform(y[:cut]), True),
            validation_data=self._dataset(
                x_scaler.transform(X[cut:]), y_scaler.transform(y[cut:])),
            epochs=self.epochs, verbose=0, shuffle=False,
            callbacks=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience)],
        )
        self.best_epoch_ = int(np.argmin(history.history["val_loss"])) + 1
        # Reinitialize and train on all outer training days, like Linear and Ridge.
        keras.backend.clear_session()
        keras.utils.set_random_seed(self.random_state)
        self.x_scaler_ = StandardScaler().fit(X)
        self.y_scaler_ = StandardScaler().fit(y)
        self.model_ = make_dense_model(X.shape[1], y.shape[1])
        self.model_.fit(self._dataset(
            self.x_scaler_.transform(X), self.y_scaler_.transform(y), True),
            epochs=self.best_epoch_, verbose=0, shuffle=False)
        return self

    def predict(self, X):
        scaled = self.x_scaler_.transform(X).astype("float32")
        prediction = self.model_(scaled, training=False).numpy()
        return self.y_scaler_.inverse_transform(prediction)


def make_models(nn_epochs=120):
    return {
        "Persistence D-7": PersistenceRegressor(lag=7),
        "Persistence D-1": PersistenceRegressor(lag=1),
        "Linear": make_linear_model(),
        "Ridge (alpha=10)": make_ridge_model(),
        "NN": DenseRegressor(epochs=nn_epochs),
    }
