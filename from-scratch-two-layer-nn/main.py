import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class TwoLayerNeuralNetwork:
    def __init__(self, layers=[4, 10, 3], lr=0.0001, epochs=10000, rand_state=5):
        self.lr = lr
        self.epochs = epochs
        self.X = None
        self.y = None
        self.sample_size = None
        self.loss = []
        self.params = {}
        self.temps = {}
        self.layers = layers
        self.rand_state = rand_state
        self.init_weights()

    def init_weights(self):
        np.random.seed(self.rand_state)
        n_x, n_h, n_y = self.layers[0], self.layers[1], self.layers[2]
        mu, sigma = 0, 0.1
        W1 = np.random.normal(mu, sigma, size=(n_h, n_x))  # (10,n_x)
        W2 = np.random.normal(mu, sigma, size=(n_y, n_h))  # (3,10)
        b1 = np.random.normal(mu, sigma, size=(n_h, 1))  # (10,1)
        b2 = np.random.normal(mu, sigma, size=(n_y, 1))  # (3,1)

        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    def sigmoid(self, x, d=False):
        '''
        Args: x is input
              d is parameter it's derivative or not. If it is True then apply derivative of function.
        '''
        if d:
            return np.multiply(np.exp(-x), np.power(self.sigmoid(x, d=False), 2))
        else:
            return 1 / (1 + np.exp(-x))

    def htan(self, x, d=False):
        '''
        Args: x is input
              d is parameter it's derivative or not. If it is True then apply derivative of function.
        '''
        if d:
            return (4 / ((np.exp(x) + np.exp(-x)) ** 2)) + np.power(self.htan(x, d=False), 2)
        else:
            return np.tanh(x)

    def forward(self, X):
        # X.shape : (n_x,sample_size)
        W1 = self.params["W1"]  # (n_h,n_x)
        W2 = self.params["W2"]  # (n_y,n_h)
        b1 = self.params["b1"]  # (n_h,1)
        b2 = self.params["b2"]  # (n_y,1)

        Z1 = np.dot(W1, X) + b1  # (n_h,n_x) @ (n_x,m) = (n_h,m) .. liner part
        A1 = self.htan(Z1)  # (n_h,m) # .. non-lineer with tanh
        # print("A1.shape forward:",A1.shape)
        # print("W2 shape:",W2.shape,"b2 shape:",b2.shape)
        Z2 = np.dot(W2, A1) + b2  # (n_y,n_h) @ (n_h,m) = (n_y,m) .. lineer part
        A2 = self.sigmoid(Z2)  # (n_y,m) ... non-lineer with sigmoid : prediction
        self.temps = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}  # .. store
        return A2

    def ln(self, x, d=False):
        '''
        Args: x is input
              d is parameter it's derivative or not. If it is True then apply derivative of function.
        '''
        if d:
            return 1 / x
        else:
            return np.log(x)

    def log_loss(self, yhat, y, d=False):
        '''
        Args: yhat is prediction
              y is real value
              d is parameter it's derivative or not. If it is True then apply derivative of function.
        '''
        if d:
            derivative = -1 * (y * self.ln(yhat, True) + (1 - y) * self.ln((1 - yhat), True))
            return derivative
        else:
            loss = (y * self.ln(yhat) + (1 - y) * self.ln(1 - yhat))
            return loss

    def cost_function(self, yhat):
        cost = (-1 / self.sample_size) * np.sum(self.log_loss(yhat, self.y))
        return np.round(cost, 3)

    def backward(self):
        A2 = self.temps["A2"]
        A1 = self.temps["A1"]
        W2 = self.params["W2"]

        # np.multiply(dA2,self.sigmoid(A2,d=True))
        dZ2 = A2 - self.y  # (n_y,m) - (1,m) = (n_y,m) = (3,120)
        dW2 = np.dot(dZ2, A1.T) / self.sample_size  # (n_y,m) @ (n_h,m).T = (n_y,n_h) = (3,10)
        db2 = np.sum(dZ2, axis=1, keepdims=True) / self.sample_size  # (n_y,m)

        dZ1 = np.multiply(np.dot(W2.T, dZ2), self.htan(A1, d=True))  # (n_y,n_h).T @ (n_y,m) * (n_h,m) = (n_h,m)

        dW1 = np.dot(dZ1, self.X.T) / self.sample_size  # (n_h,m) @ (n_x,m).T = (n_h,n_x)
        db1 = np.sum(dZ1, axis=1, keepdims=True) / self.sample_size  # (n_h,m)

        self.grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update(self):
        W1 = self.params["W1"]  # (n_h,n_x)
        W2 = self.params["W2"]  # (n_y,n_h)
        b1 = self.params["b1"]  # (n_h,1)
        b2 = self.params["b2"]  # (n_y,1)

        dW1 = self.grads["dW1"]
        db1 = self.grads["db1"]
        dW2 = self.grads["dW2"]
        db2 = self.grads["db2"]

        W1 -= self.lr * dW1
        W2 -= self.lr * dW2
        b1 -= self.lr * db1
        b2 -= self.lr * db2

        self.params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
        # return params

    def fit(self, X, y, lr=0.001, epochs=5):
        self.X = X
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.sample_size = float(y.shape[1])
        for i in range(self.epochs):
            yhat = self.forward(self.X)
            cost = self.cost_function(yhat)
            self.backward()
            self.update()
            self.loss.append(cost)

            if i > 10 and cost > self.loss[-3] and cost > self.loss[-4] and cost > self.loss[-5]:
                print("Early stopping applied..Epoch {} and current Cost: {}".format(i, cost))
                break
            if i % 1000 == 0:
                print("Epoch ", i, ":\tCost: ", cost)

        print("Total iteration:", i, "\t Final Cost:", cost)

        return self.params, self.loss

    def predict(self, X, model):
        params = model.params
        yhat_all = model.forward(X)
        sample_size = X.shape[1]
        preds = np.array([np.argmax(yhat_all[:, i].T) for i in range(sample_size)])
        return preds, yhat_all

    def accuracy_score(self, y, yhat):
        sample_size = y.shape[1]
        y_true = [np.argmax(y[:, i]) for i in range(sample_size)]
        y_true = np.array(gercekler)
        y_pred = [np.argmax(yhat[:, i]) for i in range(sample_size)]
        y_pred = np.array(tahminler)
        acc = np.round((gercekler_arr == tahminler_arr).sum() / sample_size, 2)
        return acc

    def plot_loss(self, costs):
        plt.plot(range(len(costs)), costs)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss for each epoch")
        plt.show()


def read_raw_data() -> tuple:
    X, y = load_iris(return_X_y=True)
    X = X
    y = y
    return X, y


def split_data(X, y, reshape=False) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=.2, random_state=74)
    X_train, X_test = X_train.T, X_test.T,
    if reshape:
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

    y_train, y_test = y_train.T, y_test.T

    return X_train, X_test, y_train, y_test


def encode_label(data: np.ndarray) -> np.ndarray:
    labels = []
    for i in range(len(data)):
        if data[i] == 0:
            labels.append(np.array([0, 0, 1]))
        elif data[i] == 1:
            labels.append(np.array([0, 1, 0]))
        elif data[i] == 2:
            labels.append(np.array([1, 0, 0]))
        else:
            raise Exception("error:", data[i])
    return np.array(labels)


X, y = read_raw_data()
y = encode_label(y)
X_train, X_test, y_train, y_test = split_data(X, y, reshape=False)

print("X_train.shape:", X_train.shape, "\ty_train.shape:", y_train.shape)

layers = [4, 30, 3]  # input layer :4 neuron, hidden layer 30 neuron, output layer 3 neuron
nn = TwoLayerNeuralNetwork(layers)
model_params, costs = nn.fit(X_train, y_train, lr=5e-5, epochs=10 ** 5)
preds, yhat_all = nn.predict(X_test, nn)
acc = nn.accuracy_score(y_test, yhat_all)
nn.plot_loss(costs)
