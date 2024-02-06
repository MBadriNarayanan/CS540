import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_csv(filepath):
    dataframe = pd.DataFrame()
    data_dict = {}
    df = pd.read_csv(filepath)
    for idx, row in df.iterrows():
        year = int(row["Category"].split("-")[0])
        if 1855 <= year <= 2021:
            days = row["Annual number of days"]
            if days != "-":
                data_dict[year] = int(days)
    dataframe["year"] = data_dict.keys()
    dataframe["days"] = data_dict.values()
    dataframe.to_csv("hw5.csv", index=False)


def plot_fig(file_path):
    df = pd.read_csv(file_path)
    plt.plot(df["year"], df["days"])
    plt.xlabel("Year")
    plt.ylabel("Number of Frozen Days")
    plt.savefig("plot.jpg")


def contruct_vectors(file_path):
    df = pd.read_csv(file_path)
    df["constant"] = [1 for _ in range(df.shape[0])]
    X = np.array(df[["constant", "year"]]).astype(dtype=np.int64)
    y = np.array(df["days"].to_list()).astype(dtype=np.int64)
    return X, y


def compute_matrix_product(X, y):
    Z = X.T @ X
    Z_inverse = np.linalg.inv(Z)
    pseudo_inverse = Z_inverse @ X.T
    beta = pseudo_inverse @ y
    Z = Z.astype(dtype=np.int64)
    return Z, Z_inverse, pseudo_inverse, beta


def prediction(beta):
    X_test = np.array([1, 2022]).astype(dtype=np.int64)
    return beta @ X_test


def get_symbol(beta):
    if beta[1] > 0:
        symbol = ">"
        reason = "This means the linear function has a positive slope, (i.e) y has a positive corelation with x."
    elif beta[1] < 0:
        symbol = "<"
        reason = "This means the linear function has a neagtive slope, (i.e) y has a negative corelation with x."
    else:
        symbol = "="
        reason = "This means the linear function has a no slope, (i.e) y equals beta[0] for all values of x."
    return symbol, reason


def get_model_limitation(beta):
    return -beta[0] / beta[1]


def generate_output(file_path):
    plot_fig(file_path)
    X, y = contruct_vectors(file_path)
    Z, Z_inverse, pseudo_inverse, beta = compute_matrix_product(X, y)
    y_pred = prediction(beta)
    symbol, reason = get_symbol(beta)
    X_star = get_model_limitation(beta)
    X_star_reason = "The prediction made by the Linear Regressor is not a compelling prediction. It estimates that Lake Mendota will no longer freeze in the year 2455 which does not correlate with the trends as per the data. Also this might be possible only if Global warning rates increase tremendously!"

    print("Q3a:")
    print(X)

    print("Q3b:")
    print(y)

    print("Q3c:")
    print(Z)

    print("Q3d:")
    print(Z_inverse)

    print("Q3e:")
    print(pseudo_inverse)

    print("Q3f:")
    print(beta)

    print("Q4: {}".format(y_pred))

    print("Q5a: {}".format(symbol))
    print("Q5b: {}".format(reason))

    print("Q6a: {}".format(X_star))
    print("Q6b: {}".format(X_star_reason))


if __name__ == "__main__":
    file_path = sys.argv[1]
    generate_output(file_path)
