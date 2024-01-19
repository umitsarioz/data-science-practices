import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 200)


def check_file_exist(filepath: str) -> bool:
    res = os.path.exists(filepath)
    print(f"{filepath} is exist:", res)
    return res


def read_raw_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def grab_columns(df: pd.DataFrame, car_th=20, num_cat_th=10) -> dict:
    cat_cols = [c for c in df.columns if df[c].dtype == 'O']
    num_cols = [c for c in df.columns if df[c].dtype != 'O']

    num_but_cat_cols = [c for c in num_cols if df[c].nunique() < num_cat_th]
    num_cols = [c for c in num_cols if c not in num_but_cat_cols]

    cat_but_car = [c for c in cat_cols if df[c].nunique() > car_th]
    cat_cols = [c for c in cat_cols + num_but_cat_cols if c not in cat_but_car]

    print("----------------------- First Sight ------------------------")
    print(f"Observation count: {df.shape[0]} \t | \t Variables : {df.shape[1]}")
    print("----------------------- Summary of Column Count ------------------------")
    print(f"Categorical cols: {len(cat_cols)}")
    print(f"Numerical Cols: {len(num_cols)}")
    print(f"High Cardinality Cols. {len(cat_but_car)}")
    print(f"Categorical as Numeric: {len(num_but_cat_cols)}")
    print("----------------------- Summary of Column Name ------------------------")
    print(f"Categorical cols: {cat_cols}")
    print(f"Numerical Cols: {num_cols}")
    print(f"High Cardinality Cols: {cat_but_car}")
    print(f"Categorical as Numeric: {num_but_cat_cols}")

    dct = {'categorical': cat_cols, 'numerical': num_cols,
           'cardinality': cat_but_car, 'categorical_as_numeric': num_but_cat_cols}

    print("\n[info] * Dictionary Keys:", list(dct.keys()))
    return dct


def show_summary_table(df: pd.DataFrame, columns: List[str] = None) -> None:
    if columns is None:
        columns = df.columns.tolist()

    df = pd.DataFrame([{'col': col,
                        'col_type': df[col].dtype,
                        'n_unique': df[col].nunique(),
                        'unique_values': df[col].unique(),
                        'n_missing': df[col].isnull().sum()}
                       for col in columns]) \
        .sort_values('n_unique', ascending=False) \
        .reset_index(drop=True)
    print(df, end="\n\n")


def show_feature_summary(df: pd.DataFrame, col: str, target: str) -> None:
    print(f"------------ Summary :: {col} -------------\n")
    df = pd.concat([df[col].value_counts(ascending=True),
                    df[col].value_counts(normalize=True, ascending=True),
                    df.groupby(col)[target].mean()],
                   keys=['n_observation', 'observation_ratio', 'target_mean'],
                   names=['Columns'],
                   axis=1)
    print(df, end="\n\n")


def get_outliers_columns(df: pd.DataFrame, cols: List[str] = None) -> List[str]:
    if cols is None:
        cols = df.columns.tolist()

    outlier_cols = [col for col in cols if df[col].dtype in ['int', 'float'] and df[col].nunique() > 10]
    return outlier_cols


def calculate_iqr_boundaries(df: pd.DataFrame, col: str, q1_ratio: float, q3_ratio: float) -> tuple:
    q1 = df[col].quantile(q1_ratio)
    q3 = df[col].quantile(q3_ratio)
    iqr = q3 - q1
    low_limit = q1 - (1.5 * iqr)
    up_limit = q3 + (1.5 * iqr)
    return low_limit, up_limit


def check_outliers(df: pd.DataFrame, col: str, q1_ratio: float, q3_ratio: float) -> bool:
    low_limit, up_limit = calculate_iqr_boundaries(df, col, q1_ratio, q3_ratio)
    df_outlier = df[(df[col] < low_limit) | (df[col] > up_limit)]
    return df_outlier.any(axis=None)


def check_outliers_for_given_columns(df: pd.DataFrame, q1_ratio: float, q3_ratio: float,
                                     cols: List[str] = None) -> None:
    if cols is None:
        cols = get_outliers_columns(df)

    for col in cols:
        is_outlier_exist = check_outliers(df, col, q1_ratio=q1_ratio, q3_ratio=q3_ratio)
        print(col, " is outlier exist:", is_outlier_exist)


def get_binary_columns(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if df[col].dtype == 'O' and df[col].nunique() == 2]
    return cols


def binary_encoder(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    temp_df = df.copy()
    if cols is None:
        cols = get_binary_columns(df)

    for col in cols:
        le = LabelEncoder()
        temp_df[col] = le.fit_transform(temp_df[col])
    return temp_df


def get_ohe_cols(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if df[col].dtype == 'O' and df[col].nunique() < 10]
    return cols


def one_hot_encoder(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    if cols is None:
        cols = get_ohe_cols(df)

    X = df[cols]
    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    ohe.fit(X)

    df_ohe = pd.DataFrame(ohe.transform(X), columns=ohe.get_feature_names_out())
    df_final = pd.concat([df, df_ohe], axis=1).drop(columns=ohe.feature_names_in_, axis=1)

    return df_final


def train_and_evaluate(df: pd.DataFrame, verbose=True) -> dict:
    # Prepare dataframes for training
    X = df.drop(columns=['customerID', 'Churn'])
    y = df[["Churn"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=74)

    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)

    # Create & Train Model
    model = RandomForestClassifier(random_state=74)
    model.fit(X_train_scaled, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    df_scores = pd.DataFrame.from_dict(classification_report(y_true=y_test, y_pred=y_pred, output_dict=True))
    auc_score = roc_auc_score(y_test, y_pred)
    if verbose:
        print(df_scores)
        print(f"Auc Score: {auc_score:.2f}")

    dct = {'X_train': X_train,
           'y_train': y_train,
           'X_test': X_test,
           'y_test': y_test,
           'scaler': scaler,
           'model': model,
           'classification_report': df_scores,
           'auc_score': roc_auc_score(y_test, y_pred)}
    return dct


def plot_classification_report(df_classification_report: pd.DataFrame, title='Classification Report',
                               cmap='RdBu') -> None:
    data = df_classification_report.T.iloc[:-2, :-1]
    class_names = data.index.tolist()
    class_names.remove('accuracy')
    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    font1 = {'family': 'serif', 'color': 'blue', 'size': 20}
    font2 = {'family': 'serif', 'color': 'darkred', 'size': 15}

    sns.heatmap(data, xticklabels=xticklabels, annot=True, cmap=cmap)
    plt.xlabel(xlabel, fontdict=font2)
    plt.ylabel(ylabel, fontdict=font2)
    plt.title(title, fontdict=font1)
    plt.show()


def plot_importance(model, features):
    num = len(features.columns)
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()


def main():
    filepath = "datasets/data.csv"
    is_file_exist = check_file_exist(filepath=filepath)
    if is_file_exist:
        df_raw = read_raw_data(filepath=filepath)
        _ = grab_columns(df=df_raw)
        show_summary_table(df=df_raw)
        # convert numeric but string column to numeric format
        df_raw['TotalCharges'] = pd.to_numeric(df_raw["TotalCharges"], errors='coerce')
        df_raw["Churn"] = df_raw["Churn"].apply(lambda val: 0 if val == 'No' else 1)
        show_summary_table(df=df_raw)
        # Fill Missings
        df_raw.loc[df_raw[df_raw['TotalCharges'].isna()].index, 'TotalCharges'] = df_raw["TotalCharges"].mode().iloc[0]
        show_summary_table(df=df_raw)

        # Check outliers but before get new columns information
        cols_dct = grab_columns(df=df_raw)
        num_cols = cols_dct.get('numerical')
        check_outliers_for_given_columns(df=df_raw, cols=num_cols, q1_ratio=.25, q3_ratio=.75)

        # Check ratio of values for each categoric column.
        cat_cols = cols_dct.get('categorical')
        for col in cat_cols:
            show_feature_summary(df=df_raw, col=col, target='Churn')

        # All column values ratios are okay in my opinion. So i only use Binary & OneHot encoding.
        df_encoded = binary_encoder(df=df_raw)
        df_encoded = one_hot_encoder(df=df_encoded)
        show_summary_table(df_encoded)

        # Let's train a model basic classification model
        results_dct = train_and_evaluate(df=df_encoded, verbose=True)
        plot_classification_report(df_classification_report=results_dct.get('classification_report'))
        plot_importance(model=results_dct.get('model'), features=results_dct.get('X_train'))
    else:
        raise FileNotFoundError(filepath)


if __name__ == "__main__":
    main()

