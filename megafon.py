import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)


def train_test_prep(df, train=True):
    """
    Train and test sets preparation function:
        - types convertion to save memory
        - drop unnecessary column
        - date format
    """

    df = df.copy()
    df = df.astype({
        'id': 'uint32',
        'vas_id': 'uint8'
    })

    if train:
        df = df.astype({'target': 'uint8'})

    df = df.drop(columns=['Unnamed: 0'])
    #     df = format_date(df, 'buy_time')

    return df


def features_prep(features_dask_df, ids_train_list, ids_test_list):
    """
    Features dataset cut function:
        - leave only actual ids which are contained in train and test datasets
        - drop Unnamed column
        - drop constant columns
        - convert types
        - date format
        - reset index
        - return pandas df instead of dask df
    """

    ids_list = np.unique(np.append(ids_train_list, ids_test_list))
    features_pd_df = features_dask_df.loc[features_dask_df['id'].isin(ids_list)].compute()
    features_pd_df = features_pd_df.drop(columns=['Unnamed: 0'], axis=1)

    features_pd_df = features_pd_df.astype({'id': 'uint32'})

    features_pd_df.reset_index(drop=True, inplace=True)

    return features_pd_df


def prepare_df_with_features(df, features, train=True):
    """
    function to combine train/test datasets with features
    working with following algorithm:
    - create an empty "ready_features" list which will be filled with values from features df
    according to user id and closest profile in time (indexes are the same like in "data" df)
    - convert "ready_features" list into dataframe
    - merge 2 dataframes by index
    - if train parameter is true move 'target' column to the end
    """
    ready_features = []

    for idx in df.index:
        processing_series = features.loc[(features['id'] == df.at[idx, 'id'])]
        if processing_series.shape[0] == 1:
            ready_features.append(processing_series.values[0])
        else:
            check_diff = np.abs(
                processing_series['buy_time'] - df.at[idx, 'buy_time'])
            nearest_series_id = check_diff[check_diff == np.min(check_diff)].index[0]
            ready_features.append(processing_series.loc[nearest_series_id].values)

    ready_features_df = pd.DataFrame(ready_features, columns=features.columns)
    merged_df = pd.merge(df, ready_features_df.iloc[:, 3:], left_index=True, right_index=True)

    if train:
        merged_df = merged_df[[col for col in merged_df.columns if col != 'target'] + ['target']]

    return merged_df