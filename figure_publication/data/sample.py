import os
import numpy as np
import pandas as pd

# total train pred path
TOTAL_TRAIN_PRED_PATH = '/home/sk77/PycharmProjects/omg_database_publication/data/active_learning/pareto_greedy/uncertainty_pred/current_batch_3/total_prediction_results.csv'

# total test pred path
TOTAL_TEST_PRED_PATH = '/home/sk77/PycharmProjects/omg_database_publication/figure_publication/data/pred_dir/AL3_test_pred_total_merged.csv'

# data used to train the model
TRAIN_AL3_DATA_PATH = '/home/sk77/PycharmProjects/omg_database_publication/data/active_learning/pareto_greedy/OMG_train_batch_3_chemprop_with_reaction_id.csv'

# rdkit data
TRAIN_RDKIT_PATH = '/home/sk77/PycharmProjects/omg_database_publication/run_calculation/rdkit_properties/results_csv/OMG_train/total_results/total_results.csv'
TEST_RDKIT_PATH = '/home/sk77/PycharmProjects/omg_database_publication/run_calculation/rdkit_properties/results_csv/OMG_test/total_results/total_results.csv'

# rdkit cols
rdkit_cols = ['molecular_weight', 'logP', 'qed', 'TPSA', 'normalized_monomer_phi', 'normalized_backbone_phi']
rdkit_cols_Boltzmann_average = [f'{col}_Boltzmann_average' for col in rdkit_cols]


if __name__ == '__main__':
    """
    classify
    2 -> test data
    1 -> train data (used to train a model -> converged ones)
    0 -> else
    """
    # load total train pred .csv
    df_total_train_pred = pd.read_csv(TOTAL_TRAIN_PRED_PATH)
    print(df_total_train_pred.shape)  # (12,870,976, 21)

    # merge train RDKIT data
    df_total_train_rdkit = pd.read_csv(TRAIN_RDKIT_PATH)
    df_total_train_rdkit_sorted = df_total_train_rdkit.sort_values(by='reaction_id', ascending=True)
    compare = df_total_train_pred['reaction_id'].to_numpy() == df_total_train_rdkit_sorted['reaction_id'].to_numpy()
    print(np.all(compare))
    df_total_train_pred[rdkit_cols_Boltzmann_average] = df_total_train_rdkit_sorted[rdkit_cols].to_numpy()
    print(df_total_train_pred.shape)  # (12,870,976, 27)

    # data used to train a model (converged ones)
    df_used_in_train = pd.read_csv(TRAIN_AL3_DATA_PATH)
    print(df_used_in_train.shape)  # (32,529, 27) -> including 2D chemistry
    isin = df_total_train_pred['reaction_id'].isin(df_used_in_train['reaction_id'])
    isin_arr = isin.values.astype(np.float64)
    df_total_train_pred['classify'] = np.zeros(shape=(df_total_train_pred.shape[0])) + isin_arr

    # load total test pred .csv
    df_total_test_pred = pd.read_csv(TOTAL_TEST_PRED_PATH)
    df_total_test_pred = df_total_test_pred.sort_values(by='reaction_id', ascending=True)  # random order
    print(df_total_test_pred.shape)  # (15,155, 21)

    # merge test RDKIT data
    df_total_test_rdkit = pd.read_csv(TEST_RDKIT_PATH)
    df_total_test_rdkit_sorted = df_total_test_rdkit.sort_values(by='reaction_id', ascending=True)
    compare = df_total_test_pred['reaction_id'].to_numpy() == df_total_test_rdkit_sorted['reaction_id'].to_numpy()
    print(np.all(compare))
    df_total_test_pred[rdkit_cols_Boltzmann_average] = df_total_test_rdkit_sorted[rdkit_cols].to_numpy()
    print(df_total_test_pred.shape)
    df_total_test_pred['classify'] = np.ones(shape=(df_total_test_pred.shape[0])) * 2

    # merge
    df_total = pd.concat([df_total_train_pred, df_total_test_pred], axis=0)
    print(df_total.shape)
    print(df_total['classify'].value_counts())

    # sample
    df_total_sampled = df_total.sample(n=100000, random_state=42, replace=False)
    df_total_sampled.to_csv('sampled_predictions_AL3_100K.csv', index=False)
