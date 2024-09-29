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

    # append polymerization idx
    df_total_sorted = df_total.sort_values(by='reaction_id', ascending=True)
    total_reaction_id = df_total_sorted['reaction_id'].to_numpy()
    print(total_reaction_id[-5:])

    # add polymerization mechanism idx
    df_train = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/data/active_learning/OMG_train.csv')
    df_test = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/data/active_learning/OMG_test.csv')

    # train overlap
    print(df_train.shape)
    df_train_overlap = df_train[df_train['reaction_id'].isin(total_reaction_id)]
    print(df_train_overlap.shape)

    # test overlap
    print(df_test.shape)
    df_test_overlap = df_test[df_test['reaction_id'].isin(total_reaction_id)]
    print(df_test_overlap.shape)

    # polymerization idx
    df_polymerization_idx = pd.concat([df_train_overlap, df_test_overlap], axis=0)
    df_polymerization_idx = df_polymerization_idx.sort_values(by='reaction_id', ascending=True)

    # check
    compare = total_reaction_id == df_polymerization_idx['reaction_id'].to_numpy()
    print(np.all(compare))

    # append
    df_total_sorted['polymerization_mechanism_idx'] = df_polymerization_idx['polymerization_mechanism_idx'].to_numpy()

    # save
    df_total_sorted.to_csv('./total_prediction_with_polymerization_idx.csv', index=False)

    # sample with diversity
    df_sampled = pd.DataFrame()
    for polymerization_mechanism_idx in [13, 14, 15]:
        df_sub = df_total_sorted[df_total_sorted['polymerization_mechanism_idx'] == polymerization_mechanism_idx]
        df_sampled = pd.concat([df_sampled, df_sub], axis=0)
        print(f'{polymerization_mechanism_idx} is done', flush=True)

    for polymerization_mechanism_idx in [10, 11, 12, 16]:
        df_sub = df_total_sorted[df_total_sorted['polymerization_mechanism_idx'] == polymerization_mechanism_idx]
        df_sampled = pd.concat([df_sampled, df_sub], axis=0)
        print(f'{polymerization_mechanism_idx} is done', flush=True)

    for polymerization_mechanism_idx in [7, 9, 17]:
        df_sub = df_total_sorted[df_total_sorted['polymerization_mechanism_idx'] == polymerization_mechanism_idx]
        df_sampled = pd.concat([df_sampled, df_sub], axis=0)
        print(f'{polymerization_mechanism_idx} is done', flush=True)

    for polymerization_mechanism_idx in [5, 6, 8]:
        df_sub = df_total_sorted[df_total_sorted['polymerization_mechanism_idx'] == polymerization_mechanism_idx]
        df_sub = df_sub.sample(n=10000, random_state=42, replace=False)
        df_sampled = pd.concat([df_sampled, df_sub], axis=0)
        print(f'{polymerization_mechanism_idx} is done', flush=True)

    for polymerization_mechanism_idx in [2, 4]:
        df_sub = df_total_sorted[df_total_sorted['polymerization_mechanism_idx'] == polymerization_mechanism_idx]
        df_sub = df_sub.sample(n=10000, random_state=42, replace=False)
        df_sampled = pd.concat([df_sampled, df_sub], axis=0)
        print(f'{polymerization_mechanism_idx} is done', flush=True)

    for polymerization_mechanism_idx in [1, 3]:
        df_sub = df_total_sorted[df_total_sorted['polymerization_mechanism_idx'] == polymerization_mechanism_idx]
        df_sub = df_sub.sample(n=30000, random_state=42, replace=False)
        df_sampled = pd.concat([df_sampled, df_sub], axis=0)
        print(f'{polymerization_mechanism_idx} is done', flush=True)

    # random sample
    df_sampled = df_sampled.sample(frac=1.0, random_state=42)
    df_sampled.to_csv('./diversity_sampled_predictions_AL3.csv', index=False)
