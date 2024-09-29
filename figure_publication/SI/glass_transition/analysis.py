import pandas as pd

from rdkit import Chem

if __name__ == '__main__':
    df = pd.read_csv('./experimental_Tg.csv')
    smi_list = df['SMILES (Atoms Ce and Th are placeholders for head and tail information, respectively)'].to_list()
    Tg_experiment_K_arr = df['Experiment Tg (K)'].to_numpy()

    # check smi
    for smi in smi_list:
        # default
        ce_cnt = 0
        th_cnt = 0

        # add
        ce_cnt += smi.count('[Ce]')
        th_cnt += smi.count('[Th]')

        # check
        if ce_cnt != 1 or th_cnt != 1:
            raise ValueError

    # replace with asterisk
    polymer_CRU_list = list()
    for smi in smi_list:
        # replace
        polymer_CRU = smi.replace('[Ce]', '*')
        polymer_CRU = polymer_CRU.replace('[Th]', '*')

        # canon smi
        canon_polymer_CRU = Chem.CanonSmiles(polymer_CRU)

        # append
        polymer_CRU_list.append(canon_polymer_CRU)

    # save pd
    df_save = pd.DataFrame(columns=['canon_p_smi', 'experimental_Tg'])
    df_save['canon_p_smi'] = polymer_CRU_list
    df_save['experimental_Tg'] = Tg_experiment_K_arr
    df_save.to_csv('./processed_Tg.csv', index=False)
