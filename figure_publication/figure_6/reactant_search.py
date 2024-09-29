import pandas as pd


def find_reactants(reaction_id):
    """
    This function identifies OMG reactants to OMG polymers
    """
    df_OMG = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/data/OMG_methyl_terminated_polymers.csv')
    df_target = df_OMG[df_OMG['reaction_id'] == reaction_id]

    reactant_1 = df_target['reactant_1'].tolist()[0]
    reactant_2 = df_target['reactant_2'].tolist()[0]

    return reactant_1, reactant_2


if __name__ == '__main__':
    # water soluble reaction id
    reaction_id_list = [1545345, 8572016, 3993296, 3663471]  # mechanism 1, 1, 12, 8
    for reaction_id in reaction_id_list:
        print(find_reactants(reaction_id))

    # chloroform soluble reaction id
    reaction_id_list = [4053479, 1748252, 3641700, 3668722]  # mechanism 7, 6, 8, 8
    for reaction_id in reaction_id_list:
        print(find_reactants(reaction_id))
