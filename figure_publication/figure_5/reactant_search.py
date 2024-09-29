import pandas as pd


# def find_reactants(reaction_id):
#     """
#     This function identifies OMG reactants to OMG polymers
#     """
#     df_OMG = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/data/OMG_methyl_terminated_polymers.csv')
#     df_target = df_OMG[df_OMG['reaction_id'] == reaction_id]
#
#     reactant_1 = df_target['reactant_1'].tolist()[0]
#     reactant_2 = df_target['reactant_2'].tolist()[0]
#     polymerization_mechanism_idx = df_target['polymerization_mechanism_idx'].tolist()[0]
#
#     return polymerization_mechanism_idx, reactant_1, reactant_2


if __name__ == '__main__':
    # water soluble reaction id
    df_OMG = pd.read_csv('/home/sk77/PycharmProjects/omg_database_publication/data/OMG_methyl_terminated_polymers.csv')

    # test id
    # reaction_id_list = [3655215, 1833935, 1827798, 3666946]
    # reaction_id_list = [3999522, 3999558, 3999551, 4000810, 3999554]
    # reaction_id_list = [9258272, 3992862, 1828344, 4557514, 9213018, 1828625, 9212632, 9213877 ,1826855, 4050921]
  #   reaction_id_list = [ 4044344,  3392146 , 3512923 ,11633822,  1338591, 11349772 , 3389881,  3727032,
  # 1804396,  1130246, 11702654,  1196512,  3576367,  3375633,  4038224,  4020514,
  # 7537291,  8583494,  3470675, 10584103,  1886226,  4034228,  1424311,  3575227,
  # 3424747,  3480698,  2100267,  4021412,  9665023,  3386927,  4020962,  9253597,
  # 4028432,  3385174, 11552035,  4044739,  3512709,  9315023, 12700999,  3073896,
  # 3478587,  4013312,  8563225,  4019288, 11811750,  4046704,  4007601,  1798564,
  # 4005577,  3428872,  1754158,  4039088]
  #   reaction_id_list = [9274372, 9274539]
  #   reaction_id_list = [11888541 ,11201183 ,11909368 ,11263731]

    ##### final - epsilon 0.1 (before)
    # region 1
    # reaction_id_list = [1772481, 3655215]  # chi: 0.87 / 1.49

    # region 2
    # reaction_id_list = [3999522, 9258272]  # chi: 0.86 / 1.52

    # region 3
    # reaction_id_list = [4044344, 9274539]  # chi: 0.87 / 1.46

    # region 4
    # reaction_id_list = [11888541, 6215927]  # chi: 0.87 / 1.50
    # for reaction_id in reaction_id_list:
    #     df_target = df_OMG[df_OMG['reaction_id'] == reaction_id]
    #     methyl_terminated_product = df_target['methyl_terminated_product'].tolist()[0]
    #     reactant_1 = df_target['reactant_1'].tolist()[0]
    #     reactant_2 = df_target['reactant_2'].tolist()[0]
    #     polymerization_mechanism_idx = df_target['polymerization_mechanism_idx'].tolist()[0]
    #     print(polymerization_mechanism_idx, methyl_terminated_product, reactant_1, reactant_2)

    ##### final - epsilon 0.2 - search for region 3 high chi value (4002812 instead of 9274539, rest is the same)
    # reaction_id_list = [ 9274372,  5083145,  7703572,  4002812,  9274539,  9239421,  6468624, 12219401, 6634214]

    # region 3
    # reaction_id_list = [4044344, 4002812]  # chi: 0.87 / 1.49
    for reaction_id in reaction_id_list:
        df_target = df_OMG[df_OMG['reaction_id'] == reaction_id]
        methyl_terminated_product = df_target['methyl_terminated_product'].tolist()[0]
        reactant_1 = df_target['reactant_1'].tolist()[0]
        reactant_2 = df_target['reactant_2'].tolist()[0]
        polymerization_mechanism_idx = df_target['polymerization_mechanism_idx'].tolist()[0]
        print(polymerization_mechanism_idx, methyl_terminated_product, reactant_1, reactant_2)
