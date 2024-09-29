from rdkit import Chem


def calculate_characteristic_ratio(
        number_of_backbone_atoms, number_of_ethyl_branches, square_end_to_end_distance_per_mass):
    """
    This function calculates Flory characteristic ratio of PEB.
    Assume the carbon single bond length is 1.54 angstroms.
    """
    carbon_sing_bond_length = 1.54  # angstroms
    c_inf = square_end_to_end_distance_per_mass / carbon_sing_bond_length**2
    c_inf *= (14 + 28 * number_of_ethyl_branches / number_of_backbone_atoms)

    return c_inf


if __name__ == '__main__':
    # # PEB-2. "2" ethyl branches per "100" backbone atoms
    # smi = '*CCCCCC(CC)CCCCCCCCCCCC(CC)CCCCCCCC*'
    # number_carbon_in_branches = 4
    # number_of_carbons_to_add = 100 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=100, number_of_ethyl_branches=2,
    #                                      square_end_to_end_distance_per_mass=1.21))
    # exit()

    # # PEB-4.6 -> multiply 10. "46" ethyl branches per "1,000" backbone atoms
    # smi = '*CCCCCC(CC)CCCCCCCCCCCC(CC)CCCCCCCC*' * 23  # 46 ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = 46 * 2
    # number_of_carbons_to_add = 1000 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=1000, number_of_ethyl_branches=46,
    #                                      square_end_to_end_distance_per_mass=1.15))
    # exit()

    # # PEB-7.1 -> multiply 10. "71" ethyl branches per "1,000" backbone atoms
    # smi = '*CCCCCC(CC)CCCCCCCC*' * 71  # 71 ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = 71 * 2
    # number_of_carbons_to_add = 1000 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=1000, number_of_ethyl_branches=71,
    #                                      square_end_to_end_distance_per_mass=1.05))
    # exit()

    # # PEB-9.5 -> multiply 10. "95" ethyl branches per "1,000" backbone atoms
    # smi = '*CCCC(CC)CCCC*' * 95  # 95 ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = 95 * 2
    # number_of_carbons_to_add = 1000 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=1000, number_of_ethyl_branches=95,
    #                                      square_end_to_end_distance_per_mass=1.05))
    # exit()

    # # PEB-10.6 -> multiply 10. "106" ethyl branches per "1,000" backbone atoms
    # smi = '*CCCC(CC)CCCC*' * 106  # 106 ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = 106 * 2
    # number_of_carbons_to_add = 1000 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=1000, number_of_ethyl_branches=106,
    #                                      square_end_to_end_distance_per_mass=1.06))
    # exit()

    #
    # # PEB-11.7 -> multiply 10. "117" ethyl branches per "1,000" backbone atoms
    # smi = '*CCCC(CC)CCCC*' * 117  # 117 ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = 117 * 2
    # number_of_carbons_to_add = 1000 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=1000, number_of_ethyl_branches=117,
    #                                      square_end_to_end_distance_per_mass=0.952))
    # exit()

    # # PEB-17.6 -> multiply 10. "176" ethyl branches per "1,000" backbone atoms
    # smi = '*CC(CC)CC*' * 176  # 176 ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = 176 * 2
    # number_of_carbons_to_add = 1000 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=1000, number_of_ethyl_branches=176,
    #                                      square_end_to_end_distance_per_mass=0.913))
    # exit()

    # # PEB-24.6 -> multiply 10. "246" ethyl branches per "1,000" backbone atoms: impossible to have a minimum of two ethylene units between butenes
    # number_of_branches = 246
    # smi = '*CC(CC)CC*' * number_of_branches  # ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = number_of_branches * 2
    # number_of_carbons_to_add = 1000 - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)

    # PEB-32 -> "32" ethyl branches per "100" backbone atoms: impossible to have a minimum of two ethylene units between butenes
    # number_of_branches = 32
    # number_of_backbone_atoms = 100
    # smi = '*CC(CC)*' * number_of_branches  # ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = number_of_branches * 2
    # number_of_carbons_to_add = number_of_backbone_atoms - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)

    # PEB-39.3 -> "393" ethyl branches per "1,000" backbone atoms: impossible to have a minimum of two ethylene units between butenes
    # number_of_branches = 393
    # number_of_backbone_atoms = 1000
    # smi = '*CC(CC)*' * number_of_branches  # ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = number_of_branches * 2
    # number_of_carbons_to_add = number_of_backbone_atoms - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)

    # PEB-40.9 -> "409" ethyl branches per "1,000" backbone atoms: impossible to have a minimum of two ethylene units between butenes
    # number_of_branches = 409
    # number_of_backbone_atoms = 1000
    # smi = '*CC(CC)*' * number_of_branches  # ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = number_of_branches * 2
    # number_of_carbons_to_add = number_of_backbone_atoms - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(final_smi)
    # print(final_smi.count('C'))

    # alt-PEP -> alternating poly(ethylene-co-propylene)
    # smi = '*CCCC(C)*'
    # smi = '*CCC(C)C*'  # significant difference?
    # final_smi = Chem.CanonSmiles(smi)
    # c_inf = 0.834 / 1.54**2 * 14 * (5/4)
    # print(c_inf)
    # exit()

    # alt-PEB -> alternating poly(ethylene-co-1-butene)
    # smi = '*CCCC(CC)*'
    # smi = '*CCC(CC)C*'  # significant difference?
    # final_smi = Chem.CanonSmiles(smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=4, number_of_ethyl_branches=1,
    #                                      square_end_to_end_distance_per_mass=0.692))
    # print('hi')
    # exit()

    # HHPP -> head-to-head polypropylene -- https://www.chem.tamu.edu/class/majors/chem470/Polypropylene.html
    # smi = '*CC(C)C(C)C*'
    # final_smi = Chem.CanonSmiles(smi)
    # c_inf = 0.691 / 1.54**2 * 14 * (3/2)
    # print(c_inf)
    # exit()

    # poly(ethylethylene) (PEE) -- https://www.ebi.ac.uk/chebi/searchId.do?printerFriendlyView=true&locale=null&chebiId=53358&viewTermLineage=null&structureView=applet&
    # smi = '*C(CC)C*'
    # final_smi = Chem.CanonSmiles(smi)
    # print(calculate_characteristic_ratio(number_of_backbone_atoms=2, number_of_ethyl_branches=1,
    #                                      square_end_to_end_distance_per_mass=0.507))
    # print('hi')
    # exit()

    # PVCH - poly(vinylcyclohexane) -- https://www.sigmaaldrich.com/US/en/product/aldrich/677388
    # smi = '*CC(C1CCCCC1)*'
    # final_smi = Chem.CanonSmiles(smi)
    # print(final_smi)
    c_inf = 0.323 / 1.54**2 * 14 * (4)
    print(c_inf)
    exit()


    # 1,4-PBd https://www.polymersource.ca/index.php?route=product/category&path=2_2183_14_85_2465_477&order=DESC&subtract=1&categorystart=.1-.1.2.1.2&serachproduct=
    # smi = '*C/C=C/C*'

    # PEB-14 -> "14" ethyl branches per "100" backbone atoms
    # number_of_branches = 14
    # number_of_backbone_atoms = 100
    # smi = '*CCCC(CC)CCCC*' * number_of_branches  # ethyl branches
    # smi = smi.replace('**', '')
    # number_carbon_in_branches = number_of_branches * 2
    # number_of_carbons_to_add = number_of_backbone_atoms - (smi.count('C') - number_carbon_in_branches)
    # final_smi = smi[:-1] + 'C' * number_of_carbons_to_add + '*'
    # final_smi = Chem.CanonSmiles(final_smi)
    # print(final_smi)

    # a-PP -> atactic polypropylene
    # smi = '*CC(C)*'

    # PDMS - Polydimethylsiloxane
    # smi = '*O[Si](C)(*)C'

    # Polypropylene oxide : https://en.wikipedia.org/wiki/Polypropylene_glycol
    # smi = 'CC(C*)O*'

    # PVE : 1,2-polybutadiene https://www.sigmaaldrich.com/US/en/search/1%2C2-polybutadiene?focus=products&page=1&perpage=30&sort=relevance&term=1%2C2-polybutadiene&type=product
    # smi = 'C=CC(*)C*'

    # P2MP: 1,4-poly(2-methyl-1,3-pentadiene): https://www.tandfonline.com/doi/full/10.1081/MB-100100370
    # smi = 'C/C(C*)=C/C(*)C'

    # palphaMS : poly(alpha-methylstyrene)
    # smi = 'CC(C1=CC=CC=C1)(*)C*'

    # Poly(methyl acrylate)
    # smi = '*C(C(OC)=O)C*'

    # 1,4polymyrcene https://www.researchgate.net/figure/Scheme-1-Schematic-representation-of-the-various-microstructures-of-polymeric-myrcene_fig1_341260152
    smi = 'C/C(C)=C\CC/C(C*)=C\C*'