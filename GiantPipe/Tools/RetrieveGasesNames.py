def RetrieveGasesNames(gas_name, gas_id):
    """Convert between gas name and gas ID code. 
        Takes whatever value you have and outputs 
        all infomation about the gas ."""

    if (gas_name == 'NH3') or (gas_id == 11): 
        gas_name = r'NH$_{3}$'
        gas_id = 11

    if (gas_name == 'PH3') or (gas_id == 28):
        gas_name = r'PH$_{3}$'
        gas_id = 28 

    if (gas_name == 'C2H2') or (gas_id == 26): 
        gas_name = r'C$_{2}$H$_{2}$'
        gas_id = 26 

    if (gas_name == 'C2H4') or (gas_id == 32):
        gas_name = r'C$_{2}$H$_{4}$'
        gas_id = 32

    if (gas_name == 'C2H6') or (gas_id == 27): 
        gas_name = r'C$_{2}$H$_{6}$'
        gas_id = 27

    if (gas_name == 'C4H2') or (gas_id == 30):
        gas_name = r'C$_{4}$H$_{2}$'
        gas_id = 30

    if (gas_name == 'H2') or (gas_id == 39): 
        gas_name = r'H$_{2}$'
        gas_id = 39

    if (gas_name == 'He') or (gas_id == 40):
        gas_name = r'He'
        gas_id = 40

    if (gas_name == 'CH4') or (gas_id == 6): 
        gas_name = r'CH$_{4}$'
        gas_id = 6

    return gas_name, gas_id