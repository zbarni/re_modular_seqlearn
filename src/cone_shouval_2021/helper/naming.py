"""
Naming conventions
"""

name_dict = {
    'timer_exc': 'E_L5',
    'timer_inh': 'I_L5',
    'messenger_exc': 'E_L23',
    'messenger_inh': 'I_L23',
}

def get_pop_name(key, col_id):
    return '{}_{}'.format(name_dict[key], col_id)
