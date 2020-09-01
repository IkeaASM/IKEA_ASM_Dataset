# Dataset IDs
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

def get_scenes():
    return ['01', '02', '04', '05', '06', '07', '08', '09', '10', '11']

def get_cams():
    return ['dev1', 'dev2', 'dev3']

def get_male_ids():
    return [
        1,
        2,
        4,
        5,
        7,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        25,
        26,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        48,
    ]

def get_female_ids():
    return [
        3,
        6,
        8,
        12,
        17,
        23,
        24,
        27,
        44,
        45,
        46,
        47,
    ]

def get_adult_ids():
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
    ]

def get_child_ids():
    return [
        45,
        46,
        47,
        48,
    ]

def get_calibration_ids():
    return [
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
    ]

def get_room_ids():
    return [
        1,
        2,
        3,
        4,
        5,
    ]

def get_office_room_ids():
    return [
        1,
        2,
    ]

def get_flying_room_ids():
    return [
        3,
    ]

def get_living_room_ids():
    return [
        4,
    ]

def get_music_room_ids():
    return [
        5,
    ]

# No people other than assembler
def get_unpeopled_room_ids():
    return [
        1,
        3,
        5,
    ]

# People other than assembler in the background
def get_peopled_room_ids():
    return [
        2,
        4, # Many people moving in background
    ]