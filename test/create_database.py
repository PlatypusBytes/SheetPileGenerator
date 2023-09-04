import numpy as np
import sqlite3

MATERIAL_VALUES = {
    "TOP" : {
        "YOUNG_MODULUS": {"m": 125e6, "s": 15e6},
        "UNIT_WEIGHT": {"m": 22, "s": 2},
        "FRICION_ANGLE": {"m": 39.8, "s": 2},
    },
    "MIDDLE": {
        "YOUNG_MODULUS": {"m": 6500000, "s": 2e6},
        "UNIT_WEIGHT": {"m": 15, "s": 2},
        "FRICION_ANGLE": {"m": 25.0, "s": 2},
    },
    "BOTTOM": {
        "YOUNG_MODULUS": {"m": 50e6, "s": 10e6},
        "UNIT_WEIGHT": {"m": 18, "s": 2},
        "FRICION_ANGLE": {"m": 37, "s": 1},
    }
}

def pick_values_from_normal_distribution(mean, std, size):
    return np.random.normal(mean, std, size)

if __name__ == '__main__':
    # create sql database
    conn = sqlite3.connect('inputs_outputs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS inputs
                    (id integer primary key, name text, 
                    YOUNG_MODULUS_TOP real, UNIT_WEIGHT_TOP real, FRICION_ANGLE_TOP real ,
                    YOUNG_MODULUS_MIDDLE real, UNIT_WEIGHT_MIDDLE real, FRICION_ANGLE_MIDDLE real,
                    YOUNG_MODULUS_BOTTOM real, UNIT_WEIGHT_BOTTOM real, FRICION_ANGLE_BOTTOM,
                    HEAD real)''')
    # create input data
    ids = list(range(1, 1001))
    dir_name = [f"test_{i}" for i in ids]
    YOUNG_MODULUS_TOP = pick_values_from_normal_distribution(MATERIAL_VALUES["TOP"]["YOUNG_MODULUS"]["m"], MATERIAL_VALUES["TOP"]["YOUNG_MODULUS"]["s"], 1000)
    UNIT_WEIGHT_TOP = pick_values_from_normal_distribution(MATERIAL_VALUES["TOP"]["UNIT_WEIGHT"]["m"], MATERIAL_VALUES["TOP"]["UNIT_WEIGHT"]["s"], 1000)
    FRICION_ANGLE_TOP = pick_values_from_normal_distribution(MATERIAL_VALUES["TOP"]["FRICION_ANGLE"]["m"], MATERIAL_VALUES["TOP"]["FRICION_ANGLE"]["s"], 1000)
    YOUNG_MODULUS_MIDDLE = pick_values_from_normal_distribution(MATERIAL_VALUES["MIDDLE"]["YOUNG_MODULUS"]["m"], MATERIAL_VALUES["MIDDLE"]["YOUNG_MODULUS"]["s"], 1000)
    UNIT_WEIGHT_MIDDLE = pick_values_from_normal_distribution(MATERIAL_VALUES["MIDDLE"]["UNIT_WEIGHT"]["m"], MATERIAL_VALUES["MIDDLE"]["UNIT_WEIGHT"]["s"], 1000)
    FRICION_ANGLE_MIDDLE = pick_values_from_normal_distribution(MATERIAL_VALUES["MIDDLE"]["FRICION_ANGLE"]["m"], MATERIAL_VALUES["MIDDLE"]["FRICION_ANGLE"]["s"], 1000)
    YOUNG_MODULUS_BOTTOM = pick_values_from_normal_distribution(MATERIAL_VALUES["BOTTOM"]["YOUNG_MODULUS"]["m"], MATERIAL_VALUES["BOTTOM"]["YOUNG_MODULUS"]["s"], 1000)
    UNIT_WEIGHT_BOTTOM = pick_values_from_normal_distribution(MATERIAL_VALUES["BOTTOM"]["UNIT_WEIGHT"]["m"], MATERIAL_VALUES["BOTTOM"]["UNIT_WEIGHT"]["s"], 1000)
    FRICION_ANGLE_BOTTOM = pick_values_from_normal_distribution(MATERIAL_VALUES["BOTTOM"]["FRICION_ANGLE"]["m"], MATERIAL_VALUES["BOTTOM"]["FRICION_ANGLE"]["s"], 1000)
    HEAD = pick_values_from_normal_distribution(15, 1, 1000)
    # plot data distributions
    import matplotlib.pyplot as plt
    plt.hist(YOUNG_MODULUS_TOP, bins=100, label="TOP")
    plt.hist(YOUNG_MODULUS_MIDDLE, bins=100, label="MIDDLE")
    plt.hist(YOUNG_MODULUS_BOTTOM, bins=100, label="BOTTOM")
    plt.legend()
    plt.title("Young Modulus")
    plt.show()
    plt.hist(UNIT_WEIGHT_TOP, bins=100, label="TOP")
    plt.hist(UNIT_WEIGHT_MIDDLE, bins=100, label="MIDDLE")
    plt.hist(UNIT_WEIGHT_BOTTOM, bins=100, label="BOTTOM")
    plt.title("Unit Weight")
    plt.legend()
    plt.show()
    plt.hist(FRICION_ANGLE_TOP, bins=100, label="TOP")
    plt.hist(FRICION_ANGLE_MIDDLE, bins=100, label="MIDDLE")
    plt.hist(FRICION_ANGLE_BOTTOM, bins=100, label="BOTTOM")
    plt.legend()
    plt.title("Fricion Angle")
    plt.show()
    # insert data into database
    for i in range(1000):
        c.execute(f"INSERT INTO inputs  VALUES ({ids[i]}, '{dir_name[i]}', {YOUNG_MODULUS_TOP[i]}, "
                  f"{UNIT_WEIGHT_TOP[i]}, {FRICION_ANGLE_TOP[i]}, {YOUNG_MODULUS_MIDDLE[i]}, {UNIT_WEIGHT_MIDDLE[i]}, "
                  f"{FRICION_ANGLE_MIDDLE[i]}, {YOUNG_MODULUS_BOTTOM[i]}, {UNIT_WEIGHT_BOTTOM[i]}, "
                  f"{FRICION_ANGLE_BOTTOM[i]}, {HEAD[i]})")
    # commit changes
    conn.commit()
    # close connection
    conn.close()
