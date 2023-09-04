from gstools import SRF, Gaussian
import numpy as np
import json
from .material_library import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import os


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

ELEMENTS = "TRIANGLE_3N"


def generate_field(mean, std_value, aniso_x, aniso_y, coordinates, seed):

    x = [coord[0] for coord in coordinates]
    y = [coord[1] for coord in coordinates]
    # set random field parameters
    len_scale = np.array([aniso_x, aniso_y])
    var = std_value ** 2

    model = Gaussian(dim=2, var=var, len_scale=len_scale, seed=seed)
    srf = SRF(model, mean=mean, seed=seed)
    new_values = srf((x, y), mesh_type="unstructured", seed=seed)
    return new_values.tolist()


def generate_jsons_for_le(sample_number, directory, physical_groups, nodes):
    # set up all the parameters
    le_files = [{"json_names": f"{directory}/TOP_1_le_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_1"},
                {"json_names": f"{directory}/TOP_2_le_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_2"},
                {"json_names": f"{directory}/TOP_3_le_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_3"},
                {"json_names": f"{directory}/MIDDLE_le_RF.json", "material_parameters": MATERIAL_VALUES['MIDDLE'], "physical_group": "MIDDLE"},
                {"json_names": f"{directory}/BOTTOM_le_RF.json", "material_parameters": MATERIAL_VALUES['BOTTOM'], "physical_group": "BOTTOM"}]
    young_modulus = []
    for counter, file in enumerate(le_files):
        mean = file["material_parameters"]["YOUNG_MODULUS"]["m"]
        std_value = file["material_parameters"]["YOUNG_MODULUS"]["s"]
        aniso_x = 5
        aniso_y = 2
        seed = sample_number + counter + 761993
        nodes_pg = physical_groups[le_files[counter]["physical_group"]][ELEMENTS]['connectivities']
        # get element coordinates
        elements_coordinates = []
        for node in nodes_pg:
            # get the center of the element
            x = (nodes[int(node[0] - 1)][0] + nodes[int(node[1] - 1)][0] + nodes[int(node[2] - 1)][0]) / 3
            y = (nodes[int(node[0] - 1)][1] + nodes[int(node[1] - 1)][1] + nodes[int(node[2] - 1)][1]) / 3
            elements_coordinates.append([x, y])
        new_values = generate_field(mean, std_value, aniso_x, aniso_y, elements_coordinates, seed)
        young_modulus.append(new_values)
        dict_values = {"values": new_values}
        # write the new values in the json file
        with open(file["json_names"], "w") as json_file:
            json.dump(dict_values, json_file, indent=4)
    return young_modulus


def generate_jsons_for_mc(sample_number, directory, physical_groups, nodes, young_modulus):
    total_results = []
    # set up all the parameters
    le_files = [{"json_names": f"{directory}/TOP_1_mc_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_1", "material": TOP_MC},
                {"json_names": f"{directory}/TOP_2_mc_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_2", "material": TOP_MC},
                {"json_names": f"{directory}/TOP_3_mc_RF.json", "material_parameters": MATERIAL_VALUES['TOP'], "physical_group": "TOP_3", "material": TOP_MC},
                {"json_names": f"{directory}/MIDDLE_mc_RF.json", "material_parameters": MATERIAL_VALUES['MIDDLE'], "physical_group": "MIDDLE", "material": MIDDLE_MC},
                {"json_names": f"{directory}/BOTTOM_mc_RF.json", "material_parameters": MATERIAL_VALUES['BOTTOM'], "physical_group": "BOTTOM", "material": BOTTOM_MC}]
    for counter, file in enumerate(le_files):
        aniso_x = 5
        aniso_y = 2
        seed = sample_number + counter + 761993
        nodes_pg = physical_groups[le_files[counter]["physical_group"]][ELEMENTS]['connectivities']
        # get element coordinates
        elements_coordinates = []
        for node in nodes_pg:
            # get the center of the element
            x = (nodes[int(node[0] - 1)][0] + nodes[int(node[1] - 1)][0] + nodes[int(node[2] - 1)][0]) / 3
            y = (nodes[int(node[0] - 1)][1] + nodes[int(node[1] - 1)][1] + nodes[int(node[2] - 1)][1]) / 3
            elements_coordinates.append([x, y])
        # sample young modulus
        youngs_modulus = young_modulus[counter]
        # sample the friction angle
        mean = file["material_parameters"]["FRICION_ANGLE"]["m"]
        std_value = file["material_parameters"]["FRICION_ANGLE"]["s"]
        friction_angle = generate_field(mean, std_value, aniso_x, aniso_y, elements_coordinates, seed)
        # get everything else in list format
        values = []
        for i in range(len(youngs_modulus)):
            values.append([youngs_modulus[i],
                           file["material"]["UMAT_PARAMETERS"]["POISSON_RATIO"],
                           file["material"]["UMAT_PARAMETERS"]["COHESION"],
                           friction_angle[i],
                           file["material"]["UMAT_PARAMETERS"]["DILATANCY_ANGLE"],
                           file["material"]["UMAT_PARAMETERS"]["CUTOFF_STRENGTH"],
                           file["material"]["UMAT_PARAMETERS"]["YIELD_FUNCTION_TYPE"],
                           file["material"]["UMAT_PARAMETERS"]["UNDRAINED_POISSON_RATIO"]])

        dict_values = {"values": values}
        # write the new values in the json file
        with open(file["json_names"], "w") as json_file:
            json.dump(dict_values, json_file, indent=4)
        total_results.append(values)
    return total_results

def plot_rf_values(list_of_indexes, directory, physical_groups, nodes, random_fields , save=False):
    # plot the results contour plot
    connectivity = np.empty((0, 3), int)
    for name, item in physical_groups.items():
        connectivity = np.concatenate((connectivity, np.array(item[ELEMENTS]['connectivities'])), axis=0)
    x = nodes[:, 0]
    y = nodes[:, 1]
    for index in list_of_indexes:
        # collect the values
        colormap = plt.cm.get_cmap('Greys')
        sm = plt.cm.ScalarMappable(cmap=colormap)
        sm.set_clim(vmin=min(values), vmax=max(values))
        trianges = connectivity - 1
        triang = mtri.Triangulation(x, y, trianges)
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        tcf = ax1.tricontourf(triang, values, cmap=colormap)
        #ax1.triplot(triang, 'ko-', alpha=0)
        # add colorbar with min and max values
        #cbar = fig1.colorbar(sm, ax=ax1)
        # turn off the axis
        ax1.axis('off')
        if save:
            # save the figure if the directory exists ortherwise make the directory
            if not os.path.exists(directory):
                os.makedirs(directory)
            # save the figure in gray scale
            plt.savefig(os.path.join(directory, file_name))
        else:
            plt.show()
        plt.close()

def generate_jsons_for_material(sample_number, directory, physical_groups, nodes):
    young_modulus = generate_jsons_for_le(sample_number, directory, physical_groups, nodes)
    results_rd_mc = generate_jsons_for_mc(sample_number, directory, physical_groups, nodes, young_modulus)
    #plot_rf_values([0, 3], directory, physical_groups, nodes, results_rd_mc)



