from shetpilegenerator.gmsh_to_kratos import GmshToKratos
from shetpilegenerator.material_library import *
from shetpilegenerator.output_process import OutputProcess, OutputProcessJsonReader
from shetpilegenerator.rf_generator import generate_jsons_for_material

import json
import os
import numpy as np
import sqlite3
import time
import matplotlib.pyplot as plt

from stem.soil_material import *
from stem.structural_material import *
from stem.water_boundaries import *
from stem.load import *
from stem.IO.kratos_material_io import KratosMaterialIO
from stem.IO.kratos_loads_io import KratosLoadsIO
from stem.IO.kratos_water_boundaries_io import KratosWaterBoundariesIO
from stem.water_boundaries import WaterBoundary, InterpolateLineBoundary, PhreaticLine

from gmsh_utils.gmsh_IO import GmshIO

import sys

sys.path.append(r"D:\Kratos_general\Kratos_build_version\KratosGeoMechanics")


import KratosMultiphysics as Kratos
import KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis as analysis_geo
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo

DICT_NORMALIZATION = {
    "YOUNG_MODULUS": 1e9,
    "FRICTION_ANGLE": 90,
}

NUMBER_OF_SAMPLES = 1000 # number of samples to generate

def modify_template_file(problem_name):
    with open("Project_Parameters_template.json", "r") as f:
        data = json.load(f)
    # modify the boundary conditions
    data["problem_data"]["problem_name"] = problem_name
    data["solver_settings"]["model_import_settings"]["input_filename"] = problem_name
    # write the new file
    with open("test_project_parameters.json", "w") as f:
        json.dump(data, f, indent=4)


def unit_weight_to_density_solid(unit_weight, porosity, gravity, density_water=1000):
    return (1 / (1 - porosity)) * ((unit_weight * 1000 / gravity) - (porosity * density_water))


def mohr_coulomb_parameters_to_list(umat_parameters):
    return [umat_parameters["YOUNG_MODULUS"],
            umat_parameters["POISSON_RATIO"],
            umat_parameters["COHESION"], 
            umat_parameters["FRICTION_ANGLE"],
            umat_parameters["DILATANCY_ANGLE"],
            umat_parameters["CUTOFF_STRENGTH"],
            umat_parameters["YIELD_FUNCTION_TYPE"],
            umat_parameters["UNDRAINED_POISSON_RATIO"],
            ]


def create_mohr_coloumb_model(values_dict):
    density_solid = unit_weight_to_density_solid(values_dict.get("unit_weight", 18),
                                                 values_dict.get("POROSITY", 0.3),
                                                 values_dict.get("gravity", 9.81)
                                                 )
    umat_parameters = mohr_coulomb_parameters_to_list(values_dict.get("UMAT_PARAMETERS", {}))
    # define soil 1
    soil_1_gen = TwoPhaseSoil(ndim=2,
                              DENSITY_SOLID=density_solid,
                              POROSITY=values_dict.get("POROSITY", 0.3),
                              BULK_MODULUS_SOLID=values_dict.get("BULK_MODULUS_SOLID", 1E9),
                              PERMEABILITY_XX=values_dict.get("PERMEABILITY_XX", 1E-15),
                              PERMEABILITY_YY=values_dict.get("PERMEABILITY_YY", 1E-15),
                              PERMEABILITY_XY=values_dict.get("PERMEABILITY_XY", 1E-15),
                              )
    # Define umat constitutive law parameters
    umat_constitutive_parameters = SmallStrainUdsmLaw(UDSM_PARAMETERS=umat_parameters,
                                                      UDSM_NAME=values_dict.get("UDSM_NAME", "D:/SheetPileGenerator/test/kratos_write_test/MohrCoulomb64.dll"),
                                                      UDSM_NUMBER=values_dict.get("UDSM_NUMBER", 1),
                                                      IS_FORTRAN_UDSM=True,
                                                      )
    soil_1_water = SaturatedBelowPhreaticLevelLaw()
    soil_1 = SoilMaterial(name=values_dict.get("name", "soil_1"),
                          soil_formulation=soil_1_gen,
                          constitutive_law=umat_constitutive_parameters,
                          retention_parameters=soil_1_water)
    return soil_1


def define_geometry_from_gmsh(input_points, name_labels):
    dimension = 2
    gmsh_io = GmshIO()
    gmsh_io.generate_geometry(input_points,
                              [0, 0, 0],
                              dimension,
                              "mesh_dike_2d",
                              name_labels,
                              5)

    physical_groups = gmsh_io.generate_extract_mesh(dimension, "mesh_dike_2d", ".", False, True)
    geo_data = gmsh_io.geo_data
    mesh_data = gmsh_io.mesh_data
    mesh_data['physical_groups'] = physical_groups
    total_dict = {'geo_data': geo_data, 'mesh_data': mesh_data}
    return total_dict

def define_water_boundaries(water_top_1, water_top_2, water_top_3,water_middle, water_bottom):
    # top layer boundary conditions multiline
    water_boundaries_top = []
    for counter, boundary in enumerate([water_top_1, water_top_2, water_top_3]):
        water_line_top_parameters = PhreaticLine(
            is_fixed=True,
            gravity_direction=1,
            out_of_plane_direction=2,
            specific_weight=9.81,
            value=0,
            first_reference_coordinate=water_top_1[0],
            second_reference_coordinate=water_top_1[1],
            surfaces_assigment=water_top_1[2],
        )
        water_boundaries_top.append(WaterBoundary(water_line_top_parameters, name=f"top_water_boundary_{counter + 1}"))
    # bottom layer boundary conditions multiline
    water_line_bottom_parameters = PhreaticLine(
        is_fixed=True,
        gravity_direction=1,
        out_of_plane_direction=2,
        value=0,
        first_reference_coordinate=water_bottom[0],
        second_reference_coordinate=water_bottom[1],
        specific_weight=9.81,
        surfaces_assigment=water_bottom[2],
    )
    water_boundary_bottom = WaterBoundary(water_line_bottom_parameters, name="bottom_water_boundary")
    # middle layer boundary conditions
    interpolation_type = InterpolateLineBoundary(
        is_fixed=True,
        out_of_plane_direction=2,
        gravity_direction=1,
        surfaces_assigment=water_middle,
    )
    water_boundary_interpolate = WaterBoundary(interpolation_type, name="middle_water_boundary")

    kratos_io = KratosWaterBoundariesIO(domain="PorousDomain")
    return [kratos_io.create_water_boundary_dict(boundary) for boundary in water_boundaries_top + [water_boundary_bottom, water_boundary_interpolate]]

def create_linear_elastic_model(values_dict):
    density_solid = unit_weight_to_density_solid(values_dict.get("unit_weight", 18),
                                                 values_dict.get("POROSITY", 0.3),
                                                 values_dict.get("gravity", 9.81)
                                                 )
    # define soil 1
    soil_1_gen = TwoPhaseSoil(ndim=2,
                              DENSITY_SOLID=density_solid,
                              POROSITY=values_dict.get("POROSITY", 0.3),
                              BULK_MODULUS_SOLID=values_dict.get("BULK_MODULUS_SOLID", 1E9),
                              PERMEABILITY_XX=values_dict.get("PERMEABILITY_XX", 1E-15),
                              PERMEABILITY_YY=values_dict.get("PERMEABILITY_YY", 1E-15),
                              PERMEABILITY_XY=values_dict.get("PERMEABILITY_XY", 1E-15),

                              )
    # Define umat constitutive law parameters
    umat_constitutive_parameters = LinearElasticSoil(YOUNG_MODULUS=values_dict.get("YOUNG_MODULUS", 1E9),
                                                     POISSON_RATIO=values_dict.get("POISSON_RATIO", 0.3),
                                                     )
    soil_1_water = SaturatedBelowPhreaticLevelLaw()
    soil_1 = SoilMaterial(name=values_dict.get("name", "soil_1"),
                          soil_formulation=soil_1_gen,
                          constitutive_law=umat_constitutive_parameters,
                          retention_parameters=soil_1_water)
    return soil_1


def write_materials_dict(file_name, materials, soil_model="mohr_coulomb"):
    if soil_model == "mohr_coulomb":
        all_materials = [create_mohr_coloumb_model(values_dict) for values_dict in materials]
    elif soil_model == "linear_elastic":
        all_materials = [create_linear_elastic_model(values_dict) for values_dict in materials]

    kratos_io = KratosMaterialIO(ndim=2, domain='PorousDomain')
    materials_collection = []
    for counter, material in enumerate(all_materials):
        materials_collection.append(kratos_io.create_material_dict(part_name=material.name.split(".")[-1], material=material, material_id=counter + 1))
    global_material_dictionary = {"properties": materials_collection}
    # write the json file
    with open(file_name, 'w') as outfile:
        json.dump(global_material_dictionary, outfile, indent=4)

def add_and_save_extra_parameters(template_file, output_file, extra_parameters):
    with open(template_file, 'r') as json_file:
        data = json.load(json_file)
    for key, value in extra_parameters.items():
        data["processes"][key] += value
    with open(output_file, 'w') as json_file:
        json.dump(data,
                  json_file,
                  indent=4)

def define_layers():
    TOP_1 = [
        (-80, -2, 0),
        (-20, -2, 0),
        (-20, 0, 0),
        (-80, 0, 0),
    ]
    TOP_2 = [
        (-20, -2, 0),
        (25, -2, 0),
        (25, 2, 0),
        (8, 8, 0),
        (0, 8, 0),
        (-20, 0, 0),
    ]
    TOP_3 = [
        (25, -2, 0),
        (80, -2, 0),
        (80, 2, 0),
        (25, 2, 0),
    ]
    MIDDLE = [
        (-80, -8, 0),
        (-20, -8, 0),
        (25, -8, 0),
        (80, -8, 0),
        (80, -2, 0),
        (25, -2, 0),
        (-20, -2, 0),
        (-80, -2, 0)
    ]
    BOTTOM = [
        (-80, -15, 0),
        (-20, -15, 0),
        (25, -15, 0),
        (80, -15, 0),
        (80, -8, 0),
        (25, -8, 0),
        (-20, -8, 0),
        (-80, -8, 0),
    ]
    return [TOP_1, TOP_2, TOP_3, MIDDLE, BOTTOM]


def define_water_line_based_on_outside_head(head):
    TOP_1_WL = [[-80., head, 0.], [-20., head, 0.], ['TOP_1']]
    TOP_2_WL = [[-20., head, 0.], [25., 5.58, 0.], ['TOP_2']]
    TOP_3_WL = [[25., 2, 0.], [80., 2., 0.], ['TOP_3']]
    # water line bottom
    WL3 = [[-80., head, 0.], [80.,  5.31, 0.], ["BOTTOM"]]
    # water line middle
    WL2 = "MIDDLE"
    return TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3


def get_stages(project_path, n_stages):

    parameter_file_names = ['Project_Parameters_' + str(i + 1) + '.json' for i in range(n_stages)]
    # set stage parameters
    parameters_stages = [None] * n_stages
    os.chdir(project_path)
    for idx, parameter_file_name in enumerate(parameter_file_names):
        with open(parameter_file_name, 'r') as parameter_file:
            parameters_stages[idx] = Kratos.Parameters(parameter_file.read())

    model = Kratos.Model()
    stages = [analysis_geo.GeoMechanicsAnalysis(model, stage_parameters) for stage_parameters in parameters_stages]
    for stage in stages:
        stage.Run()
    #[stage.Run() for stage in stages]
    return stages, model


def run_multistage_calculation(file_path, stages_number):
    cwd = os.getcwd()
    stages, model = get_stages(file_path, stages_number)
    os.chdir('..\..')
    plot_RFs_set_in_last_stage(stages[-1], file_path)
    model.Reset()
    return


def plot_RFs_set_in_last_stage(last_stage, directory):
    # collect data
    model_part = last_stage._list_of_output_processes[0].model_part
    elements = model_part.Elements

    x = []
    y = []
    x_nodes = [Node.X for Node in model_part.Nodes]
    y_nodes = [Node.Y for Node in model_part.Nodes]
    phi = []
    youngs_modulus = []
    connectivities = []
    for element in elements:
        values = element.CalculateOnIntegrationPoints(KratosGeo.UMAT_PARAMETERS, model_part.ProcessInfo)
        points = element.GetIntegrationPoints()
        nodes = element.GetNodes()
        # create connectivity
        connectivities.append([node.Id for node in nodes])
        for counter, umat_vector in enumerate(values):
            x.append(points[counter][0])
            y.append(points[counter][1])
            phi.append(umat_vector[3])
            youngs_modulus.append(umat_vector[0])
    # interpolate from the integration points to the nodes
    x = np.array(x)
    y = np.array(y)
    phi = np.array(phi)
    youngs_modulus = np.array(youngs_modulus)
    import scipy.interpolate as interpolate
    phi = interpolate.griddata((x, y), phi, (x_nodes, y_nodes), method='nearest')
    youngs_modulus = interpolate.griddata((x, y), youngs_modulus, (x_nodes, y_nodes), method='nearest')
    # normalize the values
    phi = phi / DICT_NORMALIZATION["FRICTION_ANGLE"]
    youngs_modulus = youngs_modulus / DICT_NORMALIZATION["YOUNG_MODULUS"]

    plot_nodal_results(
        x_nodes,
        y_nodes,
        phi,
        np.array(connectivities),
        save=True,
        file_name="FRICTION_ANGLE.png",
        directory=directory
    )
    plot_nodal_results(
        x_nodes,
        y_nodes,
        youngs_modulus,
        np.array(connectivities),
        save=True,
        file_name="YOUNGS_MODULUS.png",
        directory=directory
    )
    return None


def reduce_phi(RF, phi_init):
    import math
    return math.degrees(math.atan(math.tan(math.radians(phi_init)) / RF))


def modify_material_parameters_c_phi_reduction(project_path, stage_number,  RF, c_init, phi_init):
    # modify the parameters
    material_parameters_file = os.path.join(project_path, f'MaterialParameters_{stage_number}.json')
    # open json file
    with open(material_parameters_file, 'r') as parameter_file:
        material_parameters = json.load(parameter_file)
    # modify the parameters
    for counter, key in enumerate(material_parameters['properties']):
        material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][2] = c_init / RF
        material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][3] = reduce_phi(RF,
                                                                                                               phi_init)
    # write the modified parameters
    with open(material_parameters_file, 'w') as parameter_file:
        json.dump(material_parameters, parameter_file, indent=4)


def get_initial_c_phi_parameters(project_path, stage_number):
    # modify the parameters
    material_parameters_file = os.path.join(project_path, f'MaterialParameters_{stage_number}.json')
    # open json file
    with open(material_parameters_file, 'r') as parameter_file:
        material_parameters = json.load(parameter_file)
    # modify the parameters
    for counter, key in enumerate(material_parameters['properties']):
        c_init = material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][2]
        phi_init = material_parameters['properties'][counter]['Material']['Variables']['UMAT_PARAMETERS'][3]
    return c_init, phi_init


def run_c_phi_reduction(project_path, stage_number, RF_min, RF_max, gmsh_to_kratos, step=0.05, ):
    RF = RF_min
    c_init, phi_init = get_initial_c_phi_parameters(project_path, stage_number)
    while RF < RF_max:
        try:
            print("RF = ", RF)
            modify_material_parameters_c_phi_reduction(project_path, stage_number, RF, c_init, phi_init)
            # run the simulation
            run_multistage_calculation("kratos_write_test", 2)
            RF += step
        except Exception as e:
            if str(e) == "The maximum number of cycles is reached without convergence!":
                print("C-phi reduction finished! At RF = ", RF)
                # rerun the simulation with the last RF
                os.chdir("..")
                modify_material_parameters_c_phi_reduction(project_path, stage_number, RF - step, c_init, phi_init)
                run_multistage_calculation("kratos_write_test", 2)
                post_process(2, 2.0, gmsh_to_kratos)
                break
            else:
                raise e


def get_field_process_dict(field_process):
    return {
        "python_module": "set_parameter_field_process",
        "kratos_module": "KratosMultiphysics.GeoMechanicsApplication",
        "process_name":  "SetParameterFieldProcess",
        "Parameters":    {
            "model_part_name": f"PorousDomain.{field_process['name']}",
            "variable_name": field_process['variable'],
            "func_type": field_process['function_type'],
            "function": field_process['function'],
			"dataset": "dummy",
            "dataset_file_name": field_process['dataset_file_name'],
        }
    }

def modify_project_parameters(file_path, problem_name, material_parameters, list_of_parts, time_start, time_end):
    # open the json file
    with open(file_path, 'r') as parameter_file:
        project_parameters = json.load(parameter_file)
    # modify the parameters
    project_parameters["problem_data"]["problem_name"] = problem_name
    project_parameters["solver_settings"]["model_import_settings"]["input_filename"] = problem_name
    project_parameters["solver_settings"]["material_import_settings"]["materials_filename"] = material_parameters
    project_parameters["output_processes"]['gid_output'][0]['Parameters']['output_name'] = problem_name
    project_parameters["problem_data"]["start_time"] = time_start
    project_parameters["problem_data"]["end_time"] = time_end
    project_parameters["solver_settings"]["start_time"] = time_start
    for key, value in list_of_parts.items():
        project_parameters["solver_settings"][key] = value
    project_parameters['processes']["json_output"] = OutputProcess(output_variables=["DISPLACEMENT", "WATER_PRESSURE"],
                                                                   output_file_name=problem_name + "_output.json").to_dict()
    # save the file
    with open(file_path, 'w') as parameter_file:
        json.dump(project_parameters, parameter_file, indent=4)


def plot_nodal_results(x, y, values, connectivity, save=False, file_name="geometry.png", directory="."):
    # plot the results contour plot
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
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



def set_stage(stage_number, start_time, end_time, project_path, constrains_on_surfaces, soils, head_level, soil_model, data, loads):
    write_materials_dict( project_path + f"/MaterialParameters_{stage_number}.json",
                         soils,
                         soil_model=soil_model)
    # define water conditions stage 1
    TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3 = define_water_line_based_on_outside_head(head_level)
    dictionary_water_boundaries_stage3 = define_water_boundaries(TOP_1_WL, TOP_2_WL, TOP_3_WL, WL2, WL3)
    load_names = []
    # add the loads
    if loads is not None:
        for load in loads:
            io_load = KratosLoadsIO("PorousDomain")
            dictionary_water_boundaries_stage3.append(io_load.create_load_dict(part_name=load['name'],
                                                                               parameters=load['load'])
                                                      )
            load_names.append(load['name'])
    # add the field processes
    sub_model_names_field_process = []
    for field_process in constrains_on_surfaces['field_processes']:
        sub_model_names_field_process.append(field_process['name'])
        dictionary_water_boundaries_stage3.append(get_field_process_dict(field_process))
    # define stage 1 parameters
    add_and_save_extra_parameters("Project_Parameters_template.json",
                                  project_path + f"/Project_Parameters_{stage_number}.json",
                                  {"loads_process_list": dictionary_water_boundaries_stage3})
    # create the object
    gmsh_to_kratos = GmshToKratos(data)
    gmsh_to_kratos.read_gmsh_to_kratos(property_list=list(range(1, len(constrains_on_surfaces['surfaces']) + 1)),
                                       mpda_file=project_path + f"/test_multistage_{stage_number}.mdpa",
                                       constrains_on_surfaces=constrains_on_surfaces,
                                       top_load=bool(loads is not None),)
    lists_of_all_parts = {
        "problem_domain_sub_model_part_list": constrains_on_surfaces['surfaces'],
        "body_domain_sub_model_part_list": constrains_on_surfaces['surfaces'],
        "processes_sub_model_part_list": ["bottom_disp", "side_disp", "gravity"] +
                                         constrains_on_surfaces['names'] +
                                         load_names +
                                         sub_model_names_field_process
    }

    modify_project_parameters(project_path + f"/Project_Parameters_{stage_number}.json",
                              f"test_multistage_{stage_number}",
                              f"MaterialParameters_{stage_number}.json",
                              lists_of_all_parts,
                              time_start=start_time,
                              time_end=end_time)
    return gmsh_to_kratos


def post_process(stage_index, timestep, gmsh_to_kratos, save=False, file_name="geometry.png", directory="."):
    path_to_results = os.path.join(f"{directory}/test_multistage_{stage_index}_output.json")
    post_process = OutputProcessJsonReader(path_to_results)
    x_coordinates, y_coordinates, displacement_x, displacement_y, water_pressure = post_process.get_values_in_timestep(timestep)
    total_displacement = [np.sqrt(displacement_x[i]**2 + displacement_y[i]**2) for i in range(len(displacement_x))]
    plot_nodal_results(x_coordinates,
                       y_coordinates,
                       displacement_x,
                       gmsh_to_kratos.gmsh_dict['mesh_data']['elements']['TRIANGLE_3N']['connectivities'],
                       save=save,
                       file_name="displacement_x_" + file_name,
                       directory=directory)
    plot_nodal_results(x_coordinates,
                       y_coordinates,
                       displacement_y,
                       gmsh_to_kratos.gmsh_dict['mesh_data']['elements']['TRIANGLE_3N']['connectivities'],
                       save=save,
                       file_name="displacement_y_" + file_name,
                       directory=directory)
    plot_nodal_results(x_coordinates,
                       y_coordinates,
                       water_pressure,
                       gmsh_to_kratos.gmsh_dict['mesh_data']['elements']['TRIANGLE_3N']['connectivities'],
                       save=save,
                       file_name="water_pressure_" + file_name,
                       directory=directory)
    plot_nodal_results(x_coordinates,
                          y_coordinates,
                          total_displacement,
                          gmsh_to_kratos.gmsh_dict['mesh_data']['elements']['TRIANGLE_3N']['connectivities'],
                          save=save,
                          file_name="total_displacement_" + file_name,
                          directory=directory)

def plot_geometry(layers, save=False, file_name="geometry.png", directory="."):
    # plot the geometry with filled in polygons
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(figsize=(10, 10))
    for layer in layers:
        points_2d = [point[:2] for point in layer]
        # plot with only the boundaries shown as black lines
        ax.plot(*zip(*(points_2d + [points_2d[0]])), color='black')
    # set the limits automatically based on the geometry
    ax.autoscale()
    # turn off the axis
    ax.axis('off')
    if save:
        # save the figure in the directory and create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, file_name))
    else:
        plt.show()


def plot_materials(materials, layers, material_names, save=False, file_name="geometry.png", directory="."):
    for material_name in material_names:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.colors as colors
        import matplotlib.cm as cm
        fig, ax = plt.subplots(figsize=(10, 10))
        for counter, layer in enumerate(layers):
            material = materials[counter]
            # get the values for the material
            value = material['UMAT_PARAMETERS'][material_name]
            # normalize the value
            norm_value = DICT_NORMALIZATION[material_name]
            my_cmap = cm.get_cmap('Greys')  # or any other one
            norm = colors.Normalize(0, norm_value)
            color_i = my_cmap(norm(value))  # returns an rgba value
            # plot the value with the color
            points_2d = [point[:2] for point in layer]
            # plot patches with the value as color
            ax.add_patch(patches.Polygon(points_2d, color=color_i))
        ax.autoscale()
        # turn off the axis
        ax.axis('off')
        if save:
            # save the figure in the directory and create the directory if it does not exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(os.path.join(directory, f"{material_name}_{file_name}"))
        else:
            plt.show()


def modify_MC_parameters(materials, mc_initial_parameters):
    materials[0]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_TOP']
    materials[1]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_TOP']
    materials[2]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_TOP']
    materials[3]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_MIDDLE']
    materials[4]['UMAT_PARAMETERS']['YOUNG_MODULUS'] = mc_initial_parameters['YOUNG_MODULUS_BOTTOM']
    materials[0]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_TOP']
    materials[1]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_TOP']
    materials[2]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_TOP']
    materials[3]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_MIDDLE']
    materials[4]['UMAT_PARAMETERS']['FRICTION_ANGLE'] = mc_initial_parameters['FRICTION_ANGLE_BOTTOM']


def create_model(directory, input_values):
    constrains_on_surfaces = {
        "names": ["top_water_boundary_1", "top_water_boundary_2", "top_water_boundary_3",
                  "middle_water_boundary", "bottom_water_boundary"],
        "surfaces": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "material_per_surface": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "field_processes": [{"name": "TOP_1_RF", "variable": "YOUNG_MODULUS", "function":  "dummy", "function_type": "json_file", "dataset_file_name": "TOP_1_le_RF.json"},
                            {"name": "TOP_2_RF", "variable": "YOUNG_MODULUS", "function":  "dummy", "function_type": "json_file", "dataset_file_name": "TOP_2_le_RF.json"},
                            {"name": "TOP_3_RF", "variable": "YOUNG_MODULUS", "function":  "dummy", "function_type": "json_file", "dataset_file_name": "TOP_3_le_RF.json"},
                            {"name": "MIDDLE_RF", "variable": "YOUNG_MODULUS", "function": "dummy", "function_type": "json_file", "dataset_file_name": "MIDDLE_le_RF.json"},
                            {"name": "BOTTOM_RF", "variable": "YOUNG_MODULUS", "function": "dummy", "function_type": "json_file", "dataset_file_name": "BOTTOM_le_RF.json"}],
    }

    # set default loads
    # define the line load
    line_load = LineLoad(active=[True, True, False], value=[0.0, 0.0, 0.0])
    line_load_input = {"name": "dike_load", "load": line_load}

    layers = define_layers()
    plot_geometry(layers, save=True, file_name="geometry.png", directory=directory)
    data = define_geometry_from_gmsh(layers, constrains_on_surfaces['surfaces'])
    # set all random field json files
    generate_jsons_for_material(input_values["INDEX"], directory, data['mesh_data']['physical_groups'], data['mesh_data']['nodes']['coordinates'])
    # make stage 1
    # create materials
    TOP_LE_1 = TOP_LE.copy()
    TOP_LE_1["name"] = "PorousDomain.TOP_1"
    TOP_LE_2 = TOP_LE.copy()
    TOP_LE_2["name"] = "PorousDomain.TOP_2"
    TOP_LE_3 = TOP_LE.copy()
    TOP_LE_3["name"] = "PorousDomain.TOP_3"
    gmsh_to_kratos = set_stage(1,
                               0.0,
                               1.0,
                               directory,
                               constrains_on_surfaces,
                               [TOP_LE_1, TOP_LE_2, TOP_LE_3, MIDDLE_LE, BOTTOM_LE],
                               3.08,
                               "linear_elastic",
                               data,
                               [line_load_input])

    # make stage 2
    constrains_on_surfaces = {
        "names": ["top_water_boundary_1","top_water_boundary_2","top_water_boundary_3", "middle_water_boundary", "bottom_water_boundary"],
        "surfaces": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "material_per_surface": ["TOP_1", "TOP_2", "TOP_3", "MIDDLE", "BOTTOM"],
        "field_processes": [{"name": "TOP_1_RF", "variable": "UMAT_PARAMETERS", "function": "dummy",  "function_type": "json_file", "dataset_file_name": "TOP_1_mc_RF.json"},
                            {"name": "TOP_2_RF", "variable": "UMAT_PARAMETERS", "function": "dummy",  "function_type": "json_file", "dataset_file_name": "TOP_2_mc_RF.json"},
                            {"name": "TOP_3_RF", "variable": "UMAT_PARAMETERS", "function": "dummy",  "function_type": "json_file", "dataset_file_name": "TOP_3_mc_RF.json"},
                            {"name": "MIDDLE_RF", "variable": "UMAT_PARAMETERS", "function": "dummy", "function_type": "json_file", "dataset_file_name": "MIDDLE_mc_RF.json"},
                            {"name": "BOTTOM_RF", "variable": "UMAT_PARAMETERS", "function": "dummy", "function_type": "json_file", "dataset_file_name": "BOTTOM_mc_RF.json"}],
    }
    # create materials
    TOP_MC_1 = TOP_MC.copy()
    TOP_MC_1["name"] = "PorousDomain.TOP_1"
    TOP_MC_2 = TOP_MC.copy()
    TOP_MC_2["name"] = "PorousDomain.TOP_2"
    TOP_MC_3 = TOP_MC.copy()
    TOP_MC_3["name"] = "PorousDomain.TOP_3"
    modify_MC_parameters([TOP_MC_1, TOP_MC_2, TOP_MC_3, MIDDLE_MC, BOTTOM_MC], input_values)
    gmsh_to_kratos = set_stage(2, 1.0, 2.0,
                               directory,
                               constrains_on_surfaces,
                               [TOP_MC_1, TOP_MC_2, TOP_MC_3, MIDDLE_MC, BOTTOM_MC],
                               input_values['HEAD'],
                               "mohr_coulomb",
                               data,
                               [line_load_input])
    #plot_materials([TOP_MC_1, TOP_MC_2, TOP_MC_3, MIDDLE_MC, BOTTOM_MC],
    #               layers,
    #               ["YOUNG_MODULUS", "FRICTION_ANGLE"],
    #               save=True,
    #               directory=directory,
    #               file_name="material.png")


    return gmsh_to_kratos


if __name__ == '__main__':
    # open sqlite database and loop over the values
    conn = sqlite3.connect('inputs_outputs.db')
    c = conn.cursor()
    c.execute("SELECT * FROM inputs")
    results = c.fetchall()
    for result in results[:1]:
        input_values = {
            "INDEX": result[0],
            "YOUNG_MODULUS_TOP": result[2],
            "YOUNG_MODULUS_MIDDLE": result[5],
            "YOUNG_MODULUS_BOTTOM": result[8],
            "FRICTION_ANGLE_TOP": result[4],
            "FRICTION_ANGLE_MIDDLE": result[7],
            "FRICTION_ANGLE_BOTTOM": result[10],
            "HEAD": result[11],
            "LOAD": 0.0
        }
        directory = f"results_RF/{result[0]}"
        gmsh_to_kratos = create_model(directory, input_values)
        plt.close("all")
    conn.close()
    failed = []
    for result in results[:100]:
        try:
            directory = f"results_RF/{result[0]}"
            run_multistage_calculation(directory, 2)
        except:
            print(f"failed to run {result[0]}")
            failed.append(result[0])
            pass
    for result in results[:100]:
        try:
            directory = f"results_RF/{result[0]}"
            post_process(2, 2.0, gmsh_to_kratos, save=True, directory=directory, file_name="stage_3.png")
        except:
            print(f"failed to post process {result[0]}")
            pass
    print(failed)










