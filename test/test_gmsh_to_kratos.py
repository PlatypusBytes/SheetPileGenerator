import pytest
from shetpilegenerator.gmsh_to_kratos import GmshToKratos
from shetpilegenerator.material_global_definition import MaterialGlobalDefinition
import pickle
import json

from stem.soil_material import *
from stem.structural_material import *
from stem.water_boundaries import *
from stem.IO.kratos_material_io import KratosMaterialIO
from stem.IO.kratos_water_boundaries_io import KratosWaterBoundariesIO
from stem.water_boundaries import WaterBoundary, InterpolateLineBoundary, PhreaticMultiLineBoundary

from gmsh_utils.gmsh_IO import GmshIO

TOP = {
    "name": "top",
    "unit_weight": 22,
    "UMAT_PARAMETERS": {
        "UMAT_NAME": "umat_sand",
        "YOUNG_MODULUS": 125e3,
        "FRICTION_ANGLE": 39.8,
        "DILATANCY_ANGLE": 0.0,
        "COHESION": 1.0,
        "CUTOFF_STRENGTH": 0.0,
        "POISSON_RATIO": 0.35,
        "UNDRAINED_POISSON_RATIO": 0.35,
        "YIELD_FUNCTION_TYPE": 0,

    }
}

MIDDLE = {
    "name": "middle",
    "unit_weight": 15,
    "UMAT_PARAMETERS": {
        "UMAT_NAME": "umat_sand",
        "YOUNG_MODULUS": 6500,
        "FRICTION_ANGLE": 25.80,
        "DILATANCY_ANGLE": 0.0,
        "COHESION": 14.80,
        "CUTOFF_STRENGTH": 0.0,
        "POISSON_RATIO": 0.35,
        "UNDRAINED_POISSON_RATIO": 0.35,
        "YIELD_FUNCTION_TYPE": 0,

    }
}

BOTTOM = {
    "name": "bottom",
    "unit_weight": 18,
    "UMAT_PARAMETERS": {
        "UMAT_NAME": "umat_sand",
        "YOUNG_MODULUS": 50.00E3,
        "FRICTION_ANGLE": 37,
        "DILATANCY_ANGLE": 0.0,
        "COHESION": 1.0,
        "CUTOFF_STRENGTH": 0.0,
        "POISSON_RATIO": 0.35,
        "UNDRAINED_POISSON_RATIO": 0.35,
        "YIELD_FUNCTION_TYPE": 0,

    }
}


def modify_template_file(problem_name):
    with open("Project_Parameters_template.json", "r") as f:
        data = json.load(f)
    # modify the boundary conditions
    data["problem_data"]["problem_name"] = problem_name
    data["solver_settings"]["model_import_settings"]["input_filename"] = problem_name
    # write the new file
    with open("test_project_parameters.json", "w") as f:
        json.dump(data, f, indent=4)


def unit_weight_to_density_solid(unit_weight, porosity, gravity, density_water=1):
    return (1 / (1 - porosity)) * ((unit_weight / gravity) - (porosity * density_water))


def mohr_coulomb_parameters_to_list(umat_parameters):
    return [umat_parameters["YOUNG_MODULUS"],
            umat_parameters["FRICTION_ANGLE"],
            umat_parameters["DILATANCY_ANGLE"],
            umat_parameters["COHESION"],
            umat_parameters["POISSON_RATIO"],
            umat_parameters["CUTOFF_STRENGTH"],
            umat_parameters["UNDRAINED_POISSON_RATIO"],
            umat_parameters["YIELD_FUNCTION_TYPE"]]


def create_mohr_coloumb_model(values_dict):
    density_solid = unit_weight_to_density_solid(values_dict.get("unit_weight", 18),
                                                 values_dict.get("POROSITY", 0.3),
                                                 values_dict.get("gravity", 9810)
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
    umat_constitutive_parameters = SmallStrainUmatLaw(UMAT_PARAMETERS=umat_parameters,
                                                      UMAT_NAME=values_dict.get("UMAT_NAME", "MohrCoulombUMAT.so"),
                                                      IS_FORTRAN_UMAT=True,
                                                      STATE_VARIABLES=[]
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
                              -1)

    physical_groups = gmsh_io.generate_extract_mesh(dimension, "mesh_dike_2d", ".", False, False)
    geo_data = gmsh_io.geo_data
    mesh_data = gmsh_io.mesh_data
    mesh_data['physical_groups'] = physical_groups
    total_dict = {'geo_data': geo_data, 'mesh_data': mesh_data}
    return total_dict

def define_water_boundaries(water_top, water_bottom, water_middle):
    # top layer boundary conditions multiline
    water_line_top_parameters = PhreaticMultiLineBoundary(
        is_fixed=True,
        gravity_direction=1,
        out_of_plane_direction=2,
        water_pressure=0,
        x_coordinates=water_top[0],
        y_coordinates=water_top[1],
        surfaces_assigment=water_top[2],
    )
    water_boundary_top = WaterBoundary(water_line_top_parameters, name="top_water_boundary")
    # bottom layer boundary conditions multiline
    water_line_bottom_parameters = PhreaticMultiLineBoundary(
        is_fixed=True,
        gravity_direction=1,
        out_of_plane_direction=2,
        water_pressure=0,
        x_coordinates=water_bottom[0],
        y_coordinates=water_bottom[1],
        surfaces_assigment=water_bottom[2],
    )
    water_boundary_bottom = WaterBoundary(water_line_top_parameters, name="bottom_water_boundary")
    # middle layer boundary conditions
    interpolation_type = InterpolateLineBoundary(
        surfaces_assigment=water_middle,
    )
    water_boundary_interpolate = WaterBoundary(interpolation_type, name="middle_water_boundary")

    kratos_io = KratosWaterBoundariesIO(domain="PorousDomain")
    return [kratos_io.create_water_boundary_dict(boundary) for boundary in [water_boundary_top, water_boundary_bottom, water_boundary_interpolate]]

def write_materials_dict(file_name):
    soil_1 = create_mohr_coloumb_model(TOP)
    soil_2 = create_mohr_coloumb_model(MIDDLE)
    soil_3 = create_mohr_coloumb_model(BOTTOM)

    all_materials = [soil_1, soil_2, soil_3]
    kratos_io = KratosMaterialIO(ndim=2)
    kratos_io.write_material_parameters_json(all_materials, file_name)

def add_and_save_extra_parameters(template_file, output_file, extra_parameters):
    with open(template_file, 'r') as json_file:
        data = json.load(json_file)
    for key, value in extra_parameters.items():
        data["processes"][key] += value
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4)

class TestGmshToKratos:

    def test_init(self):
        layer_1_points = [
            (-80, 0, 0),
            (-20, 0, 0),
            (0, 8, 0),
            (8, 8, 0),
            (25, 2, 0),
            (80, 2, 0),
            (80, -2, 0),
            (-80, -2, 0)
        ]
        layer_2_points = [
            (-80, -2, 0),
            (80, -2, 0),
            (80, -8, 0),
            (-80, -8, 0),
        ]
        layer_3_points = [
            (-80, -8, 0),
            (80, -8, 0),
            (80, -15, 0),
            (-80, -15, 0),
        ]
        data = define_geometry_from_gmsh([layer_1_points, layer_2_points, layer_3_points], ["TOP", "MIDDLE", "BOTTOM"])

        write_materials_dict("materials.json")
        # water line top
        WL1 = [[-80, 0, 0.5, 7, 25, 80], [7.08, 7.08, 6.08, 5.58, 2, 2], ["TOP"]]
        # water line bottom
        WL3 = [[-80, -22, 45.5, 80], [7.08, 5.91, 5.31, 5.31], ["BOTTOM"]]
        # water line middle
        WL2 = "MIDDLE"
        dictionary_water_boundaries = define_water_boundaries(WL1, WL2, WL3)
        add_and_save_extra_parameters("Project_Parameters_template.json",
                                      "Project_Parameters.json",
                                      {"loads_process_list": dictionary_water_boundaries})
        constrains_on_surfaces = {
            "names": ["top_water_boundary", "bottom_water_boundary", "middle_water_boundary"],
            "surfaces": ["TOP", "BOTTOM", "MIDDLE"]
        }
        # create the object
        gmsh_to_kratos = GmshToKratos(data)
        gmsh_to_kratos.read_gmsh_to_kratos(["TOP", "MIDDLE", "BOTTOM"],
                                           list(range(1, 3 + 1)),
                                           "test_write.mdpa",
                                           constrains_on_surfaces)
