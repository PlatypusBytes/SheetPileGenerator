import json
from shetpilegenerator.output_process import OutputProcess

def modify_template_file(problem_name):
    with open("Project_Parameters_template.json", "r") as f:
        data = json.load(f)
    # modify the boundary conditions
    data["problem_data"]["problem_name"] = problem_name
    data["solver_settings"]["model_import_settings"]["input_filename"] = problem_name
    # write the new file
    with open("test_project_parameters.json", "w") as f:
        json.dump(data, f, indent=4)


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


def add_and_save_extra_parameters(template_file, output_file, extra_parameters):
    with open(template_file, 'r') as json_file:
        data = json.load(json_file)
    for key, value in extra_parameters.items():
        data["processes"][key] += value
    with open(output_file, 'w') as json_file:
        json.dump(data,
                  json_file,
                  indent=4)