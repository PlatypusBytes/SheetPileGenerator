import json


class OutputProcessJsonReader:

    def __init__(self, json_file_name):
        self.json_file_name = json_file_name
        self.get_output_variables()

    def get_output_variables(self):
        with open(self.json_file_name) as json_file:
            data = json.load(json_file)
        self.timesteps = data["TIME"]
        self.displacements_x = []
        self.displacements_y = []
        self.water_pressures = []
        self.coordinates = []
        data.pop("TIME")
        # get nodal outputs
        for key, item in data.items():
            key_split = key.split("_")
            if key_split[0] == "NODE":
                coordinate = [float(key_split[2]), float(key_split[4]), float(key_split[6])]
                self.coordinates.append(coordinate)
                displacement_y = item["DISPLACEMENT_Y"]
                displacement_x = item["DISPLACEMENT_X"]
                water_pressure = item["WATER_PRESSURE"]
                self.displacements_y.append(displacement_y)
                self.displacements_x.append(displacement_x)
                self.water_pressures.append(water_pressure)

    def get_values_in_timestep(self, timestep):
        # get timestep index
        timestep_index = self.timesteps.index(timestep)
        # get values
        displacement_x = [displacement[timestep_index] for displacement in self.displacements_x]
        displacement_y = [displacement[timestep_index] for displacement in self.displacements_y]
        water_pressure = [water_pressure[timestep_index] for water_pressure in self.water_pressures]
        x_coordinates = [coordinate[0] for coordinate in self.coordinates]
        y_coordinates = [coordinate[1] for coordinate in self.coordinates]
        return x_coordinates, y_coordinates, displacement_x, displacement_y, water_pressure


class OutputProcess:

    def __init__(self, output_variables, output_file_name):
        self.time_frequency = 0.005
        self.historical_value = True
        self.resultant_solution = False
        self.use_node_coordinates = True
        self.output_variables = output_variables
        self.output_file_name = output_file_name
        self.model_part_name = "PorousDomain.OutputProcess"

    def to_dict(self):
        parameters = {
            "model_part_name": self.model_part_name,
            "time_frequency": self.time_frequency,
            "historical_value": self.historical_value,
            "resultant_solution": self.resultant_solution,
            "use_node_coordinates": self.use_node_coordinates,
            "output_variables": self.output_variables,
            "output_file_name": self.output_file_name
        }
        return [{
                     "python_module": "json_output_process",
                     "kratos_module": "KratosMultiphysics",
                     "process_name": "JsonOutputProcess",
                     "Parameters": parameters
                    }]