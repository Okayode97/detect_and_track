from typing import Optional
import os
import json

def incremental_average(current_average: float, new_value: float, count: int) -> float:
    # https://math.stackexchange.com/questions/106700/incremental-averaging
    if count > 2:
        return current_average + ((new_value - current_average)/count)
    
    # calculate average if less than two entries
    return sum([current_average, new_value]) / (count + 1)

def log_results(data: dict, filename: str, model_name: str) -> None:

    # if the file exists load the data, else keep empty dict
    loaded_data = {}
    if os.path.exists(f"{filename}_log.json"):
        with open(f"{filename}_log.json", "r") as f:
            loaded_data = json.load(f)
    
    # check if model_name exist in loaded data
    # if it does not creates a new dict to store the data
    if model_name not in loaded_data.keys():
        loaded_data[model_name] = {}
    
    # update existing dict with values stored in data input
    for metric in data.keys():
        if metric not in loaded_data[model_name].keys():
            loaded_data[model_name][metric] = {"avg": data[metric], "count": 1}

        else:
            loaded_data[model_name][metric]["avg"] = incremental_average(loaded_data[model_name][metric]["avg"],
                                                                         data[metric],
                                                                         loaded_data[model_name][metric]["count"])
            loaded_data[model_name][metric]["count"] += 1

    
    # write loaded_data to file
    with open(f"{filename}_log.json", "w") as f:
        json.dump(loaded_data, f, indent=4)
