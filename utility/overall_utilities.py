import os
import json
import pandas
import numpy as np

def save_dict_to_json(dictionary:dict, path:str) -> None:
    with open(path, 'w') as file:
        json.dump(dictionary, file, indent=4)

def load_dict_from_json(path:str) -> dict:
    with open(path, 'r') as file:
        return json.load(file)
    
def save_plot(plot, folder, filename, format='png'):
    """Saves the plot in the specified folder with the specified filename,
    in the specified format, using the savefig method of the plot object.

    Args:
        plot (_type_): the plot object to save
        folder (str): the path of the folder in which to save the plot
        filename (str): the name of the file
        format (str, optional): the format of the image file. Defaults to 'png'.
    """
    if not filename.endswith(format):
        filename = filename.split('.')[0] + '.' + format

    fig_path = os.path.join(folder, filename)
    plot.savefig(fig_path, format=format)