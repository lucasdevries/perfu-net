import os

import json
from easydict import EasyDict
from pprint import pprint
import wandb
from utils.dirs import create_dirs

import yaml

def yaml_load(filename):
    with open(filename, 'r') as f:
        fc = EasyDict(yaml.safe_load(f))
    return fc

def process_yaml(yaml_file):
    """
    Get the json file then editing the path of the experiments folder, creating the dir and return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config = yaml_load(yaml_file)
    try:
        config.general.summary_dir = os.path.join("experiments", config.general.exp_name, "summaries/")
        config.general.checkpoint_dir = os.path.join("experiments", config.general.exp_name, "checkpoints/")
        config.general.out_dir = os.path.join("experiments", config.general.exp_name, "out/")
        config.general.json_dir = os.path.join("experiments", config.general.exp_name, "config")
        create_dirs([config.general.summary_dir, config.general.checkpoint_dir, config.general.out_dir, config.general.json_dir])
        _dir = os.path.join(config.general.json_dir, config.general.exp_name+'_parameters.yaml')
        with open(_dir, 'w') as f:
            yaml.dump(config, f, indent=4)

    except AttributeError:
        print("ERROR!!..Please provide the exp_name in yaml file..")
        exit(-1)

    return config

def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
        except ValueError as e:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config, config_dict


def process_config(json_file):
    """
    Get the json file then editing the path of the experiments folder, creating the dir and return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    config, _ = get_config_from_json(json_file)
    print(" THE Configuration of your experiment ..")
    print(" *************************************** ")

    # try:
    #     # config.summary_dir = os.path.join("experiments", config.exp_name, "summaries/")
    #     # config.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints/")
    #     # config.out_dir = os.path.join("experiments", config.exp_name, "out/")
    #     # config.json_dir = os.path.join("experiments", config.exp_name, "config")
    #     # create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.json_dir])
    #     # _dir = os.path.join(config.json_dir, config.exp_name+'_parameters.json')
    #     # with open(_dir, 'w') as f:
    #     #     json.dump(config, f, indent=4)
    #
    # except AttributeError:
    #     print("ERROR!!..Please provide the exp_name in json file..")
    #     exit(-1)

    return config