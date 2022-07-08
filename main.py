"""
__author__ = "Lucas de Vries"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.config import process_config
from utils.train_utils import set_seed
import wandb
from agents import *
from pprint import pprint
from utils.dirs import create_dirs
import json

def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()
    # parse the config json file
    config = process_config(args.config)
    # set environment variable for offline runs
    os.environ["WANDB_MODE"] = "online"
    # Pass them to wandb.init
    wandb.init(config=dict(config))
    # Access all hyperparameter values through wandb.config
    config = wandb.config
    set_seed(config['seed'])
    config['run_name'] = wandb.run.name
    config['run_id'] = wandb.run.id

    #Make the folders that were previously in the expeiment folder

    config['checkpoint_dir'] = os.path.join(wandb.run.dir, 'experiments', 'checkpoints/')
    config['json_dir'] = os.path.join(wandb.run.dir, 'experiments', 'config/')
    create_dirs([config['checkpoint_dir'], config['json_dir']])
    _dir = os.path.join(config['json_dir'], str(config['run_id'])+'_parameters.json')
    with open(_dir, 'w') as f:
        json.dump(dict(config), f)
    pprint(config)
    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config['agent']]
    agent = agent_class()
    agent.run()
    agent.finalize()


if __name__ == '__main__':
    main()
