
import json
from DecisionMaking.Constants import *
from DecisionMaking.Exceptions import *


"""
    Class to read a model configuration from a .json file
"""
class ModelConf:
    
    def __init__(self, conf_file):

        with open(conf_file, 'r') as f:
            conf_data = json.loads(f.read())

        known_models = [Q_LEARNING, MDP, MDP_CD, MDP_DT, Q_DT]
        if not MODEL in conf_data:
            raise ConfigurationError("Model not provided in configuration file: " + conf_file)
        if not conf_data[MODEL] in known_models:
            raise ConfigurationError("Unknown model type in configuration file: " + conf_file)

        self.model = conf_data[MODEL]
        self.conf  = conf_data


    def get_model_conf(self):

        return self.conf


    def get_model_type(self):

        return self.model


