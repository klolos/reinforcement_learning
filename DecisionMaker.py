
from DecisionMaking.Constants import *
from DecisionMaking.QModel import QModel
from DecisionMaking.MDPModel import MDPModel, State, QState
from DecisionMaking.MDPCDModel import MDPCDModel
from DecisionMaking.MDPDTModel import MDPDTModel
from DecisionMaking.QDTModel import QDTModel
from DecisionMaking.Configuration import ModelConf
import json
import os.path
import logging


class DecisionMaker(object):

    def __init__(self, conf_file, training_file=None):

        self.training_file = training_file
        self.last_meas = None

        conf = ModelConf(conf_file)
        self.model_type = conf.get_model_type()
        model_conf = conf.get_model_conf()

        if self.model_type == MDP:
            self.do_vi = True
            self.model = MDPModel(model_conf)
        elif self.model_type == MDP_CD:
            self.do_vi = True
            self.model = MDPCDModel(model_conf)
        elif self.model_type == MDP_DT:
            self.do_vi = True
            self.model = MDPDTModel(model_conf)
        elif self.model_type == Q_DT:
            self.do_vi = False
            self.model = QDTModel(model_conf)
        elif self.model_type == Q_LEARNING:
            self.do_vi = False
            self.model = QModel(model_conf)

        self.install_logger()


    def install_logger(self):

        self.my_logger = logging.getLogger('DecisionMaker')
        self.my_logger.setLevel(logging.DEBUG)

        handler = logging.handlers.RotatingFileHandler(
            LOG_FILENAME, maxBytes=2*1024*1024, backupCount=5)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        handler.setFormatter(formatter)
        self.my_logger.addHandler(handler)


    def train(self):

        if self.training_file is None or not os.path.isfile(self.training_file):
            self.my_logger.error("No training file, aborting training")
            return

        self.my_logger.debug("Starting training ...")

        num_exp = 0
        skipped_exp = 0
        with open(self.training_file, 'r') as f:
            for line in f:
                old_meas, actionlist, new_meas = json.loads(line)
                self.add_network_usage(old_meas)
                self.add_network_usage(new_meas)
                action = tuple(actionlist)
                reward = self.get_reward(new_meas, action)
                self.model.set_state(old_meas)

                available_actions = self.model.get_legal_actions()
                if not action in available_actions:
                    skipped_exp += 1
                    continue

                self.model.update(action, new_meas, reward)
                num_exp += 1
                if num_exp % 100 == 0 and self.do_vi:
                    self.model.value_iteration(0.1)

                #self.my_logger.debug("Trained with experience %d" % num_exp)

        self.my_logger.debug("Trained the model with %d experiences, skipped %d" % (num_exp, skipped_exp))


    def set_state(self, measurements):

        self.add_network_usage(measurements)
        self.last_meas = measurements
        self.model.set_state(measurements)
        self.my_logger.debug("State set")


    def add_network_usage(self, measurements):

        measurements[NETWORK_USAGE] = measurements[BYTES_IN] + measurements[BYTES_OUT]


    def update(self, action, meas, reward=None):

        experience = [self.last_meas, action, meas]
        if not self.training_file is None:
            with open(self.training_file, "a") as f:
                f.write(json.dumps(experience)+'\n')
                f.flush()
                self.my_logger.debug("Recorded experience")

        self.add_network_usage(meas)
        if reward is None:
            reward = self.get_reward(meas, action)

        self.last_meas = meas
        self.model.update(action, meas, reward)


    def get_model(self):

        return self.model


    def set_splitting(self, split_crit, cons_trans=True):

        if self.model_type != MDP_DT:
            self.my_logger.error("Splitting criteria apply only to MDP_DT models!")
            return

        self.model.set_splitting(split_crit, cons_trans)


    def get_reward(self, measurements, action):
        
        vms = measurements[NUMBER_OF_VMS]
        throughput = measurements[TOTAL_THROUGHPUT]
        reward = throughput - 800 * vms

        return reward


    def get_legal_actions(self):

        return self.model.get_legal_actions()


    def suggest_action(self):

        return self.model.suggest_action()


    def set_value_iteration(self, error=0.1):

        self.model.set_value_iteration(error)


    def set_prioritized_sweeping(self, error=0.1, max_steps=200):

        self.model.set_prioritized_sweeping(error, max_steps)


    def set_stat_test(self, test):

        self.model.set_stat_test(test)

