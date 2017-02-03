
from __future__ import division

from DecisionMaking.Constants import *
import random
import math



"""
    A simulation scenario where the percentage of reads varies periodically from 50% to 100%
    and the capacity of each VM is proportional to that percentage.
"""
class ReadLoadScenario(object):

    def __init__(self, training_steps, load_period=250, init_vms=10, min_vms=1, max_vms=20):

        self.time = 0
        self.load_period = load_period
        self.training_steps = training_steps
        self.MIN_VMS = min_vms
        self.MAX_VMS = max_vms
        self.measurements = self._get_measurements(init_vms)


    """
        Returns the measurements for the current state of the system
    """
    def get_current_measurements(self):

        return dict(self.measurements)


    """
        Executes the given action, updating the current measurements accordingly
    """
    def execute_action(self, action):

        self.time += 1
        num_vms = self.measurements[NUMBER_OF_VMS]
        action_type, action_value = action
        if action_type == ADD_VMS:
            num_vms += action_value
        if action_type == REMOVE_VMS:
            num_vms -= action_value
        if num_vms < self.MIN_VMS:
            num_vms = self.MIN_VMS
        if num_vms > self.MAX_VMS:
            num_vms = self.MAX_VMS

        self.measurements = self._get_measurements(num_vms)
        reward = self._get_reward(action)
        return reward


    """
        Returns the reward gained by executing the given action under the current measurements
    """
    def _get_reward(self, action):

        vms         = self.measurements[NUMBER_OF_VMS]
        load        = self.measurements[INCOMING_LOAD]
        capacity    = self.get_current_capacity()
        served_load = min(capacity, load)
        reward      = served_load - 3 * vms

        return reward


    """
        Returns the current throughput capacity of the cluster
    """
    def get_current_capacity(self):

        vms         = self.measurements[NUMBER_OF_VMS]
        read_load   = self.measurements[PC_READ_LOAD]
        capacity    = read_load * 10 * vms

        return capacity


    """
        Returns the parameters that are relevant to the behaviour of the system
    """
    def get_relevant_params(self):

        return [NUMBER_OF_VMS, PC_READ_LOAD, INCOMING_LOAD]


    """
        Returns the measurements for the given number of vms and time
    """
    def _get_measurements(self, num_vms):

        measurements = {
            NUMBER_OF_VMS    : num_vms,
            RAM_SIZE         : self._get_ram_size(),
            NUMBER_OF_CPUS   : self._get_num_cpus(),
            STORAGE_CAPACITY : self._get_storage_capacity(),
            PC_FREE_RAM      : self._get_free_ram(),
            PC_CPU_USAGE     : self._get_cpu_usage(),
            IO_REQS          : self._get_io_per_sec(),
            INCOMING_LOAD    : self._get_load(),
            PC_READ_LOAD     : self._get_read_load(),
            TOTAL_LATENCY    : self._get_latency()
        }

        return measurements


    """
        Methods that return the current values for each of the parameters
    """
    def _get_load(self):

        # double the frequency during the testing period
        if self.time <= self.training_steps:
            return 50.0 + 50 * math.sin(2 * math.pi * self.time / self.load_period)
        else: 
            return 50.0 + 50 * math.sin(2 * math.pi * self.time * 2 / self.load_period)


    def _get_read_load(self):

        return 0.75 + 0.25 * math.sin(2 * math.pi * self.time / 340)


    def _get_latency(self):

        return 0.5 + 0.5 * random.uniform(0, 1)


    def _get_free_ram(self):

        return 0.4 + 0.4 * random.uniform(0, 1)


    def _get_cpu_usage(self):

        return 0.6 + 0.3 * random.uniform(0, 1)


    def _get_io_per_sec(self):

        return 1000 + 800 * random.uniform(0, 1)


    def _get_storage_capacity(self):

        return random.choice([10, 20])


    def _get_num_cpus(self):

        return random.choice([2, 4])


    def _get_ram_size(self):

        return random.choice([1024, 2048])




