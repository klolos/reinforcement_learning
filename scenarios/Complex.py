
from __future__ import division

from Constants import *
import random
import math



"""
    A simulation scenario where:
     - the percentage of reads varies periodically betwwn 50% and 100%
     - the IO operations per second vary periodically between 0.2 and 1 (assume normalized)
     - the capacity of each VM is proportional to the read percentage and also depends
       on the number of IO operations per second
     - the number of requests per second provided is the number of served requests
"""
class ComplexScenario(object):

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

        load        = self.measurements[TOTAL_LOAD]
        capacity    = self.get_current_capacity()
        served_load = min(capacity, load)

        curr_meas = dict(self.measurements)
        curr_meas[TOTAL_LOAD] = served_load
        return curr_meas


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
        load        = self.measurements[TOTAL_LOAD]
        capacity    = self.get_current_capacity()
        served_load = min(capacity, load)
        reward      = served_load - 2 * vms

        return reward


    """
        Returns the current throughput capacity of the cluster
    """
    def get_current_capacity(self):

        vms        = self.measurements[NUMBER_OF_VMS]
        read_load  = self.measurements[PC_READ_LOAD]
        io_per_sec = self.measurements[IO_PER_SEC]
        ram_size   = self.measurements[RAM_SIZE]

        if io_per_sec < 0.7:
            io_penalty = 0.0
        elif io_per_sec < 0.9:
            io_penalty = 10.0 * (io_per_sec - 0.7)
        else:
            io_penalty = 2.0

        if ram_size == 1024:
            ram_penalty = 0.3
        else:
            ram_penalty = 0.0

        capacity = (read_load * 10 - io_penalty - ram_penalty) * vms

        return capacity


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
            IO_PER_SEC       : self._get_io_per_sec(),
            TOTAL_LOAD       : self._get_load(),
            PC_READ_LOAD     : self._get_read_load(),
            TOTAL_LATENCY    : self._get_latency()
        }

        return measurements


    """
        Returns the incoming load
    """
    def get_incoming_load(self):

        return self.measurements[TOTAL_LOAD]


    """
        Returns the parameters that are relevant to the performance of the system
    """
    def get_relevant_params(self):

        return [NUMBER_OF_VMS, IO_PER_SEC, TOTAL_LOAD, PC_READ_LOAD, RAM_SIZE]


    """
        Returns the parameters that only marginally affect the performance of the system
    """
    def get_marginal_params(self):

        return [RAM_SIZE]


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


    def _get_io_per_sec(self):

        return 0.6 + 0.4 * math.sin(2 * math.pi * self.time / 195)


    def _get_ram_size(self):

        if self.time % 440 < 220:
            return 1024
        else:
            return 2048


    def _get_latency(self):

        return 0.5 + 0.5 * random.uniform(0, 1)


    def _get_free_ram(self):

        return 0.4 + 0.4 * random.uniform(0, 1)


    def _get_cpu_usage(self):

        return 0.6 + 0.3 * random.uniform(0, 1)


    def _get_storage_capacity(self):

        return random.choice([10, 20, 40])


    def _get_num_cpus(self):

        return random.choice([2, 4])


