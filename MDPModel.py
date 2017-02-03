
from __future__ import division
from DecisionMaking.Configuration import ConfigurationError
from DecisionMaking.Constants import *
from DecisionMaking.Exceptions import *


"""
    Represents a Q-state in the MDP model
"""
class QState(object): 

    def __init__(self, action, num_states, qvalue = 0.0):

        self.action      = action
        self.num_taken   = 0
        self.qvalue      = qvalue
        self.transitions = [0] * num_states
        self.rewards     = [0] * num_states
        self.num_states  = num_states

        action_type, action_value = action
        if action_type == ADD_VMS:
            self.action_name = "Add %s VMs   " % action_value
        elif action_type == REMOVE_VMS:
            self.action_name = "Remove %s VMs" % action_value
        else:
            self.action_name = "no op       "
        # TODO the rest of the actions


    """
        Updates the transition and reward estimations after the given transition
    """
    def update(self, new_state, reward):

        self.num_taken += 1
        state_num = new_state.get_state_num()
        self.transitions[state_num] += 1
        self.rewards[state_num] += reward


    """
        Returns the action that corresponds to this Q-state
    """
    def get_action(self):

        return self.action


    """
        Returns the q-value of the q-state
    """
    def get_qvalue(self):

        return self.qvalue


    """
        Returns true if the estimated transition probability to the given state is non zero
    """
    def has_transition(self, state_num):

        return self.transitions[state_num] > 0


    """
        Returns the estimated transition probability to the given state.
        Returns 1 over the number of states if the action has never been taken
    """
    def get_transition(self, state_num):

        if self.num_taken == 0:
            return 1 / self.num_states
        else:
            return self.transitions[state_num] / self.num_taken


    """
        Returns the number of recorded transitions to the given state
    """
    def get_num_transitions(self, state_num):

        return self.transitions[state_num]


    """
        Returns the estimated reward after taking this action
    """
    def get_reward(self, state_num):

        if self.transitions[state_num] == 0:
            return 0.0
        else:
            return self.rewards[state_num] / self.transitions[state_num]


    """
        The qvalue for this action
    """
    def set_qvalue(self, qvalue):

        self.qvalue = qvalue


    """
        The number of times this action has been taken
    """
    def get_num_taken(self):

        return self.num_taken


    """
        Returns a list containing the number of transitions to each state
    """
    def get_transitions(self):

        return list(self.transitions)


    """
        Returns a list containing the total rewards gained after transitioning to each state
    """
    def get_rewards(self):

        return list(self.rewards)


    """
        String representatin for a Q-state
    """
    def __str__(self):

        return "Action: %s \tQ-value: %2.3f  \tTaken: %d" % \
               (self.action_name, self.qvalue, self.num_taken)

    def __repr__(self):

        return str(self)


"""
    Represents a state in the MDP model
"""
class State(object):

    def __init__(self, parameters = [], state_num = 0, initial_value = 0, num_states = 0):

        self.parameters  = list(parameters)
        self.qstates     = []
        self.state_num   = state_num
        self.num_states  = num_states
        self.value       = 0
        self.best_qstate = None
        self.num_visited = 0


    """
        Increments the number of times the state has been visited
    """
    def visit(self):

        self.num_visited += 1


    """
        The unique number of the state in the MDP model
    """
    def get_state_num(self):

        return self.state_num


    """
        Sets the total number of states in the model
    """
    def set_num_states(self, num_states):

        self.num_states = num_states


    """
        The current value of the state
    """
    def get_value(self):

        return self.value


    """
        Returns the Q-state with the highest q-value
    """
    def get_best_qstate(self):

        return self.best_qstate


    """
        Returns the optimal action for this state
    """
    def get_optimal_action(self):

        return self.best_qstate.get_action()


    """
        Retuns the number of times the optimal action has been executed
    """
    def best_action_num_taken(self):

        return self.best_qstate.get_num_taken()


    """
        Updates the value of the state based on the values of its Q-states
    """
    def update_value(self):

        self.best_qstate = self.qstates[0]
        self.value       = self.qstates[0].get_qvalue()
        for qs in self.qstates:
            if qs.get_qvalue() > self.value:
                self.best_qstate = qs
                self.value       = qs.get_qvalue()


    """
        Returns a list containing the names and values of the parameters for this state
    """
    def get_parameters(self):

        return self.parameters


    """
        Adds a new parameter-value pair to the list of parameters that this state represents
    """
    def add_new_parameter(self, name, values):

        self.parameters.append((name, values))


    """
        Returns the value for the given parameter
    """
    def get_parameter(self, param):

        for par, values in self.parameters:
            if par == param:
                return values

        return None


    """
        Adds a new Q-state to this state
    """
    def add_qstate(self, qstate):

        self.qstates.append(qstate)
        if self.best_qstate is None:
            self.best_qstate = qstate


    """
        Returns the list of Q-states for this state
    """
    def get_qstates(self):

        return self.qstates


    """
        Returns the Q-state that corresponds to the given action
    """
    def get_qstate(self, action):

        for qs in self.qstates:
            if qs.get_action() == action:
                return qs


    """
        Returns a dict that contains the maximum transition probability for any action
        for all the states that there is a non-zero transition probability
    """
    def get_max_transitions(self):

        transitions = {}
        for i in range(self.num_states):
            for qs in self.qstates:
                if qs.has_transition(i):
                    if i in transitions:
                        transitions[i] = max(transitions[i], qs.get_transition(i))
                    else:
                        transitions[i] = qs.get_transition(i)

        return transitions


    """
        Returns all the possible actions from this state
    """
    def get_legal_actions(self):

        return [qs.get_action() for qs in self.qstates]


    """
        String representation for a state
    """
    def __str__(self):

        return "%d: %s" % (self.state_num, str(self.parameters))

    def __repr__(self):

        return str(self)


    """
        Prints the details of the state and its q-states
    """
    def print_detailed(self):

        print("%d: %s, visited: %d" % (self.state_num, str(self.parameters), self.num_visited))
        for qs in self.get_qstates():
            print(qs)


"""
    Class that represents a full Markov Decision Process model.
"""
class MDPModel:

    """
        Creates a model from a given configuration dict
    """
    def __init__(self, conf):

        required_fields = [PARAMETERS, ACTIONS, DISCOUNT, INITIAL_QVALUES]
        for f in required_fields:
            if not f in conf:
                raise ConfigurationError("%s not provided in the configuration" % f)

        self.discount      = conf[DISCOUNT]
        self.states        = [State()]
        self.index_params  = []
        self.index_states  = list(self.states)
        self.current_state = None
        self.update_error  = 0.01
        self.max_updates   = 100

        self._assert_modeled_params(conf)
        parameters = self._get_params(conf[PARAMETERS])
        
        # create all the states of the model
        for name, param in parameters.items():
            self.index_params.append((name, param[VALUES]))
            self._update_states(str(name), param)

        # set the final number of states to all states
        num_states = len(self.states)
        for s in self.states:
            s.set_num_states(num_states)

        self._set_maxima_minima(parameters, conf[ACTIONS])
        self._add_qstates(conf[ACTIONS], conf[INITIAL_QVALUES])

        # set the default update algorithm
        self.update_algorithm  = SINGLE_UPDATE

        # initialize the reverse transition indexes and priorities for prioritized sweeping
        self.reverse_transitions = []
        self.priorities = [0] * len(self.states)
        for i in range(len(self.states)):
            self.reverse_transitions.append({})


    """
        Asserts that action dependent parameters are being modeled
    """
    def _assert_modeled_params(self, conf):
        
        if ADD_VMS in conf[ACTIONS] or REMOVE_VMS in conf[ACTIONS]:
            if not NUMBER_OF_VMS in conf[PARAMETERS]:
                raise ConfigurationError("Add/Remove VM actions require %s parameter" % NUMBER_OF_VMS)

        # TODO the rest of the actions


    """
        The values of each model parameter are represented as a [min, max] touple.
        This method asserts that values are provided for all the parameters and converts
        distinct values to [min, max] touples.
    """
    def _get_params(self, parameters):

        new_pars = {}
        for name, par in parameters.items():

            new_pars[name] = {}

            # we convert both values and limits to pairs of limits so we can treat them uniformly
            if VALUES in par:
                if not isinstance(par[VALUES], list):
                    raise ConfigurationError("Provided values for %s must be in a list" % name)
                if len(par[VALUES]) <= 1:
                    raise ConfigurationError("At least two values must be provided for " + name)

                values = []
                for v in par[VALUES]:
                    values.append((v, v))
                new_pars[name][VALUES] = values

            elif LIMITS in par:
                if not isinstance(par[LIMITS], list):
                    raise ConfigurationError("Provided limits for %s must be in a list" % name)
                if len(par[LIMITS]) <= 2:
                    raise ConfigurationError("At least three limits must be provided for " + name)

                values = []
                for i in range(1, len(par[LIMITS])):
                    values.append((par[LIMITS][i-1], par[LIMITS][i]))
                new_pars[name][VALUES] = values

            if not VALUES in new_pars[name]:
                raise ConfigurationError("Values or limits must be provided for parameter " + name)

        return new_pars


    """
        Initializes the current state based on the given measurements
    """
    def set_state(self, measurements):

        self.current_state = self._get_state(measurements)


    """
        Extends the current states to include all the possible values of the
        given parameter, multiplying their number with the number of values
        of the parameter.
    """
    def _update_states(self, name, new_parameter):

        state_num = 0
        new_states = []
        for value in new_parameter[VALUES]:
            for s in self.states:
                new_state = State(s.get_parameters(), state_num)
                new_state.add_new_parameter(name, value)
                new_states.append(new_state)
                state_num += 1

        self.states = new_states


    """
        Stores the maxima and minima for the parameters that have actions which
        need to be limited
    """
    def _set_maxima_minima(self, parameters, actions):

        if ADD_VMS in actions or REMOVE_VMS in actions:
            vm_values = parameters[NUMBER_OF_VMS][VALUES]
            self.max_VMs = max([max(x) for x in vm_values])
            self.min_VMs = min([min(x) for x in vm_values])

        # TODO the rest of the actions


    """
        Adds the given actions to all the states
    """
    def _add_qstates(self, actions, initial_qvalue):

        num_states = len(self.states)
        for action_type, values in actions.items():
            for action_value in values:
                action = (action_type, action_value)
                for s in self.states:
                    if self._is_permissible(s, action):
                        s.add_qstate(QState(action, num_states, initial_qvalue))

        for s in self.states:
            s.update_value()


    """
        Returns true if we are allowed to take that action from that state
    """
    def _is_permissible(self, state, action):
        
        action_type, action_value = action
        if action_type == ADD_VMS:
            param_values = state.get_parameter(NUMBER_OF_VMS)
            return max(param_values) + action_value <= self.max_VMs
            
        elif action_type == REMOVE_VMS:
            param_values = state.get_parameter(NUMBER_OF_VMS)
            return min(param_values) - action_value >= self.min_VMs

        # TODO the rest of the actions
        return True


    """
        Returns the state that corresponds to given set of measurementes
    """
    def _get_state(self, measurements): # TODO this with indexing

        for name, values in self.index_params:
            if not name in measurements:
                raise ParameterError("Missing measurement: " + name)
        
        for s in self.states:
            matches = True
            for name, values in s.get_parameters():
                min_v, max_v = values
                if measurements[name] < min_v or measurements[name] > max_v:
                    matches = False
                    break
            if matches:
                return s


    """
        Suggest the next action based on the greedy criterion
    """
    def suggest_action(self):
        
        if self.current_state is None:
            raise StateNotSetError()
        
        return self.current_state.get_optimal_action()


    """
        Returns all the legal actions from the current_state
    """
    def get_legal_actions(self):
        
        if self.current_state is None:
            raise StateNotSetError()
        
        return self.current_state.get_legal_actions()


    """
        Stops the model from performing any updates to q-values
    """
    def set_no_update(self):

        self.update_algorithm = NO_UPDATE


    """
        Update only the value of the starting state after each transition
    """
    def set_single_update(self):

        self.update_algorithm = SINGLE_UPDATE


    """
        Perform a full value iteration after each transition
    """
    def set_value_iteration(self, update_error):

        self.update_algorithm = VALUE_ITERATION
        self.update_error     = update_error


    """
        Perform prioritized sweeping after each transition
    """
    def set_prioritized_sweeping(self, update_error, max_updates):

        self.update_algorithm = PRIORITIZED_SWEEPING
        self.update_error     = update_error
        self.max_updates      = max_updates


    """
        Updates the model after taking the given action and ending up in the
        state corresponding to the given measurements.
    """
    def update(self, action, measurements, reward):

        if self.current_state is None:
            raise StateNotSetError()
        
        self.current_state.visit()
        qstate = self.current_state.get_qstate(action)
        if qstate is None:
            # TODO log
            return

        new_state = self._get_state(measurements)
        qstate.update(new_state, reward)
        #print("old state: %s" % self.current_state)
        #print("new state: %s" % new_state)

        if self.update_algorithm == SINGLE_UPDATE:
            self._q_update(qstate)
            self.current_state.update_value()
        elif self.update_algorithm == VALUE_ITERATION:
            self.value_iteration()
        elif self.update_algorithm == PRIORITIZED_SWEEPING:
            self.prioritized_sweeping()

        self.current_state = new_state

    """
        Runs a single update for the Q-value of the given state-action pair.
    """
    def _q_update(self, qstate):

        new_qvalue = 0
        for i in range(len(self.states)):
            t = qstate.get_transition(i)
            r = qstate.get_reward(i)
            new_qvalue += t * (r + self.discount * self.states[i].get_value())

        qstate.set_qvalue(new_qvalue)


    """
        Recalculates the value of all the q-states of the given state,
        and updates the value of the state accordingly.
    """
    def _v_update(self, state):

        for qs in state.get_qstates():
            self._q_update(qs)

        state.update_value()


    """
        Runs the value iteration algorithm on the model.
    """
    def value_iteration(self, error=None):
        
        if error is None:
            error = self.update_error

        repeat = True
        while (repeat):
            repeat = False
            for s in self.states:
                old_value = s.get_value()
                self._v_update(s)
                new_value = s.get_value()
                if abs(old_value - new_value) > error:
                    repeat = True


    """
        Runs prioritized sweeping starting from the given state.
    """
    def prioritized_sweeping(self, initial_state=None, error=None, max_updates=None, debug=False):

        if self.current_state is None and initial_state is None:
            raise StateNotSetError()
        
        if initial_state is None:
            initial_state = self.current_state
        if error is None:
            error = self.update_error
        if max_updates is None:
            max_updates = self.max_updates

        # transition probabilities have changed for the initial state
        max_transitions = initial_state.get_max_transitions()
        initial_s_num = initial_state.get_state_num()
        for s_num, t in max_transitions.items():
            self.reverse_transitions[s_num][initial_s_num] = t

        s = initial_state
        num_updates = 0
        for i in range(max_updates):

            num_updates += 1

            # update the state value
            old_value = s.get_value()
            self._v_update(s)
            new_value = s.get_value()
            delta = abs(new_value - old_value)

            # update the priorities of the predecessors
            rev_transitions = self.reverse_transitions[s.get_state_num()]
            for s_num, t in rev_transitions.items():
                self.priorities[s_num] = max(t * delta, self.priorities[s_num])

            # zero the updated state's priority
            self.priorities[s.get_state_num()] = 0
            if debug:
                print("sweeping for %d, delta = %f" % (s.get_state_num(), delta))
                print(self.reverse_transitions)
                print(self.priorities)

            # choose the next max priority state
            # TODO with Priority Queue - but needs to support item removal
            max_index = 0
            max_priority = 0
            for i in range(len(self.priorities)):
                if self.priorities[i] > max_priority:
                    max_priority = self.priorities[i]
                    max_index = i

            # stop if the priority gets below the supplied limit
            if max_priority <= error:
                if debug:
                    print("max_priority = %s, stopping" % max_priority)
                break
    
            s = self.states[max_index]


    """
        Returns a list of the names of all the parameters in the states of the model
    """
    def get_parameters(self):
    
        return [name for name, values in self.index_params]


    """
        Prints the states of the model.
        If detailed is True it also prints the q-states
    """
    def print_model(self, detailed=False):

        for s in self.states:
            if detailed:
                s.print_detailed()
                print("")
            else:
                print(s)


    """
        Returns the percentage of actions that have never been taken
    """
    def get_percent_not_taken(self):

        total = 0
        not_taken = 0
        for s in self.states:
            for qs in s.get_qstates():
                total += 1
                if qs.get_num_taken() == 0:
                    not_taken += 1

        return not_taken / total



