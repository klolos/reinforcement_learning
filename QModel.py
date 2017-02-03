
from DecisionMaking.Configuration import ConfigurationError
from DecisionMaking.Constants import *
from DecisionMaking.Exceptions import *


"""
    Represents a Q-state in the Q-model
"""
class QState(object): 

    def __init__(self, action, qvalue=0):

        self.action    = action
        self.qvalue    = qvalue
        self.num_taken = 0

        action_type, action_value = action
        if action_type == ADD_VMS:
            self.action_name = "Add %s VMs   " % action_value
        elif action_type == REMOVE_VMS:
            self.action_name = "Remove %s VMs" % action_value
        else:
            self.action_name = "no op       "


    """
        Returns the action that corresponds to this q-state
    """
    def get_action(self):

        return self.action


    """
        Returns the q-value of this q-state
    """
    def get_qvalue(self):

        return self.qvalue


    """
        Sets the q-value of the q-state
    """
    def set_qvalue(self, qvalue):

        self.qvalue = qvalue


    """
        Increments the number of times this action has been taken
    """
    def incr_taken(self):

        self.num_taken += 1


    """
        Returns the number of times this action has been taken
    """
    def get_num_taken(self):

        return self.num_taken


    """
        String representation for a Q-state
    """
    def __str__(self):

        return "Action: %s \tQ-value: %2.3f  \tTaken: %d" % \
               (self.action_name, self.qvalue, self.num_taken)

    def __repr__(self):

        return str(self)


"""
    Represents a State in the Q-model
"""
class State:

    def __init__(self, parameters = []):

        self.parameters  = list(parameters)
        self.qstates     = []
        self.num_visited = 0


    """
        Returns the list of parameter names and values for this state
    """
    def get_parameters(self):

        return self.parameters


    """
        Adds a parameter name-value tuple to the list of parameters for this state
    """
    def add_new_parameter(self, name, values):

        self.parameters.append((name, values))


    """
        Returns the value of the given parameter for this state
    """
    def get_parameter(self, param):

        for par, values in self.parameters:
            if par == param:
                return values


    """
        Adds a new q-state to the list of q-states for this state
    """
    def add_qstate(self, qstate):

        self.qstates.append(qstate)


    """
        Returns a list containing all the q-states of this state
    """
    def get_qstates(self):

        return self.qstates


    """
        Returns the q-state corresponding to the given action
    """
    def get_qstate(self, action):

        for qs in self.qstates:
            if qs.get_action() == action:
                return qs


    """
        Returns the action with the highest Q-value
    """
    def get_optimal_action(self):

        best_action = self.qstates[0].get_action()
        best_qvalue = self.qstates[0].get_qvalue()
        for qs in self.qstates[1:]:
            if qs.get_qvalue() > best_qvalue:
                best_qvalue = qs.get_qvalue()
                best_action = qs.get_action()

        return best_action


    """
        Returns the maximum q-value of all the q-states of this state
    """
    def get_max_qvalue(self):

        return max([qs.get_qvalue() for qs in self.qstates])


    """
        Returns all the possible actions from this state
    """
    def get_legal_actions(self):

        return [qs.get_action() for qs in self.qstates]
    

    """
        Increments the nubmer of times this state has been visited
    """
    def visit(self):

        self.num_visited += 1


    """
        String representation for a State
    """
    def __str__(self):

        return str(self.parameters)

    def __repr__(self):

        return str(self.parameters)


    """
        Prints the details of the state and its Q-states
    """
    def print_detailed(self):

        print("%s, visited: %d" % (str(self.parameters), self.num_visited))
        for qs in self.get_qstates():
            print(qs)
    

"""
    Class that implements a Q-learning model for a Markov Decision Process.
    Only models Q-values instead of transitions and rewards.
"""
class QModel:

    """
        Sets up all the states and all needed parameters needed for the model
    """
    def __init__(self, conf):

        required_fields = [PARAMETERS, ACTIONS, DISCOUNT, INITIAL_QVALUES, LEARNING_RATE]
        for f in required_fields:
            if not f in conf:
                raise ConfigurationError("%s not provided in the configuration" % f)

        self.learning_rate = conf[LEARNING_RATE]
        self.discount      = conf[DISCOUNT]
        self.states        = [State()]
        self.current_state = None
        self._assert_modeled_params(conf)

        # create all the states of the model
        params = self._get_parameters(conf[PARAMETERS])
        for name, values in params.items():
            self._update_states(name, values)

        self._set_maxima_minima(params, conf[ACTIONS])
        self._add_qstates(conf[ACTIONS], conf[INITIAL_QVALUES])


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
    def _get_parameters(self, parameters):

        new_params = {}
        for name, par in parameters.items():

            # we convert both values and limits to pairs of limits so we can treat them uniformly
            if VALUES in par:
                if not isinstance(par[VALUES], list):
                    raise ConfigurationError("Provided values for %s must be in a list" % name)
                if len(par[VALUES]) <= 1:
                    raise ConfigurationError("At least two values must be provided for " + name)

                values = []
                for v in par[VALUES]:
                    values.append((v, v))
                new_params[str(name)] = values

            elif LIMITS in par:
                if not isinstance(par[LIMITS], list):
                    raise ConfigurationError("Provided limits for %s must be in a list" % name)
                if len(par[LIMITS]) <= 2:
                    raise ConfigurationError("At least three limits must be provided for " + name)

                values = []
                for i in range(1, len(par[LIMITS])):
                    values.append((par[LIMITS][i-1], par[LIMITS][i]))
                new_params[str(name)] = values

            else:
                raise ConfigurationError("Values or limits must be provided for parameter " + name)

        return new_params


    """
        Initializes the current state with the given measurements
    """
    def set_state(self, measurements):
        
        self.current_state = self._get_state(measurements)

    
    """
        Returns all the possible actions from the current state
    """
    def get_legal_actions(self):

        if self.current_state is None:
            raise StateNotSetError()

        return self.current_state.get_legal_actions()


    """
        Returns the optimal next action derived from the q-values of the current state
    """
    def suggest_action(self):

        if self.current_state is None:
            raise StateNotSetError()

        return self.current_state.get_optimal_action()


    """
        Extends the current states to include all the possible values of the
        given parameter, multiplying their number with the number of values
        of the parameter.
    """
    def _update_states(self, name, values):

        new_states = []
        for value in values:
            for s in self.states:
                new_state = State(s.get_parameters())
                new_state.add_new_parameter(name, value)
                new_states.append(new_state)

        self.states = new_states


    """
        Stores the maxima and minima for the parameters that have actions that
        need to be limited
    """
    def _set_maxima_minima(self, parameters, actions):

        if ADD_VMS in actions or REMOVE_VMS in actions:
            vm_values = parameters[NUMBER_OF_VMS]
            self.max_VMs = max([max(x) for x in vm_values])
            self.min_VMs = min([min(x) for x in vm_values])

        # TODO the rest of the actions


    """
        Adds the given actions to all the states
    """
    def _add_qstates(self, actions, qvalue):

        for action_type, values in actions.items():
            for action_value in values:
                action = (action_type, action_value)
                for s in self.states:
                    if self._is_permissible(s, action):
                        s.add_qstate(QState(action, qvalue))

    
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
        Updates the Q-value accordingly for a transition to the state deriving
        from the given measurements, after performing the given action and 
        receiving the given reward.
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
        a = self.learning_rate
        g = self.discount
        qvalue = (1 - a)*qstate.get_qvalue() + a*(reward + g*new_state.get_max_qvalue())
        qstate.set_qvalue(qvalue)
        qstate.incr_taken()

        self.current_state = new_state


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


