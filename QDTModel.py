
from __future__ import division
from scipy import stats, std
import numpy
import warnings
from DecisionMaking.Configuration import ConfigurationError
from DecisionMaking.Constants import *
from DecisionMaking.QModel import QState
from pprint import pprint


"""
    Class to represent a q-state in a Decision Tree MDP model.
"""
class QStateDT(QState):

    def __init__(self, action, qvalue=0):

        super(QStateDT, self).__init__(action, qvalue)
        self.incr_measurements = []
        self.decr_measurements = []


    """
        Returns the recorded transitions that increased the q-value.
    """
    def get_incr_measurements(self):

        return self.incr_measurements


    """
        Returns the recoreded transitions that decreased the q-value.
    """
    def get_decr_measurements(self):

        return self.decr_measurements


    """
        Stores a transition that increased the q-value.
    """
    def store_incr_measurement(self, measurement):

        self.incr_measurements.append(measurement)


    """
        Stores a transition that decreased the q-value.
    """
    def store_decr_measurement(self, measurement):

        self.decr_measurements.append(measurement)


    """
        String representation for a Q-state
    """
    def __str__(self):

        meas_str = "\tIncr: %d, Decr: %d" % (len(self.incr_measurements), 
                                             len(self.decr_measurements))
        #meas_str += "\nIncr:"
        #for i in self.incr_measurements:
        #    meas_str += "\n" + str(i)
        #meas_str += "\nDecr:"
        #for d in self.decr_measurements:
        #    meas_str += "\n" + str(d)
        return super(QStateDT, self).__str__() + meas_str

    def __repr__(self):

        return str(self)


"""
    A leaf node in the decision tree, and one of the states of the MDP.
"""
class LeafNode(object):

    def __init__(self, parent, model, actions, qvalues=None):
        
        self.parent          = parent
        self.actions         = actions
        self.initial_qvalues = qvalues
        self.model           = model
        self.value           = 0

        self.qstates = []
        for name, values in actions.items():
            for value in values:
                action = (name, value)
                if qvalues is None:
                    qstate = QStateDT(action, 0)
                else:
                    qstate = QStateDT(action, qvalues[action])
                self.qstates.append(qstate)

        self.update_value()


    """
        Sets the q-values for all the q-states to the given value
    """
    def set_all_qvalues(self, qvalue):

        for qs in self.get_qstates():
            qs.set_qvalue(qvalue)


    """
        This is a leaf node.
    """
    def is_leaf(self):

        return True


    """
        Replaces this leaf node with a decision node in the decision tree
        and updates all the MDP states accordingly.
    """
    def split(self, param, limits, qvalues=None):

        if qvalues is None:
            qvalues = {}
            for qs in self.get_qstates():
                qvalues[qs.get_action()] = qs.get_qvalue()

        # remove the leaf node from the model
        self.model.remove_state(self)

        # create the decision node to replace it and add it to the model
        d_node = DecisionNode(self.parent, self.model, param, limits, self.actions, qvalues)
        new_states = d_node.get_leaves()
        self.model.add_states(new_states)
        self.parent.replace_node(self, d_node)
        return new_states


    """
        The optimal action is the one with the biggest Q value
    """
    def get_optimal_action(self):

        max_value   = float("-inf")
        best_action = None
        for q in self.qstates:
            if max_value < q.get_qvalue():
                max_value   = q.get_qvalue()
                best_action = q.get_action()

        return best_action


    """
        Returns all the possible actions from this state
    """
    def get_legal_actions(self):

        return [qs.get_action() for qs in self.qstates]


    """
        Returns all the leaves contained in this subtree, which is itself.
    """
    def get_leaves(self):
        
        return [self]


    """
        Returns the state on this subtree that corresponds to the given measurements.
    """
    def get_state(self, measurements):

        return self


    """
        Returns all the qstates for all the actions from this state
    """
    def get_qstates(self):

        return self.qstates


    """
        Returns the qstate that corresponds to the given action from this state
    """
    def get_qstate(self, action):

        for qs in self.qstates:
            if qs.get_action() == action:
                return qs


    """
        Return the value of the state
    """
    def get_value(self):

        return self.value


    """
        Updates the value of the state to be equal to the value of the best qstate
    """
    def update_value(self):

        self.value = max([qs.get_qvalue() for qs in self.qstates])


    """
        String representation for a leaf node
    """
    def __str__(self):

        return "Q-Model State"

    def __repr__(self):

        return str(self)


    """
        Prints the node along with its Q-states.
    """
    def print_detailed(self):

        print(self)
        for qs in self.get_qstates():
            print(qs)



"""
    A decision node in the decision tree. This will only hold references to other nodes
    and does not represent a state of the MDP.
"""
class DecisionNode(object):
    
    def __init__(self, parent, model, parameter, limits, actions, initial_qvalues):

        self.parent     = parent
        self.parameter  = parameter
        self.limits     = limits
        self.model      = model

        self.children = []
        num_children = len(limits) + 1
        for i in range(num_children):
            l = LeafNode(self, model, actions, initial_qvalues)
            self.children.append(l)


    """
        This is not a leaf node
    """
    def is_leaf(self):

        return False


    """
        Replaces the given child node with the new one.
        This happens when one of the child nodes is split.
    """
    def replace_node(self, old_node, new_node):

        for i, c in enumerate(self.children):
            if c is old_node:
                self.children[i] = new_node
                return

        raise InternalError("Tried to replace a node that did not exist")


    """
        Splits all the children nodes.
        This should only be used when initializing the model with multiple parameters.
    """
    def split(self, param, limits):

        for c in self.children:
            c.split(param, limits)


    """
        Returns all the leaves in the current subtree
    """
    def get_leaves(self):

        leaves = []
        for c in self.children:
            leaves += c.get_leaves()

        return leaves


    """
        Returns the state on this subtree that corresponds to the given measurements.
    """
    def get_state(self, measurements):

        if not self.parameter in measurements:
            raise ParameterError("Missing measurement: " + self.parameter)
        
        m = measurements[self.parameter]
        for i, l in enumerate(self.limits):
            if m < l:
                return self.children[i].get_state(measurements)

        return self.children[-1].get_state(measurements)



"""
    Class that represents a Q-Learning model with a decision tree state structure.
"""
class QDTModel:

    """
        Creates a model from a given configuration dict
    """
    def __init__(self, conf):

        required_fields = [INITIAL_PARAMETERS, PARAMETERS, ACTIONS, DISCOUNT, LEARNING_RATE,
                           INITIAL_QVALUES, SPLIT_ERROR, MIN_MEASUREMENTS]
        for f in required_fields:
            if not f in conf:
                raise ConfigurationError("%s not provided in the configuration" % f)

        self.discount         = conf[DISCOUNT]
        self.learning_rate    = conf[LEARNING_RATE]
        self.parameters       = list(conf[PARAMETERS])
        self.min_measurements = max(conf[MIN_MEASUREMENTS], 1)
        self.split_error      = conf[SPLIT_ERROR]
        self.root             = LeafNode(self, self, conf[ACTIONS])
        self.root.set_all_qvalues(conf[INITIAL_QVALUES])
        self.current_state    = None
        self.current_meas     = None
        self.update_qvalues   = True
        self.reuse_meas       = False
        self.states           = [self.root]
        self.transition_data  = []
        self.splits           = {}

        # create all the initial decision nodes of the model
        parameters = self._get_parameters(conf[INITIAL_PARAMETERS])
        for name, limits in parameters.items():
            self.root.split(name, limits)

        # initialize the split counters
        self.allow_splitting = True
        for p in self.parameters:
            self.splits[p] = 0

        # initialize the reverse transition indexes and priorities for prioritized sweeping
        self.reverse_transitions = []
        self.priorities = [0] * len(self.states)
        for i in range(len(self.states)):
            self.reverse_transitions.append({})


    """
        Configure the defined limits or values for the initial parameters
        so that they can be used by a decision node.
    """
    def _get_parameters(self, parameters):

        new_pars = {}
        for name, par in parameters.items():

            # for discrete values we define the midpoint as the margin
            if VALUES in par:
                if not isinstance(par[VALUES], list):
                    raise ConfigurationError("Provided values for %s must be in a list" % name)
                if len(par[VALUES]) <= 1:
                    raise ConfigurationError("At least two values must be provided for " + name)

                limits = []
                for i in range(len(par[VALUES]) - 1):
                    limits.append((par[VALUES][i] + par[VALUES][i+1]) / 2)
                new_pars[name] = limits

            # for continuous values we just ignore the outer margins
            elif LIMITS in par:
                if not isinstance(par[LIMITS], list):
                    raise ConfigurationError("Provided limits for %s must be in a list" % name)
                if len(par[LIMITS]) <= 2:
                    raise ConfigurationError("At least three limits must be provided for " + name)

                new_pars[name] = par[LIMITS][1:-1]

            else:
                raise ConfigurationError("Values or limits must be provided for parameter " + name)

        return new_pars


    """
        Replaces the root node with the given decision node.
        This should happen when the root node splits.
    """
    def replace_node(self, old_node, new_node):

        if not old_node is self.root:
            raise InternalError("Tried to replace the root node with a different initial node")
    
        self.root = new_node


    """
        Initializes the current state based on the given measurements
    """
    def set_state(self, measurements):

        self.current_meas  = measurements
        self.current_state = self.root.get_state(measurements)


    """
        Allow updates to q-values
    """
    def set_update_qvalues(self, update=True):

        self.update_qvalues = update


    """
        Removes the state with the given state_num from the model
    """
    def remove_state(self, state):

        state_num = None
        for i in range(len(self.states)):
            if self.states[i] is state:
                state_num = i

        if state_num is None:
            raise InternalError("Tried to remove a state that did not exist")

        del self.states[state_num]


    """
        Stores the given transition data to be used later on for retraining
    """
    def store_transition_data(self, data):

        self.transition_data += data


    """
        Adds new states to the model
    """
    def add_states(self, states):

        self.states += states


    """
        Suggest the optimal action to take from the current state
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
        Updates the model after taking the given action and ending up in the
        state corresponding to the given measurements.
    """
    def update(self, action, measurements, reward, debug=False):

        if self.current_meas is None:
            raise StateNotSetError()

        # Recalculate the current state in case it was removed
        # TODO move this to the splitting function, no need to do this every update
        self.current_state = self.root.get_state(self.current_meas)

        # update the q-value in the current state
        new_state = self.root.get_state(measurements)
        self._q_update(action, reward, self.current_meas, new_state)

        # consider splitting the initial_state
        if self.allow_splitting:
            self.split_mid_point(debug=debug)

        # update the current state and store the last measurements
        self.current_state = new_state
        self.current_meas  = measurements


    """
        Runs a single update for the Q-value of the given state-action pair.
    """
    def _q_update(self, action, reward, measurements, new_state, initial_state=None):

        if initial_state is None:
            initial_state = self.current_state

        # update the qvalue
        qstate  = initial_state.get_qstate(action)
        qvalue  = qstate.get_qvalue()
        a       = self.learning_rate
        g       = self.discount
        delta_q = a * (reward + g * new_state.get_value() - qvalue)
        if self.update_qvalues:
            qstate.set_qvalue(qvalue + delta_q)

        # store the measurements in the q-state
        if (delta_q > 0):
            qstate.store_incr_measurement((measurements, delta_q))
        else:
            qstate.store_decr_measurement((measurements, delta_q))

        qstate.incr_taken()
        initial_state.update_value()


    """
        Allow or prevent the decision tree from splitting its nodes
    """
    def set_allow_splitting(self, allow_splitting=True):

        self.allow_splitting = allow_splitting


    """
        Attempts to split the current state in the midpoint between transitions that
        would increase and decrease the value of the optimal q-state
    """
    def split_mid_point(self, state=None, debug=False):

        if state is None:
            state = self.current_state

        # collect the transitions that occured after taking the optimal action
        optimal_action    = state.get_optimal_action()
        opt_qstate        = state.get_qstate(optimal_action)
        incr_measurements = opt_qstate.get_incr_measurements()
        decr_measurements = opt_qstate.get_decr_measurements()

        # only consider splitting if there are enough data
        if len(incr_measurements) + len(decr_measurements) < self.min_measurements:
            return

        # do not split if the standard deviation of the q-value changes is low
        delta_qs = [m[1] for m in incr_measurements] + [m[1] for m in decr_measurements]
        dq_mean  = numpy.mean(delta_qs)
        dq_stdd  = numpy.std(delta_qs)
        if 2 * dq_stdd < dq_mean:
            return

        # find the parameter with the lowest null hypothesis probability
        best_par    = None
        lowest_prob = 1
        for par in self.parameters:
            incr_par = [m[0][par] for m in incr_measurements]
            decr_par = [m[0][par] for m in decr_measurements]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_prob = stats.ttest_ind(incr_par, decr_par)[1]

            if t_prob < lowest_prob:
                lowest_prob   = t_prob
                best_par      = par
                best_incr_par = incr_par
                best_decr_par = decr_par

        if best_par is None or lowest_prob > self.split_error:
            return

        # perform a split using the means for the best parameter
        incr_mean = numpy.mean(best_incr_par)
        decr_mean = numpy.mean(best_decr_par)
        splitting_point = (incr_mean + decr_mean) / 2
        old_qvalues = {}
        for qs in state.get_qstates():
            old_qvalues[qs.get_action()] = qs.get_qvalue()

        state.split(best_par, [splitting_point], old_qvalues)
        self.splits[best_par] += 1

        # store the measurements in the new states that were created
        if self.reuse_meas:
            self.store_measurements(state)

        if debug:
            print("Split with", best_par, "at", splitting_point)


    """
        Stores all the measurements in the given state in the states of the model.
        Persumably the given state has been split and removed.
    """
    def store_measurements(self, state):

        # get all the measurements for each action
        incr_meas = {}
        decr_meas = {}
        for qs in state.get_qstates():
            action = qs.get_action()
            incr_meas[action] = qs.get_incr_measurements()
            decr_meas[action] = qs.get_decr_measurements()

        # store them again in the states of the model
        for action, meas in incr_meas.items():
            for m in meas:
                qstate = self.root.get_state(m[0]).get_qstate(action)
                qstate.store_incr_measurement(m)
                qstate.incr_taken()
        for action, meas in decr_meas.items():
            for m in meas:
                qstate = self.root.get_state(m[0]).get_qstate(action)
                qstate.store_decr_measurement(m)
                qstate.incr_taken()


    """
        Enables reusing the measurements stored in states that got split for new states
    """
    def set_reuse_meas(self, reuse=True):

        self.reuse_meas = reuse


    """
        Returns the number of splits that happened for each parameter
    """
    def get_splits_per_parameter(self):

        return self.splits


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


    """
        Prints all the stored transition data for all the states in the model
    """
    def print_transition_data(self):

        if self.transition_data:
            print("Temporary data in the model:")
            pprint(self.transition_data)

        for s in self.states:
            print("State %d:" % s.get_state_num())
            pprint(s.get_transition_data())


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
        Prints the qstates and the transition and reward lists for each qstate
    """
    def print_state_details(self):

        for s in self.states:
            print(s)
            for qs in s.get_qstates():
                print(qs)
            print("")


