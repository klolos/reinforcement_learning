
from __future__ import division
from scipy import stats
from numpy import mean
import warnings
from DecisionMaking.Constants import *
from DecisionMaking.Exceptions import *
from DecisionMaking.Logging import get_logging_handler
import logging
from timeit import default_timer as timer
from DecisionMaking.MDPModel import QState
from pprint import pprint
import math


"""
    Class to represent a q-state in a Decision Tree MDP model.
"""
class QStateDT(QState):

    """
        Removes the transition and reward information for that state
    """
    def remove_state(self, state_num):

        self.num_taken -= self.transitions[state_num]
        self.transitions[state_num] = 0
        self.rewards[state_num] = 0


    """
        Adds num_states extra states to the reward and transition info
    """
    def extend_states(self, num_states):
    
        self.num_states  += num_states
        self.transitions += [0] * num_states
        self.rewards     += [0] * num_states



"""
    A leaf node in the decision tree, and one of the states of the MDP.
"""
class LeafNode(object):

    def __init__(self, parent, model, actions, qvalues, state_num, num_states):
        
        self.parent          = parent
        self.actions         = actions
        self.initial_qvalues = qvalues
        self.num_states      = num_states
        self.model           = model
        self.state_num       = state_num
        self.transition_data = [[] for x in range(num_states)]
        self.num_visited     = 0
        self.value           = 0

        self.qstates = []
        for name, values in actions.items():
            for value in values:
                action = (name, value)
                qstate = QStateDT(action, num_states, qvalues)
                self.qstates.append(qstate)


    """
        This is a leaf node.
    """
    def is_leaf(self):

        return True


    """
        Replaces this leaf node with a decision node in the decision tree
        and updates all the MDP states accordingly.
    """
    def split(self, param, limits, sweep=False):

        # remove the leaf node from the model
        self.model.remove_state(self.state_num)

        # create the decision node to replace it and add it to the model
        d_node = DecisionNode(self.parent, self.model, param, limits, self.actions, 
                              self.initial_qvalues, self.num_states, self.state_num)
        self.model.add_states(d_node.get_leaves())
        self.parent.replace_node(self, d_node)

        # retrain the model with the stored data
        self.model.retrain()


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
        Stores the given info for a transition to the given state number
    """
    def store_transition(self, data, new_state_num):

        self.transition_data[new_state_num].append(data)
        self.num_visited += 1


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
        Returns the list of measurements that are stored in this state
    """
    def get_transition_data(self):

        return self.transition_data


    """
        Updates the value of the state to be equal to the value of the best qstate
    """
    def update_value(self):

        self.value = max([qs.get_qvalue() for qs in self.qstates])


    """
        Removes the transition and reward information for that state
    """
    def remove_state(self, state_num):

        # if this is the state to be removed store all the transition data to the model
        if state_num == self.state_num:
            for data in self.transition_data:
                self.model.store_transition_data(data)

        # else only remove the transition data towards the removed state
        else:
            data = self.transition_data[state_num]
            self.transition_data[state_num] = []
            self.model.store_transition_data(data)

        # remove the information from the qstates
        num_visited = 0
        for qs in self.qstates:
            num_visited += qs.get_num_transitions(state_num)
            qs.remove_state(state_num)
        self.num_visited -= num_visited


    """
        Adds num_states extra states to the reward and transition info
    """
    def extend_states(self, num_states):
    
        self.num_states      += num_states
        self.transition_data += [[] for x in range(num_states)]
        for qs in self.qstates:
            qs.extend_states(num_states)


    """
        Returns a dict containing the maximum transition probability to each state for each action
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
        Getter for the index of the node in the model state list
    """
    def get_state_num(self):

        return self.state_num


    """
        Setter for the index of the node in the model state list
    """
    def set_state_num(self, state_num):

        self.state_num = state_num


    """
        String representation for a leaf node
    """
    def __str__(self):

        return "State %d, visited = %d" % (self.state_num, self.num_visited)

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
    
    def __init__(self, parent, model, parameter, limits, actions, 
                       initial_qvalues, num_states, replaced_state_num):

        self.parent     = parent
        self.parameter  = parameter
        self.limits     = limits
        self.num_states = num_states + len(limits)
        self.model      = model

        self.children   = [LeafNode(self, model, actions, initial_qvalues,
                                    replaced_state_num, num_states + len(limits))]

        for i in range(len(limits)):
            l = LeafNode(self, model, actions, initial_qvalues, 
                         num_states + i, num_states + len(limits))
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

        old_state_num = old_node.get_state_num()

        for i in range(len(self.children)):
            if self.children[i].is_leaf() and self.children[i].get_state_num() == old_state_num:
                self.children[i] = new_node
                return

        raise InternalError("Tried to replace a node that did not exist", self.model.logger)


    """
        Splits all the children nodes.
        This should only be used when initializing the model with multiple parameters.
    """
    def split(self, param, limits):
    
        for c in self.children:
            c.split(param, limits)


    """
        Removes the transition and reward information for that state from the subtree
    """
    def remove_state(self, state_num):
        
        for c in self.children:
            c.remove_state(state_num)


    """
        Adds num_states extra states to the reward and transition info for all states in the subtree
    """
    def extend_states(self, num_states):
    
        self.num_states += num_states
        for c in self.children:
            c.extend_states(num_states)


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
            raise ParameterError("Missing measurement: " + self.parameter, self.model.logger)

        m = measurements[self.parameter]
        for i, l in enumerate(self.limits):
            if m < l:
                return self.children[i].get_state(measurements)

        return self.children[-1].get_state(measurements)



"""
    Class that represents a Markov Decision Process model with a decision tree state structure.
"""
class MDPDTModel:

    """
        Creates a model from a given configuration dict
    """
    def __init__(self, conf):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(get_logging_handler(LOG_FILENAME))

        required_fields = [INITIAL_PARAMETERS, PARAMETERS, ACTIONS, DISCOUNT, 
                           INITIAL_QVALUES, SPLIT_ERROR, MIN_MEASUREMENTS]

        for f in required_fields:
            if not f in conf:
                raise ConfigurationError("%s not provided in the configuration" % f, self.logger)

        self.discount         = conf[DISCOUNT]
        self.parameters       = list(conf[PARAMETERS])
        self.min_measurements = max(conf[MIN_MEASUREMENTS], 1)
        self.split_error      = conf[SPLIT_ERROR]
        self.actions          = conf[ACTIONS]
        self.initial_qvalues  = conf[INITIAL_QVALUES]
        self.init_pars        = self._get_params(conf[INITIAL_PARAMETERS])
        self.current_state    = None
        self.current_meas     = None
        self.new_states       = []
        self.transition_data  = []
        self.update_error     = 0.1   # default value for value iteration and PS error
        self.max_updates      = 100   # default value for prioritized sweeping updates
        self.test             = STUDENT_TTEST

        # initiate the decision tree
        self.root       = LeafNode(self, self, self.actions, self.initial_qvalues, 0, 1)
        self.states     = [self.root]
        self.priorities = [0]
        for name, values in self.init_pars.items():
            self.root.split(name, values)

        # set the default update and splitting algorithms and initialize the split counters
        self.update_algorithm = SINGLE_UPDATE
        self.split_criterion  = MID_POINT
        self.consider_trans   = True
        self.allow_splitting  = True
        self.splits           = {}
        for p in self.parameters:
            self.splits[p] = 0

        self.logger.debug("Initialized MDPDT model with %d states" % len(self.states))


    """
        Returns the decision tree to its initial state, preserving all measurements collected
    """
    def reset_decision_tree(self, vi_error=None):

        # collect the transition information from all the states
        self.transition_data = [t for s in self.states for ts in s.get_transition_data() for t in ts]

        # recreate the decision tree
        self.root   = LeafNode(self, self, self.actions, self.initial_qvalues, 0, 1)
        self.states = [self.root]
        for name, param in self.init_pars.items():
            self.root.split(name, param)

        # store the transition data in the new states and recalculate the state values
        self.retrain()
        self.value_iteration(error=vi_error)

        # reset the split counters
        for p in self.parameters:
            self.splits[p] = 0

        self.logger.debug("Decision Tree has been reset")


    """
        Extract the defined limits or values for the initial parameters
        so that they can be used by a decision node.
    """
    def _get_params(self, parameters):

        new_pars = {}
        for name, v in parameters.items():

            # for discrete values we define the midpoint as the margin
            if VALUES in v:
                if not isinstance(v[VALUES], list):
                    raise ConfigurationError("Provided values for %s must be in a list" % name, self.logger)
                if len(v[VALUES]) <= 1:
                    raise ConfigurationError("At least two values must be provided for " + name, self.logger)

                limits = []
                for i in range(len(v[VALUES]) - 1):
                    limits.append((v[VALUES][i] + v[VALUES][i+1]) / 2)
                new_pars[name] = limits

            # for continuous values we just ignore the outer margins
            elif LIMITS in v:
                if not isinstance(v[LIMITS], list):
                    raise ConfigurationError("Provided limits for %s must be in a list" % name, self.logger)
                if len(v[LIMITS]) <= 2:
                    raise ConfigurationError("At least three limits must be provided for " + name, self.logger)

                new_pars[name] = v[LIMITS][1:-1]

            else:
                raise ConfigurationError("Values or limits must be provided for parameter " + name, self.logger)

        return new_pars


    """
        Replaces the root node with the given decision node.
        This should happen when the root node splits.
    """
    def replace_node(self, old_node, new_node):

        if not self.root.is_leaf():
            raise InternalError("Tried to replace the root node but it was not a leaf node", self.logger)

        if not old_node.get_state_num() is self.root.get_state_num():
            raise InternalError("Tried to replace the root node with a different initial node", self.logger)
    
        self.root = new_node


    """
        Sets the current state based on the given measurements
    """
    def set_state(self, measurements):

        self.current_meas  = measurements
        self.current_state = self.root.get_state(measurements)


    """
        Removes the state with the given state_num from the model
    """
    def remove_state(self, state_num):

        self.root.remove_state(state_num)
        self.states[state_num]     = None
        self.priorities[state_num] = 0


    """
        Stores the given transition data to be used later on for retraining
    """
    def store_transition_data(self, data):

        self.transition_data += data


    """
        Adds new states to the model. The first will go in the empty spot and the rest at the end.
    """
    def add_states(self, states):

        # the first state will not be appended at the end
        self.root.extend_states(len(states) - 1)
        self.priorities += [0] * (len(states) - 1)
        self.new_states  = states

        # place the first state in the empty spot and the rest at the end
        replaced_state_num = states[0].get_state_num()
        if not self.states[replaced_state_num] is None:
            raise InternalError("Replaced state was not None")

        self.states[replaced_state_num] = states[0]
        self.states += states[1:]


    """
        Suggest the optimal action to take from the current state
    """
    def suggest_action(self):

        if self.current_state is None:
            raise StateNotSetError(self.logger)

        return self.current_state.get_optimal_action()


    """
        Returns all the legal actions from the current_state
    """
    def get_legal_actions(self):
        
        if self.current_state is None:
            raise StateNotSetError(self.logger)
        
        return self.current_state.get_legal_actions()


    """
        Stops the model from performing any updates to q-values
    """
    def set_no_update(self):

        self.update_algorithm = NO_UPDATE
        self.logger.debug("Update algorithm set to NO_UPDATE")


    """
        Update only the value of the starting state after each transition
    """
    def set_single_update(self):

        self.update_algorithm = SINGLE_UPDATE
        self.logger.debug("Update algorithm set to SINGLE_UPDATE")


    """
        Perform a full value iteration after each transition
    """
    def set_value_iteration(self, update_error):

        self.update_algorithm = VALUE_ITERATION
        self.update_error     = update_error
        self.logger.debug("Update algorithm set to VALUE_ITERATION with error " + str(update_error))


    """
        Perform prioritized sweeping after each transition
    """
    def set_prioritized_sweeping(self, update_error, max_updates):

        self.update_algorithm = PRIORITIZED_SWEEPING
        self.update_error     = update_error
        self.max_updates      = max_updates
        self.logger.debug("Update algorithm set to PRIORITIZED_SWEEPING with error = " + \
                          str(update_error) + " and max updates = " + str(max_updates))


    """
        Set the statistical test to use for splitting
    """
    def set_stat_test(self, stat_test):

        self.test = stat_test


    """
        Updates the model after taking the given action and ending up in the
        state corresponding to the given measurements.
    """
    def update(self, action, measurements, reward, debug=False):

        if self.current_meas is None:
            raise StateNotSetError(self.logger)

        # TODO move this where the splitting is decided
        self.current_state = self.root.get_state(self.current_meas)
        
        # determine the new state
        new_state = self.root.get_state(measurements)
        new_num   = new_state.get_state_num()

        # store the transition information
        trans_data = (self.current_meas, measurements, action, reward)
        self.current_state.store_transition(trans_data, new_num)

        # update the qstate
        qstate = self.current_state.get_qstate(action)
        qstate.update(new_state, reward)

        # update the model values according to the chosen algorithm
        if self.update_algorithm == SINGLE_UPDATE:
            self._q_update(qstate)
            self.current_state.update_value()
        elif self.update_algorithm == VALUE_ITERATION:
            self.value_iteration()
        elif self.update_algorithm == PRIORITIZED_SWEEPING:
            self.prioritized_sweeping()

        # consider splitting the initial_state
        if self.allow_splitting:
            self.split(debug=debug)

        # update the current state and store the last measurements
        self.current_state = new_state
        self.current_meas  = measurements


    """
        Attempts to split the given node with the chosen splitting algorithm
    """
    def split(self, state=None, debug=False):

        if self.split_criterion == MID_POINT:
            return self.split_mid_point(state=state, debug=debug)
        elif self.split_criterion == ANY_POINT:
            return self.split_any_point(state=state, debug=debug)
        elif self.split_criterion == MEDIAN_POINT:
            return self.split_median_point(state=state, debug=debug)
        elif self.split_criterion == MAX_POINT:
            return self.split_max_point(state=state, debug=debug)
        elif self.split_criterion == QVALUE_DIFF:
            return self.split_qvalue_diff(state=state, debug=debug)
        elif self.split_criterion == INFO_GAIN:
            return self.split_info_gain(state=state, debug=debug)
        else:
            raise InternalError("Unknown splitting algorithm: " + self.split_criterion, self.logger)


    """
        Retrains the model with the transition data temporarily stored in the model
    """
    def retrain(self):

        for m1, m2, a, r in self.transition_data:
            # determine the states involved in the transition
            old_state = self.root.get_state(m1)
            new_state = self.root.get_state(m2)

            # store the transition data in the initial state of the transition
            new_num = new_state.get_state_num()
            old_state.store_transition((m1, m2, a, r), new_num)

            # update the qstate
            qstate = old_state.get_qstate(a)
            qstate.update(new_state, r)

        # clear the transition data from the model
        self.transition_data = []


    """
        Repeatedly attempts to split all the nodes until no splits are possible
    """
    def chain_split(self):

        num_splits = 0
        did_split  = True
        while (did_split):
            did_split = False
            states = list(self.states)
            for s in states:
                if self.split(state=s):
                    did_split = True
                    num_splits += 1

            if did_split:
                self.value_iteration()

        self.logger.debug("Chain splitting complete after %d splits" % num_splits)


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

        start = timer()
        repeat = True
        while (repeat):
            repeat = False
            for s in self.states:
                old_value = s.get_value()
                self._v_update(s)
                new_value = s.get_value()
                if abs(old_value - new_value) > error:
                    repeat = True
        end = timer()

        self.logger.debug("Value iteration complete after %f seconds" % (end - start))


    """
        Runs prioritized sweeping starting from the given state.
    """
    def prioritized_sweeping(self, initial_state=None, error=None, max_updates=None, debug=False):

        if self.current_state is None and initial_state is None:
            raise StateNotSetError(self.logger)
        
        if initial_state is None:
            initial_state = self.current_state
        if error is None:
            error = self.update_error
        if max_updates is None:
            max_updates = self.max_updates

        # transition probabilities have changed for the initial state
        reverse_transitions = [{} for s in self.states]
        for s in self.states:
            for s_num, t in s.get_max_transitions().items():
                reverse_transitions[s_num][s.get_state_num()] = t

        s = initial_state
        for i in range(max_updates):

            # update the state value
            old_value = s.get_value()
            self._v_update(s)
            new_value = s.get_value()
            delta = abs(new_value - old_value)

            # update the priorities of the predecessors
            rev_transitions = reverse_transitions[s.get_state_num()]
            for s_num, t in rev_transitions.items():
                self.priorities[s_num] = max(t * delta, self.priorities[s_num])

            # zero the updated state's priority
            self.priorities[s.get_state_num()] = 0
            if debug:
                print("sweeping for %d, delta = %f" % (s.get_state_num(), delta))
                print("reverse transitions: " + str(reverse_transitions))
                print("priorities: " + str(self.priorities))

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
        Allow or prevent the decision tree from splitting its nodes
    """
    def set_allow_splitting(self, allow=True):

        self.allow_splitting = allow
        self.logger.debug("Allow splitting set to " + str(allow))


    """
       Set the splitting criterion
    """
    def set_splitting(self, split_criterion, consider_transitions=True):

        if not split_criterion in SPLIT_CRITERIA:
            raise ParameterError("Unknown splitting algorithm: " + split_criterion, self.logger)

        self.split_criterion = split_criterion
        self.consider_trans  = consider_transitions
        self.logger.debug("Splitting criterion set to %s, consider transitions set to %s" % \
                          (split_criterion, consider_transitions))


    def stat_test(self, x1, x2):

        if self.test == STUDENT_TTEST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, stat = stats.ttest_ind(x1, x2)
        elif self.test == WELCH_TTEST:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, stat = stats.ttest_ind(x1, x2, equal_var=False)
        elif self.test == MANN_WHITNEY_UTEST:
            try:
                u, stat_one_sided = stats.mannwhitneyu(x1, x2)
                stat = 2 * stat_one_sided
            except ValueError:
                stat = 1
        elif self.test == KOLMOGOROV_SMIRNOV:
            ks, stat = stats.ks_2samp(x1, x2)

        return stat


    """
        Attempts to split the current state in the midpoint between transitions that
        would increase and decrease the value of the optimal q-state.
        Returns True if the split was made.
    """
    def split_mid_point(self, state=None, debug=False):

        start = timer()

        if state is None:
            state = self.current_state

        # collect the transitions that occured after taking the optimal action
        optimal_action = state.get_optimal_action()
        t_data         = state.get_transition_data()
        transitions    = [t for ts in t_data for t in ts if t[2] == optimal_action]

        # never split if there are no transitions
        if len(transitions) == 0:
            return False

        incr_measurements = []
        decr_measurements = []
        if self.consider_trans:
            # partition the transitions to those that would increase or decrease the q-value
            for m1, m2, a, r in transitions:
                new_state_value = self.root.get_state(m2).get_value()
                q_value = r + self.discount * new_state_value
                if q_value >= state.get_value():
                    incr_measurements.append(m1)
                else:
                    decr_measurements.append(m1)
        else:
            # partition the transitions to those that gave higher or lower rewards than average
            average_rewards = mean([t[3] for t in transitions])
            for m1, m2, a, r in transitions:
                if r >= average_rewards:
                    incr_measurements.append(m1)
                else:
                    decr_measurements.append(m1)

        # only consider splitting if there are enough data for either side
        if min(len(incr_measurements), len(decr_measurements)) < self.min_measurements:
            return False
            
        # find the parameter with the lowest null hypothesis probability
        best_par     = None
        lowest_error = 1
        for par in self.parameters:
            incr_par = [m[par] for m in incr_measurements]
            decr_par = [m[par] for m in decr_measurements]
            t1_error = self.stat_test(incr_par, decr_par)

            if t1_error < lowest_error:
                lowest_error  = t1_error
                best_par      = par
                best_incr_par = incr_par
                best_decr_par = decr_par

        if best_par is None or lowest_error > self.split_error:
            return False

        # perform a split using the means for the best parameter
        incr_mean = mean(best_incr_par)
        decr_mean = mean(best_decr_par)
        splitting_point = (incr_mean + decr_mean) / 2
        state.split(best_par, [splitting_point])
        self.splits[best_par] += 1

        # recalculate the values of the new states generated by the split
        for s in self.new_states:
            self._v_update(s)
        self.new_states = []

        end = timer()
        self.logger.debug("Split with %s at %s with prob %s after %s seconds" % \
                          (best_par, splitting_point, lowest_error, end - start))

        return True


    """
        Attempts to split the current state on any single point between two recorded values.
        Returns True if the split was made.
    """
    def split_any_point(self, state=None, debug=False):

        start = timer()

        if state is None:
            state = self.current_state

        # collect the transitions that occured after taking the optimal action
        optimal_action = state.get_optimal_action()
        t_data         = state.get_transition_data()
        transitions    = [t for ts in t_data for t in ts if t[2] == optimal_action]

        # calculate the q-value resulting from each of the transitions
        q_values = []
        if self.consider_trans:
            for m1, m2, a, r in transitions:
                new_state_value = self.root.get_state(m2).get_value()
                q_value = r + self.discount * new_state_value
                q_values.append((m1, q_value))
        else: # do not consider transitions and only keep the immediate reward
            for m1, m2, a, r in transitions:
                q_values.append((m1, r))

        # find the splitting point with the lowest null hypothesis probability
        best_par     = None
        bast_point   = None
        lowest_error = 1
        for par in self.parameters:
            par_values = sorted([(q[0][par], q[1]) for q in q_values])
            # only consider points that leave at least min_measurements points on either side
            for i in range(self.min_measurements, len(transitions) - self.min_measurements + 1):
                # only split between distinct measurements
                if par_values[i][0] == par_values[i-1][0]:
                    continue

                low_values  = [p[1] for p in par_values[:i]]
                high_values = [p[1] for p in par_values[i:]]
                t1_error = self.stat_test(low_values, high_values)
                if debug:
                    print(par, "point =", (par_values[i][0] + par_values[i-1][0]) / 2, "prob =", t1_error)

                if t1_error < lowest_error:
                    lowest_error = t1_error
                    best_par     = par
                    best_point   = (par_values[i][0] + par_values[i-1][0]) / 2

        if best_par is None or lowest_error > self.split_error:
            return False

        # perform a split at the selected point
        state.split(best_par, [best_point])
        self.splits[best_par] += 1

        # recalculate the values of the new states generated by the split
        for s in self.new_states:
            self._v_update(s)
        self.new_states = []

        end = timer()
        self.logger.debug("Split with %s at %s with prob %s after %s seconds" % \
                          (best_par, best_point, lowest_error, end - start))

        return True

    """
        Attempts to split the current state on the median point between two recorded values
        of a parameter. Returns True if the split was made.
    """
    def split_median_point(self, state=None, debug=False):

        start = timer()

        if state is None:
            state = self.current_state

        # collect the transitions that occured after taking the optimal action
        optimal_action = state.get_optimal_action()
        t_data         = state.get_transition_data()
        transitions    = [t for ts in t_data for t in ts if t[2] == optimal_action]

        # calculate the q-value resulting from each of the transitions
        q_values = []
        if self.consider_trans:
            for m1, m2, a, r in transitions:
                new_state_value = self.root.get_state(m2).get_value()
                q_value = r + self.discount * new_state_value
                q_values.append((m1, q_value))
        else: # do not consider transitions and only keep the immediate reward
            for m1, m2, a, r in transitions:
                q_values.append((m1, r))

        # find the splitting point with the lowest null hypothesis probability
        best_par     = None
        bast_point   = None
        lowest_error = 1
        median       = int(len(transitions) / 2)
        for par in self.parameters:
            par_values = sorted([(q[0][par], q[1]) for q in q_values])
            split_index = None
            # only consider points that leave at least min_measurements points on either side
            for i in range(median - self.min_measurements):

                # only split between distinct measurements
                if par_values[median - i][0] != par_values[median - i - 1][0]:
                    split_index = median - i
                    break

                if par_values[median + i][0] != par_values[median + i + 1][0]:
                    split_index = median + i + 1
                    break

            if split_index is None:
                continue

            low_values  = [p[1] for p in par_values[:split_index]]
            high_values = [p[1] for p in par_values[split_index:]]
            t1_error = self.stat_test(low_values, high_values)
            if debug:
                print(par, "point =", (par_values[split_index][0] + par_values[split_index-1][0]) / 2, \
                      "prob =", t1_error)

            if t1_error < lowest_error:
                lowest_error = t1_error
                best_par     = par
                best_point   = (par_values[split_index][0] + par_values[split_index-1][0]) / 2

            if best_par is None or lowest_error > self.split_error:
                return False

            # perform a split at the selected point
            state.split(best_par, [best_point])
            self.splits[best_par] += 1

            # recalculate the values of the new states generated by the split
            for s in self.new_states:
                self._v_update(s)
            self.new_states = []

            end = timer()
            self.logger.debug("Split with %s at %s with prob %s after %s seconds" % \
                              (best_par, best_point, lowest_error, end - start))

            return True


    """
        Attempts to split the current state on any single point between two recorded values,
        trying to maximize the difference in q-values between the split nodes.
        Returns True if the split was made.
    """
    def split_max_point(self, state=None, debug=False):

        start = timer()

        if state is None:
            state = self.current_state

        # collect the transitions that occured after taking the optimal action
        optimal_action = state.get_optimal_action()
        t_data         = state.get_transition_data()
        transitions    = [t for ts in t_data for t in ts if t[2] == optimal_action]

        # calculate the q-value resulting from each of the transitions
        q_values = []
        if self.consider_trans:
            for m1, m2, a, r in transitions:
                new_state_value = self.root.get_state(m2).get_value()
                q_value = r + self.discount * new_state_value
                q_values.append((m1, q_value))
        else: # do not consider transitions and only keep the immediate reward
            for m1, m2, a, r in transitions:
                q_values.append((m1, r))

        # find the splitting point that generates the maximum q-value difference
        best_par   = None
        bast_point = None
        max_diff   = 0
        for par in self.parameters:
            par_values = sorted([(q[0][par], q[1]) for q in q_values])
            # only consider points that leave at least min_measurements points on either side
            for i in range(self.min_measurements, len(transitions) - self.min_measurements + 1):
                # only split between distinct measurements
                if par_values[i][0] == par_values[i-1][0]:
                    continue

                low_values  = [p[1] for p in par_values[:i]]
                high_values = [p[1] for p in par_values[i:]]
                t1_error = self.stat_test(low_values, high_values)
                if debug:
                    print("low_values: " + str(low_values))
                    print("high_values: " + str(high_values))
                    print("state value: " + str(state.get_value()))

                # only consider splitting when there at least the minimum confidence
                if t1_error > self.split_error:
                    continue

                # find the point that maximizes the q-value difference
                low_avg  = mean(low_values)
                high_avg = mean(high_values)
                if abs(high_avg - low_avg) > max_diff:
                    max_diff   = abs(high_avg - low_avg)
                    best_par   = par
                    best_point = (par_values[i][0] + par_values[i-1][0]) / 2

        if best_par is None:
            return False

        # perform a split at the selected point
        state.split(best_par, [best_point])
        self.splits[best_par] += 1

        # recalculate the values of the new states generated by the split
        for s in self.new_states:
            self._v_update(s)
        self.new_states = []

        end = timer()
        self.logger.debug("Split with %s at %s with q-value difference %s after %s seconds" % \
                          (best_par, best_point, max_diff, end - start))

        return True


    """
        Attempts to perform a split on the point that maximizes the information gain,
        according to the ID3 / C4.5 algorithms.
    """
    def split_info_gain(self, state=None, debug=False):

        start = timer()

        if state is None:
            state = self.current_state

        # collect the transitions that occured after taking the optimal action
        optimal_action = state.get_optimal_action()
        t_data         = state.get_transition_data()
        transitions    = [t for ts in t_data for t in ts if t[2] == optimal_action]

        if debug:
            print("transitions: " + str(transitions))

        # calculate the q-value resulting from each of the transitions
        q_values = []
        if self.consider_trans:
            for m1, m2, a, r in transitions:
                new_state_value = self.root.get_state(m2).get_value()
                q_value = r + self.discount * new_state_value
                q_values.append((m1, q_value))
        else: # do not consider transitions and only keep the immediate reward
            for m1, m2, a, r in transitions:
                q_values.append((m1, r))

        if debug:
            print("q_values: " + str(q_values))

        # calculate the information required for the current state
        state_value = state.get_value()
        high_q = sum(1 for q in q_values if q[1] > state_value)
        low_q  = len(q_values) - high_q
        state_info = self._info(high_q, low_q)

        if debug:
            print("state value: " + str(state_value))
            print("high_q: " + str(high_q))
            print("low_q: " + str(low_q))

        # find the splitting point that maximizes the information gain
        best_par = None
        bast_point = None
        min_info = float("inf")
        for par in self.parameters:
            par_values = sorted([(q[0][par], q[1]) for q in q_values])
            for i in range(self.min_measurements, len(transitions) - self.min_measurements + 1):
                # only split between distinct measurements
                if par_values[i][0] == par_values[i-1][0]:
                    continue

                low_values  = [p[1] for p in par_values[:i]]
                high_values = [p[1] for p in par_values[i:]]

                low_incr  = sum(1 for q in low_values  if q > state_value)
                high_incr = sum(1 for q in high_values if q > state_value)
                low_decr  = len(low_values)  - low_incr
                high_decr = len(high_values) - high_incr
            
                info = self._expected_info(low_incr, low_decr, high_incr, high_decr)

                if debug:
                    print("par: " + str(par))
                    print("par_values: " + str(par_values))
                    print("low_values: " + str(low_values))
                    print("high_values: " + str(high_values))
                    print("state_value: " + str(state.get_value()))
                    print("high_q: " + str(high_q))
                    print("low_q: " + str(low_q))
                    print("state_info: " + str(state_info))
                    print("low_incr: " + str(low_incr))
                    print("low_decr: " + str(low_decr))
                    print("high_incr: " + str(high_incr))
                    print("high_decr: " + str(high_decr))
                    print("info: " + str(info))

                if info < min_info:
                    min_info = info
                    best_par = par
                    best_point = 0.5 * (par_values[i][0] + par_values[i-1][0])

        if best_par is None or min_info > state_info:
            return False

        if debug:
            print("best_par: " + str(best_par))
            print("best_point: " + str(best_point))

        # perform a split at the selected point
        state.split(best_par, [best_point])
        self.splits[best_par] += 1

        # recalculate the values of the new states generated by the split
        for s in self.new_states:
            self._v_update(s)
        self.new_states = []

        end = timer()
        self.logger.debug("Split with %s at %s with min_info %s after %s seconds" % \
                          (best_par, best_point, min_info, end - start))

        return True


    """
        Returns the expected information required as per Quinlan's ID3
    """
    def _expected_info(self, p1, n1, p2, n2):

        s = p1 + n1 + p2 + n2
        s1 = p1 + n1
        s2 = p2 + n2

        return (s1 / s) * self._info(p1, n1) + (s2 / s) * self._info(p2, n2)


    """
        Returns the expected classification information as per Quinlan's ID3
    """
    def _info(self, p, n):

        if n <= 0 or p <= 0:
            return 0
        else:
            return -(p/(p+n))*math.log(p/(p+n), 2) - (n/(p+n))*math.log(n/(p+n), 2)


    """
        Attempts to split the current state on any single point between two recorded values,
        trying to minimize the difference in q-values between the optimal and any non-optimal action.
        DEPRECATED
    """
    def split_qvalue_diff(self, state=None, debug=False):

        start = timer()

        if state is None:
            state = self.current_state

        # calculate the q-value resulting from each of the transitions
        # transitions: [(initial measurements, final measurements, action, rewards)]
        # q_values:    [(initial measurements, action, q_value)]
        transitions = [t for ts in state.get_transition_data() for t in ts]
        q_values    = []
        #if len(transitions) < 4 * self.min_measurements:
        #    if debug:
        #        print("Only", len(transitions), "transitions, not splitting")
        #    return False
        for m1, m2, a, r in transitions:
            new_state_value = self.root.get_state(m2).get_value()
            q_value = r + self.discount * new_state_value
            q_values.append((m1, a, q_value))

        # find the splitting point that generates the maximum q-value difference
        best_par       = None
        bast_point     = None
        min_diff       = float("inf")
        optimal_action = state.get_optimal_action()
        for par in self.parameters:
            # par_values: [(parameter value, action, q_value)]
            if debug:
                print("parameter =", par)
            par_values = sorted([(q[0][par], q[1], q[2]) for q in q_values])

            # only consider points that leave at least min_measurements points on either side
            for i in range(self.min_measurements, len(transitions) - self.min_measurements + 1):
                # only split between distinct measurements
                if par_values[i][0] == par_values[i-1][0]:
                    continue

                # low/high_values: [(action, q_value)]
                # low/high_qvalues: {action: q_value}
                low_values      = [(p[1], p[2]) for p in par_values[:i]]
                high_values     = [(p[1], p[2]) for p in par_values[i:]]

                # low/high_opt_values: [(q_value)]
                low_opt_values  = [p[1] for p in low_values  if p[0] == optimal_action]
                high_opt_values = [p[1] for p in high_values if p[0] == optimal_action]

                if len(low_opt_values) < self.min_measurements or \
                   len(high_opt_values) < self.min_measurements:
                    continue

                low_qvalues  = {}
                high_qvalues = {}
                low_qvalues[optimal_action]  = mean(low_opt_values)
                high_qvalues[optimal_action] = mean(high_opt_values)

                actions = state.get_legal_actions()
                for action in actions:
                    if action == optimal_action:
                        continue
                    # calculate the qvalues for the actions in each part of the split
                    # low/high_action_values: [(q_value)]
                    low_action_values  = [p[1] for p in low_values  if p[0] == action]
                    high_action_values = [p[1] for p in high_values if p[0] == action]
                    if len(low_action_values) < self.min_measurements:
                        low_qvalues[action] = None
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            t_prob = stats.ttest_ind(low_opt_values, low_action_values, 
                                                     equal_var=self.equal_var)[1]
                            if t_prob > self.split_error:
                                low_qvalues[action] = None
                            else:
                                low_qvalues[action] = mean(low_action_values)

                    if len(high_action_values) < self.min_measurements:
                        high_qvalues[action] = None
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            t_prob = stats.ttest_ind(high_opt_values, high_action_values,
                                                     equal_var=self.equal_var)[1]
                            if t_prob > self.split_error:
                                high_qvalues[action] = None
                            else:
                                high_qvalues[action] = mean(high_action_values)

                if debug:
                    print("low_values")
                    pprint(low_values)
                    print("high_values")
                    pprint(high_values)
                    print("low_qvalues  =", low_qvalues)
                    print("high_qvalues =", high_qvalues)

                best_non_optimal_low  = max([low_qvalues[a]  for a in actions if a != optimal_action])
                best_non_optimal_high = max([high_qvalues[a] for a in actions if a != optimal_action])
                # there must be at least one transition with an optimal an a non-optimal action to split
                #if low_qvalues[optimal_action] is None or high_qvalues[optimal_action] is None \
                #   or max(best_non_optimal_low, best_non_optimal_high) is None:
                #    continue
                if best_non_optimal_low is None and best_non_optimal_high is None:
                    continue

                # calculate the min difference between the optimal and any non-optimal q-value
                if best_non_optimal_low is None:
                    diff = high_qvalues[optimal_action] - best_non_optimal_high
                elif best_non_optimal_high is None:
                    diff = low_qvalues[optimal_action] - best_non_optimal_low
                else:
                    diff = min(high_qvalues[optimal_action] - best_non_optimal_high,
                               low_qvalues[optimal_action] - best_non_optimal_low)
                
                if debug:
                    print("optimal_action =", optimal_action)
                    print("best_non_optimal_low =", best_non_optimal_low)
                    print("best_non_optimal_high =", best_non_optimal_high)
                    print("diff =", diff)

                if diff < min_diff:
                    min_diff   = diff
                    best_par   = par
                    best_point = (par_values[i][0] + par_values[i-1][0]) / 2

        if best_par is None:
            return False

        # perform a split at the selected point
        state.split(best_par, [best_point])
        self.splits[best_par] += 1

        # recalculate the values of the new states generated by the split
        for s in self.new_states:
            self._v_update(s)
        self.new_states = []

        end = timer()
        self.logger.debug("Split with %s at %s with min difference %s after %s seconds" % \
                          (best_par, best_point, min_diff, end - start))

        return True


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
            print("Node %d:" % s.get_state_num())
            for qs in s.get_qstates():
                print(qs)
                print("Transitions:", qs.get_transitions())
                print("Rewards:    ", qs.get_rewards())
            print("")


