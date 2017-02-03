
from __future__ import division
from DecisionMaking.MDPModel import MDPModel, QState
from DecisionMaking.Constants import *
from DecisionMaking.Exceptions import *


"""
    Class to represent a q-state in an MDP with Content Detection
"""
class QStateCD(QState):

    def __init__(self, action, num_states, qvalue = 0.0):
        
        super(QStateCD, self).__init__(action, num_states, qvalue)
        self.sum_square_transitions = 0
        self.total_reward = 0
        
    def update(self, new_state, reward, debug=False):

        # calculate the transition error
        j   = new_state.get_state_num()
        n   = self.num_taken
        tj  = self.transitions[j]
        Sti = self.sum_square_transitions
        if n > 0:
            t_error = (Sti + n * (n - 2 * tj)) / ((n * (n + 1)) ** 2)
        else:
            t_error = (self.num_states - 1) / self.num_states

        # update the sum of squared transitions
        self.sum_square_transitions += 2 * tj + 1

        # remember the previous average reward and perform the update
        if self.num_taken > 0:
            old_reward = self.total_reward / self.num_taken
        else:
            old_reward = 0

        super(QStateCD, self).update(new_state, reward)
        self.total_reward += reward

        # normalize the transition error to [0,1]
        zt = 0.5 * (self.num_taken ** 2)
        t_error *= zt

        # calculate the reward error
        new_reward = self.total_reward / self.num_taken
        r_error = (new_reward - old_reward) ** 2

        if debug:
            print("Q-state: " + str(self))
            print("old reward: " + str(old_reward))
            print("new reward: " + str(new_reward))
            print("r-error: " + str(r_error))

        return (t_error, r_error, self.num_taken)


"""
    Class that represents a full Markov Decision Process model.
"""
class MDPModelCD(MDPModel):

    """
        Adds the given actions to all the states
    """
    def _add_qstates(self, actions, initial_qvalue):

        # add the qstates
        num_states = len(self.states)
        for action_type, values in actions.items():
            for action_value in values:
                action = (action_type, action_value)
                for s in self.states:
                    if self._is_permissible(s, action):
                        s.add_qstate(QStateCD(action, num_states, initial_qvalue))

        # update the values of the states
        for s in self.states:
            s.update_value()


    """
        Updates the model after taking the given action and ending up in the
        state corresponding to the given measurements.
    """
    def update(self, action, measurements, reward, debug=False):

        if self.current_state is None:
            raise StateNotSetError()

        qstate = self.current_state.get_qstate(action)
        if qstate is None:
            # TODO log
            return None

        self.current_state.visit()
        new_state = self._get_state(measurements)
        if debug:
            print("old state: %s" % self.current_state)
            print("new state: %s" % new_state)

        errors = qstate.update(new_state, reward, debug)

        if self.update_algorithm == SINGLE_UPDATE:
            self._q_update(qstate)
            self.current_state.update_value()
        elif self.update_algorithm == VALUE_ITERATION:
            self.value_iteration()
        elif self.update_algorithm == PRIORITIZED_SWEEPING:
            self.prioritized_sweeping()

        self.current_state = new_state

        return errors


    """
        Returns the number of times the suggested action has been taken for this model
    """
    def suggested_action_num_taken(self):

        if self.current_state is None:
            raise StateNotSetError()

        return self.current_state.best_action_num_taken()


"""
    Class that implements a multiple MDP model using Context Detection.
"""
class MDPCDModel(object):

    """
        Creates a model from a given configuration dict
    """
    def __init__(self, conf):
        
        required_fields = [PARAMETERS, OPTIONAL_PARAMETERS, ACTIONS, DISCOUNT, INITIAL_QVALUES, 
                           MAX_OPTIONAL_PARAMETERS, TRAINING_WINDOW, QUALITY_RATE]

        for f in required_fields:
            if not f in conf:
                raise ConfigurationError("%s not provided in the configuration" % f)
        
        # create the configurations for all the MDP modelds
        configurations = self._create_configurations(conf)

        self.models = []
        for c in configurations:
            m = MDPModelCD(c)
            self.models.append(m)

        self.training_window   = conf[TRAINING_WINDOW]
        self.reward_importance = conf[REWARD_IMPORTANCE]
        self.quality_rate      = conf[QUALITY_RATE]
        self.qualities         = [0] * len(self.models)
        self.choices           = [0] * len(self.models)
        self.best_model        = 0
        self.max_reward        = float("-inf")
        self.min_reward        = float("inf")


    """
        Returns a list of all the MDPModel configurations containing
        all the basic parameters plus up to a specified number of optional parameters
    """
    def _create_configurations(self, conf):

        # create all the combinations of optional parameters
        combinations = [[[]]]
        for i in range(conf[MAX_OPTIONAL_PARAMETERS]):
            combinations.append([])
            for combo in combinations[i]:
                for p in conf[OPTIONAL_PARAMETERS]:
                    if not combo or p > combo[-1]:
                        combinations[i+1].append(combo + [str(p)])

        # flatten the list of lists of combinations
        combinations = [x for y in combinations for x in y]

        model_configurations = []
        for combo in combinations:
            new_conf = {}
            new_conf[MODEL]           = MDP
            new_conf[DISCOUNT]        = conf[DISCOUNT]
            new_conf[INITIAL_QVALUES] = conf[INITIAL_QVALUES]

            # add the default parameters of the model
            new_conf[PARAMETERS] = {}
            for name, val in conf[PARAMETERS].items():
                if VALUES in val:
                    new_conf[PARAMETERS][str(name)] = {VALUES : list(val[VALUES])}
                elif LIMITS in val:
                    new_conf[PARAMETERS][str(name)] = {LIMITS : list(val[LIMITS])}
                else:
                    raise ConfigurationError("%s or %s must be provided for parameter %s" \
                                              % (VALUES, LIMITS, name))

            # add the optional parameters of the model
            for name in combo:
                c = conf[OPTIONAL_PARAMETERS][name]
                if VALUES in c:
                    new_conf[PARAMETERS][name] = {VALUES : list(c[VALUES])}
                elif LIMITS in c:
                    new_conf[PARAMETERS][name] = {LIMITS : list(c[LIMITS])}
                else:
                    raise ConfigurationError("%s or %s must be provided for parameter %s" \
                                              % (VALUES, LIMITS, name))
            
            # add the actions of the model
            new_conf[ACTIONS] = {}
            for name, val in conf[ACTIONS].items():
                new_conf[ACTIONS][name] = list(val)

            model_configurations.append(new_conf)
        
        return model_configurations


    """
        Sets the state for the entire model based on the given measurements
    """
    def set_state(self, measurements):
        
        for m in self.models:
            m.set_state(measurements)


    """
        Suggest the next action besed on the most accurate model
    """
    def suggest_action(self):

        self.choices[self.best_model] += 1
        return self.models[self.best_model].suggest_action()


    """
        Suggest the next action besed on the most accurate model
        Prefers actions that have already been taken
    """
    def suggest_taken_action(self):

        # only consider models that have taken their suggested action
        best_action  = None
        best_quality = float("-inf")
        for i, m in enumerate(self.models):
            if m.suggested_action_num_taken() > 0 and self.qualities[i] > best_quality:
                best_quality = self.qualities[i]
                chosen_model = i
                best_action  = self.models[i].suggest_action()

        if not best_action is None:
            self.choices[chosen_model] += 1
            return self.models[chosen_model].suggest_action()
        else:
            self.choices[self.best_model] += 1
            return self.models[self.best_model].suggest_action()


    """
        Returns all the legal actions from the current state
    """
    def get_legal_actions(self):

        # TODO do we need something special here?
        return self.models[self.best_model].get_legal_actions()


    """
        Stops all the models from performing updates to q-values
    """
    def set_no_update(self):

        for m in self.models:
            m.set_no_update()


    """
        Update only the value of the starting state after each transition for all models
    """
    def set_single_update(self):

        for m in self.models:
            m.set_single_update()


    """
        Perform a full value iteration after each transition on all the models
    """
    def set_value_iteration(self, update_error):

        for m in self.models:
            m.set_value_iteration(update_error)


    """
        Perform prioritized sweeping after each transition on all models
    """
    def set_prioritized_sweeping(self, update_error, max_updates):

        for m in self.models:
            m.set_prioritized_sweeping(update_error, max_updates)


    """
        Updates the model after taking the given action and ending up in the state
        corresponding to the give measurements.
    """
    def update(self, action, measurements, reward, debug=False):
        
        if debug:
            print("action = " + str(action))
            print("measur = " + str(measurements))
            print("reward = " + str(reward))
            print("max reward = " + str(self.max_reward))
            print("min reward = " + str(self.min_reward) + '\n')

        best_quality = 0
        self.best_model = 0

        # update the max and min rewards
        if reward > self.max_reward:
            self.max_reward = reward
        if reward < self.min_reward:
            self.min_reward = reward

        # update all the models and calculate the new model qualities
        for i in range(len(self.models)):

            # perform the model update and get the errors
            errors = self.models[i].update(action, measurements, reward, debug)
            if errors is None: # the action is not permissible in that model
                continue
            t_error, r_error, n = errors

            # normalize the reward error to [0,1]
            if n == 1:
                r_error = 1
            elif self.max_reward > self.min_reward:
                r_error *= (n / (self.max_reward - self.min_reward)) ** 2
            else:
                r_error = 0
            
            qr = 1 - r_error                                     # instantaneous reward quality
            qt = 1 - t_error                                     # instantaneous transition quality
            c  = min((n - 1) / self.training_window, 1)          # confidence
            ri = self.reward_importance                          # reward importance
            q  = (ri * qr + (1 - ri) * qt)                       # instantaneous quality
            g  = self.quality_rate                               # quality adjustment coefficient
            self.qualities[i] += g * c * (q - self.qualities[i]) # model quality

            # update the new best model
            if self.qualities[i] > best_quality:
                best_quality    = self.qualities[i]
                self.best_model = i

            if debug:
                print("model %d:" % i)
                print("n  = %d" % n)
                print("r_error = %f" % r_error)
                print("t_error = %f" % t_error)
                print("qr = %f" % qr)
                print("qt = %f" % qt)
                print("c  = %f" % c)
                print("q  = %f" % q)
                print("Q  = %f\n" % self.qualities[i])


    """
        Runs the value iteration algorithm on all the sub-models
    """
    def value_iteration(self, error=None, verbose=False):

        for i in range(len(self.models)):
            self.models[i].value_iteration(error)
            if verbose:
                print("Model %d: value iteration complete!" % i)


    """
        Runs prioritized sweeping starting from the current state on all the sub-models
    """
    def prioritized_sweeping(self, error=None, max_updates=None):
    
        for m in self.models:
            m.prioritized_sweeping(None, error, max_updates)


    """
        Prints the parameters of all the submodels along with their quality factor
    """
    def print_model(self, detailed=False, model_detailed=False):

        for i in range(len(self.models)):
            params  = self.models[i].get_parameters()
            quality = self.qualities[i]
            choices = self.choices[i]
            print("Q: %.12f, P: %s, C: %d" % (quality, str(params), choices))
            if detailed:
                self.models[i].print_model(model_detailed)
                print("")


    """
        Zeroes the counters for the number of times each model has been chosen to suggest an action
    """
    def zero_choice_count(self):

        self.choices = [0] * len(self.models)


    """
        Returns a list of the percentages of the actions than have never been taken
        for each sub-model
    """
    def get_percent_not_taken(self):

        return [(str(m.get_parameters()), m.get_percent_not_taken()) for m in self.models]

    
    """
        Returns a list containing all the qualities of the models
    """
    def get_qualities(self):
        
        return list(self.qualities)


