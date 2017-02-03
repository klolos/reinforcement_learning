
"""
    Error class for unsupported or missing parameters
"""
class ParameterError(Exception):

    def __init__(self, message, logger=None):

        super(ParameterError, self).__init__(message)

        if not logger is None:
            logger.error(message)


"""
    Raised when the model is used before setting the initial state
"""
class StateNotSetError(Exception):

    def __init__(self, logger=None):

        super(StateNotSetError, self).__init__("State has not been set")

        if not logger is None:
            logger.error("State has not been set")


"""
    Raised for errors that are caused by internal bugs.
    These will never happen.
"""
class InternalError(Exception):

    def __init__(self, message, logger=None):

        super(InternalError, self).__init__(message)

        if not logger is None:
            logger.error(message)


"""
    Error class for errors in the configuration file
"""
class ConfigurationError(Exception):

    def __init__(self, message, logger=None):

        super(ConfigurationError, self).__init__(message)

        if not logger is None:
            logger.error(message)

