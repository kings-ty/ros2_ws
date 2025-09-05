from enum import Enum
import time

class RobotState(Enum):
    SEARCHING = "searching"
    FOLLOWING = "following"
    AVOIDING = "avoiding"
    EMERGENCY = "emergency"

class StateMachine:
    def __init__(self, logger=None):
        self.current_state = RobotState.SEARCHING
        self.logger = logger
        
    def transition_to(self, new_state):
        if self.current_state != new_state:
            self.current_state = new_state
            if self.logger:
                self.logger.info(f'State: -> {new_state.value}')
    
    def get_current_state(self):
        return self.current_state
