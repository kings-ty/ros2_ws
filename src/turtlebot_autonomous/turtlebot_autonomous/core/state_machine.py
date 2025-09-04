
# ==== turtlebot_autonomous/core/state_machine.py ====
"""Robot state machine for behavior management"""

from enum import Enum
import time

class RobotState(Enum):
    """Robot behavior states"""
    INITIALIZING = "initializing"
    SEARCHING = "searching"
    FOLLOWING = "following"
    AVOIDING = "avoiding"
    EMERGENCY = "emergency"
    STOPPED = "stopped"

class StateMachine:
    """Manages robot behavior states and transitions"""
    
    def __init__(self, logger=None):
        self.current_state = RobotState.INITIALIZING
        self.previous_state = None
        self.state_entry_time = time.time()
        self.logger = logger
        self._state_history = []
        
    def transition_to(self, new_state):
        """Transition to new state with logging"""
        if self.current_state != new_state:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_entry_time = time.time()
            
            # Keep state history
            self._state_history.append({
                'state': new_state,
                'timestamp': self.state_entry_time,
                'previous': self.previous_state
            })
            
            # Keep only last 10 states
            if len(self._state_history) > 10:
                self._state_history.pop(0)
            
            if self.logger:
                self.logger.info(f'State: {self.previous_state.value} -> {new_state.value}')
    
    def get_current_state(self):
        return self.current_state
    
    def get_time_in_state(self):
        return time.time() - self.state_entry_time
    
    def get_state_history(self):
        return self._state_history.copy()
    
    def is_stuck(self, timeout=5.0):
        """Check if robot has been in same state too long"""
        return self.get_time_in_state() > timeout