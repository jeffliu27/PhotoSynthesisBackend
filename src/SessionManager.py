import string
import random
import uuid

# Description: 
    # Stores global data and helps differentiate between different clients through a session ID.
class SessionManager:
    def __init__(self, max_sessions):
        self.sessions = {} #session_id -> album
        self.max_sessions = max_sessions

    def create_session_id(self):
        while True:
            new_id = str(uuid.uuid1())

            if new_id not in self.sessions:
                return new_id

    def new_session(self,album):
        if len(self.sessions) < self.max_sessions:
            session_id = self.create_session_id()

        self.sessions[session_id] = album
        return session_id