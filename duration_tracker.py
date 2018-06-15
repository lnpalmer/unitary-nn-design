import time

class DurationTracker:

    def __init__(self):
        self.durations_ = {}
    
    def track(self, tag):
        callback = lambda duration: self.record_duration(duration, tag)
        return TrackedDuration(callback)
    
    def record_duration(self, duration, tag="miscellaneous"):
        if not tag in self.durations_:
            self.durations_[tag] = []
        self.durations_[tag].append(duration)

class TrackedDuration:

    def __init__(self, callback):
        self.callback_ = callback
    
    def __enter__(self):
        self.start_ = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_
        self.callback_(duration)