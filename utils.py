import time
from functools import wraps
from collections import deque
from datetime import datetime, timedelta

class RateLimiter:
    def __init__(self, max_requests=10, time_window=60):
        self.max_requests = max_requests
        self.time_window = time_window  # in seconds
        self.requests = deque()
    
    def can_request(self):
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.time_window)
        
        # Remove old requests
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
        
        return len(self.requests) < self.max_requests
    
    def add_request(self):
        self.requests.append(datetime.now())

def rate_limited(max_requests=30, time_window=60):  # Removed delays, increased max_requests
    limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if limiter.can_request():
                try:
                    limiter.add_request()
                    return func(*args, **kwargs)
                except Exception as e:
                    raise e
            else:
                raise Exception("Rate limit exceeded")
        return wrapper
    return decorator

class ThrottledRequests:
    def __init__(self, base_delay=0.5, max_delay=10, backoff_factor=1.5):  # Reduced delays
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.current_delay = base_delay
        self.last_request_time = 0
        self.consecutive_errors = 0

    def wait(self):
        """Wait appropriate time between requests"""
        now = time.time()
        time_since_last = now - self.last_request_time
        
        if time_since_last < self.current_delay:
            time.sleep(self.current_delay - time_since_last)
        
        self.last_request_time = time.time()

    def success(self):
        """Reset delay after successful request"""
        self.consecutive_errors = 0
        self.current_delay = self.base_delay

    def failure(self):
        """Increase delay after failed request"""
        self.consecutive_errors += 1
        self.current_delay = min(
            self.max_delay,
            self.base_delay * (self.backoff_factor ** self.consecutive_errors)
        )

def throttled(base_delay=0.5, max_delay=10, backoff_factor=1.5):  # Updated parameters
    throttler = ThrottledRequests(base_delay, max_delay, backoff_factor)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            throttler.wait()
            try:
                result = func(*args, **kwargs)
                throttler.success()
                return result
            except Exception as e:
                throttler.failure()
                raise e
        return wrapper
    return decorator
