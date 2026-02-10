
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

class CircuitOpenError(Exception):
    pass

class CircuitBreaker:
    """
    States:
    - CLOSED (normal): requests pass through
    - OPEN (tripped): requests immediately rejected, alert sent
    - HALF_OPEN (testing): allow 1 request to test recovery
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: int = 60,
        on_trip: Optional[Callable[[int], None]] = None
    ):
        self.state = "CLOSED"
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time: Optional[datetime] = None
        self.on_trip = on_trip
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Wrap an async function with circuit breaker logic."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                logger.info("Circuit HALF_OPEN - attempting recovery request")
                self.state = "HALF_OPEN"
            else:
                # If state is OPEN, last_failure_time should be set.
                if self.last_failure_time:
                    remaining = (self.last_failure_time + timedelta(seconds=self.recovery_timeout) - datetime.utcnow()).total_seconds()
                else:
                    remaining = self.recovery_timeout
                raise CircuitOpenError(f"Circuit open, retry after {remaining:.0f}s")
                
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                logger.info("Circuit recovered - CLOSED")
                self._on_success()
            elif self.state == "CLOSED":
                self.failure_count = 0
            return result
        except Exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        if not self.last_failure_time:
            return True
        return (datetime.utcnow() - self.last_failure_time).total_seconds() >= self.recovery_timeout

    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        if self.state != "OPEN" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit tripped OPEN after {self.failure_count} failures")
            if self.on_trip:
                try:
                    self.on_trip(self.failure_count)
                except Exception as e:
                    logger.error(f"Error in on_trip callback: {e}")
