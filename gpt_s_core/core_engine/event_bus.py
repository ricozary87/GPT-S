import heapq
import logging
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Coroutine, Any
from collections import defaultdict
import time

# Setup logger
logger = logging.getLogger('event_bus')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

@dataclass(order=True)
class Event:
    # Use tuple for ordering (priority, timestamp, counter)
    _ordering_key: tuple = field(init=False, repr=False)
    
    event_type: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    priority: int = 5  # Default medium priority (1-10)
    source: str = "system"
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Create immutable ordering key for heap comparison
        self._ordering_key = (self.priority, self.timestamp)

class EventBus:
    def __init__(self, max_queue_size: int = 10000):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_queue = []
        self.counter = 0
        self.max_queue_size = max_queue_size
        self._processing = False
        self._dead_letter_queue = []

    def subscribe(self, event_type: str, callback: Callable[[Event], Any], priority: int = 5):
        """Subscribe to event type with optional priority"""
        if not callable(callback):
            raise ValueError("Callback must be callable")
            
        self.subscribers[event_type].append(callback)
        logger.info(f"New subscriber for {event_type} (total: {len(self.subscribers[event_type]})")

    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event type"""
        if event_type in self.subscribers:
            try:
                self.subscribers[event_type].remove(callback)
                logger.info(f"Unsubscribed from {event_type}")
            except ValueError:
                logger.warning(f"Callback not found for {event_type}")

    def publish(self, event: Event):
        """Publish event to the bus with overflow protection"""
        if len(self.event_queue) >= self.max_queue_size:
            logger.error("Event queue overflow! Moving to DLQ")
            self._dead_letter_queue.append(event)
            return
            
        heapq.heappush(self.event_queue, (event._ordering_key, self.counter, event))
        self.counter += 1
        logger.debug(f"Published event: {event.event_type} (prio: {event.priority})")

    async def process_events(self, max_events: Optional[int] = None):
        """Process events asynchronously with error handling"""
        self._processing = True
        processed = 0
        
        while self.event_queue and (max_events is None or processed < max_events):
            try:
                _, _, event = heapq.heappop(self.event_queue)
                await self._dispatch_event(event)
                processed += 1
            except Exception as e:
                logger.exception(f"Error processing event: {str(e)}")
                self._dead_letter_queue.append(event)
                
        self._processing = False
        logger.info(f"Processed {processed} events. Remaining: {len(self.event_queue)}")

    async def _dispatch_event(self, event: Event):
        """Dispatch event to subscribers with async support"""
        callbacks = self.subscribers.get(event.event_type, [])
        
        if not callbacks:
            logger.warning(f"No subscribers for event type: {event.event_type}")
            return
            
        logger.debug(f"Dispatching {event.event_type} to {len(callbacks)} subscribers")
        
        for callback in callbacks:
            try:
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    # Run synchronous callbacks in thread pool
                    await asyncio.to_thread(callback, event)
            except Exception as e:
                logger.error(f"Subscriber error: {callback.__name__} - {str(e)}")

    def dead_letter_count(self) -> int:
        """Get number of events in dead letter queue"""
        return len(self._dead_letter_queue)

    def clear_queue(self):
        """Clear all pending events"""
        self.event_queue = []
        logger.warning("Event queue cleared")

    def get_queue_size(self) -> int:
        """Get current event queue size"""
        return len(self.event_queue)

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """Get subscriber count for event type or total"""
        if event_type:
            return len(self.subscribers.get(event_type, []))
        return sum(len(subs) for subs in self.subscribers.values())

# Example usage
if __name__ == "__main__":
    async def main():
        bus = EventBus(max_queue_size=100)
        
        # Sample subscribers
        def log_event(event: Event):
            print(f"LOG: {event.event_type} - {event.data}")
        
        async def process_order(event: Event):
            print(f"Processing order: {event.data['order_id']}")
            await asyncio.sleep(0.1)
            
        def critical_alert(event: Event):
            print(f"ALERT! {event.data['message']}")
        
        # Subscribe to events
        bus.subscribe("order_created", log_event)
        bus.subscribe("order_created", process_order)
        bus.subscribe("system_alert", critical_alert, priority=1)
        
        # Publish events
        bus.publish(Event(
            event_type="order_created",
            data={"order_id": "12345", "amount": 100.0},
            priority=3
        ))
        
        bus.publish(Event(
            event_type="system_alert",
            data={"message": "High CPU usage"},
            priority=1  # High priority
        ))
        
        bus.publish(Event(
            event_type="user_login",
            data={"user_id": "user1"},
            priority=7
        ))
        
        # Process events
        await bus.process_events()
        
        print(f"Queue size: {bus.get_queue_size()}")
        print(f"Subscribers: {bus.get_subscriber_count()}")
        print(f"Dead letters: {bus.dead_letter_count()}")

    asyncio.run(main())