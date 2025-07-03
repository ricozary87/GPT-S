from dataclasses import dataclass
from typing import Callable
import heapq

@dataclass
class Event:
    timestamp: float
    priority: int  # 1-10 (1=urgent)
    data: dict

class EventBus:
    def __init__(self):
        self.subscribers = {}
        self.event_queue = []

    def subscribe(self, event_type: str, callback: Callable):
        self.subscribers.setdefault(event_type, []).append(callback)

    def publish(self, event: Event):
        heapq.heappush(self.event_queue, (event.priority, event))

    def process_events(self):
        while self.event_queue:
            _, event = heapq.heappop(self.event_queue)
            for callback in self.subscribers.get(event.type, []):
                callback(event.data)
