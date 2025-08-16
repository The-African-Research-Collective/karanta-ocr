import redis
from typing import Dict, List


class GPURouter:
    def __init__(self, ports: List[str]):
        self.redis_client = redis.Redis(host="localhost", port=6379, db=0)
        self.gpu_queues = [f"gpu_queue_{str(port)}" for port in ports]

    def get_best_queue(self) -> str:
        """Get best queue based on current load"""

        # Get queue lengths
        queue_lengths = {}
        for queue in self.gpu_queues:
            length = self.redis_client.llen(queue)
            queue_lengths[queue] = length

        # Return queue with minimum length
        return min(queue_lengths, key=queue_lengths.get)

    def get_queue_stats(self) -> Dict:
        """Get current queue statistics"""
        stats = {}
        for queue in self.gpu_queues:
            stats[queue] = {
                "length": self.redis_client.llen(queue),
                "processing": self.redis_client.llen(f"{queue}_processing"),
            }
        return stats
