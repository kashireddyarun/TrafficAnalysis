"""
Intelligent Traffic Signal Optimization System
Optimizes traffic light timing based on real-time traffic conditions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict, deque

class SignalState(Enum):
    """Traffic signal states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"

@dataclass
class SignalTiming:
    """Traffic signal timing configuration"""
    green_time: float
    yellow_time: float = 3.0
    red_time: float = 1.0
    min_green: float = 10.0
    max_green: float = 60.0

@dataclass
class TrafficDemand:
    """Traffic demand data for signal optimization"""
    vehicle_count: int
    queue_length: int
    average_speed: float
    waiting_time: float
    direction: str  # 'north', 'south', 'east', 'west'

class SignalOptimizer:
    """Intelligent traffic signal optimization system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.signal_config = config.get('traffic_management', {}).get('signal_optimization', {})
        
        # Default timing parameters
        self.min_green_time = self.signal_config.get('min_green_time', 10)
        self.max_green_time = self.signal_config.get('max_green_time', 60)
        self.default_cycle_time = self.signal_config.get('default_cycle_time', 120)
        
        # Signal states for each direction
        self.signal_states = {
            'north_south': SignalState.GREEN,
            'east_west': SignalState.RED
        }
        
        # Timing history and optimization data
        self.demand_history = defaultdict(lambda: deque(maxlen=50))
        self.performance_metrics = deque(maxlen=100)
        
        # Current timing
        self.current_timing = {
            'north_south': SignalTiming(green_time=30),
            'east_west': SignalTiming(green_time=30)
        }
        
        # Optimization algorithms
        self.optimization_methods = {
            'adaptive': self._adaptive_optimization,
            'webster': self._webster_optimization,
            'reinforcement': self._rl_optimization
        }
        
    def calculate_traffic_demand(self, detections: List, tracks: List, 
                               direction: str) -> TrafficDemand:
        """Calculate traffic demand for a specific direction"""
        
        # Filter vehicles by direction
        direction_vehicles = self._filter_by_direction(detections, direction)
        direction_tracks = self._filter_tracks_by_direction(tracks, direction)
        
        # Calculate metrics
        vehicle_count = len(direction_vehicles)
        queue_length = self._estimate_queue_length(direction_vehicles)
        avg_speed = self._calculate_average_speed(direction_tracks)
        waiting_time = self._estimate_waiting_time(direction_tracks)
        
        return TrafficDemand(
            vehicle_count=vehicle_count,
            queue_length=queue_length,
            average_speed=avg_speed,
            waiting_time=waiting_time,
            direction=direction
        )
    
    def _filter_by_direction(self, detections: List, direction: str) -> List:
        """Filter detections by traffic direction"""
        # This would need actual direction detection logic
        # For now, simulate based on position
        if direction in ['north', 'south']:
            return [d for d in detections if d.get('lane', 'ns') == 'ns']
        else:
            return [d for d in detections if d.get('lane', 'ew') == 'ew']
    
    def _filter_tracks_by_direction(self, tracks: List, direction: str) -> List:
        """Filter tracks by movement direction"""
        return [t for t in tracks if t.get('direction', direction) == direction]
    
    def _estimate_queue_length(self, vehicles: List) -> int:
        """Estimate queue length based on vehicle positions and speeds"""
        if not vehicles:
            return 0
            
        # Count stationary or slow-moving vehicles
        stopped_vehicles = sum(1 for v in vehicles 
                             if v.get('speed', 0) < 2)  # < 2 km/h
        return stopped_vehicles
    
    def _calculate_average_speed(self, tracks: List) -> float:
        """Calculate average speed for tracks"""
        if not tracks:
            return 0.0
            
        speeds = [t.get('speed', 0) for t in tracks if t.get('speed', 0) > 0]
        return np.mean(speeds) if speeds else 0.0
    
    def _estimate_waiting_time(self, tracks: List) -> float:
        """Estimate average waiting time"""
        if not tracks:
            return 0.0
            
        waiting_times = []
        for track in tracks:
            if track.get('speed', 0) < 1:  # Essentially stopped
                waiting_times.append(track.get('stationary_time', 0))
                
        return np.mean(waiting_times) if waiting_times else 0.0
    
    def _adaptive_optimization(self, demands: Dict[str, TrafficDemand]) -> Dict[str, SignalTiming]:
        """Adaptive signal optimization based on current demand"""
        
        total_demand = sum(d.vehicle_count + d.queue_length for d in demands.values())
        if total_demand == 0:
            return self.current_timing
            
        optimized_timing = {}
        
        for direction, demand in demands.items():
            # Calculate demand ratio
            direction_demand = demand.vehicle_count + demand.queue_length * 2
            demand_ratio = direction_demand / total_demand
            
            # Base green time on demand ratio
            base_green = self.default_cycle_time * 0.5 * demand_ratio
            
            # Adjust for queue length
            queue_adjustment = min(demand.queue_length * 2, 20)
            
            # Adjust for waiting time
            wait_adjustment = min(demand.waiting_time / 10, 15)
            
            # Calculate final green time
            green_time = base_green + queue_adjustment + wait_adjustment
            green_time = max(self.min_green_time, min(green_time, self.max_green_time))
            
            optimized_timing[direction] = SignalTiming(green_time=green_time)
            
        return optimized_timing
    
    def _webster_optimization(self, demands: Dict[str, TrafficDemand]) -> Dict[str, SignalTiming]:
        """Webster's method for signal optimization"""
        
        # Calculate saturation flows and arrival rates
        saturation_flow = 1800  # vehicles per hour per lane
        
        optimized_timing = {}
        total_y = 0  # Total flow ratio
        
        # Calculate flow ratios
        flow_ratios = {}
        for direction, demand in demands.items():
            if demand.vehicle_count > 0:
                # Estimate arrival rate (vehicles per second)
                arrival_rate = demand.vehicle_count * 3600 / 30  # Assume 30-second window
                flow_ratio = arrival_rate / saturation_flow
                flow_ratios[direction] = min(flow_ratio, 0.9)  # Cap at 0.9
                total_y += flow_ratios[direction]
        
        if total_y == 0:
            return self.current_timing
            
        # Calculate optimal cycle time
        lost_time = 4  # seconds per phase
        optimal_cycle = (1.5 * lost_time + 5) / (1 - total_y)
        optimal_cycle = max(60, min(optimal_cycle, 150))  # Constrain cycle time
        
        # Calculate green times
        for direction, demand in demands.items():
            if direction in flow_ratios:
                effective_green = (flow_ratios[direction] / total_y) * (optimal_cycle - lost_time)
                green_time = max(self.min_green_time, min(effective_green, self.max_green_time))
                optimized_timing[direction] = SignalTiming(green_time=green_time)
            else:
                optimized_timing[direction] = SignalTiming(green_time=self.min_green_time)
                
        return optimized_timing
    
    def _rl_optimization(self, demands: Dict[str, TrafficDemand]) -> Dict[str, SignalTiming]:
        """Reinforcement learning-based optimization (simplified)"""
        
        # This is a simplified RL approach - in practice would use trained models
        optimized_timing = {}
        
        for direction, demand in demands.items():
            # State features
            state_features = [
                demand.vehicle_count / 50,  # Normalized vehicle count
                demand.queue_length / 20,   # Normalized queue length
                demand.average_speed / 50,  # Normalized speed
                demand.waiting_time / 60    # Normalized waiting time
            ]
            
            # Simple policy: longer green for higher demand
            demand_score = (demand.vehicle_count * 0.4 + 
                          demand.queue_length * 0.6)
            
            green_time = self.min_green_time + (demand_score / 30) * (self.max_green_time - self.min_green_time)
            green_time = max(self.min_green_time, min(green_time, self.max_green_time))
            
            optimized_timing[direction] = SignalTiming(green_time=green_time)
            
        return optimized_timing
    
    def optimize_signals(self, detections: List, tracks: List, 
                        method: str = 'adaptive') -> Dict[str, SignalTiming]:
        """Main signal optimization function"""
        
        # Calculate demand for each direction
        directions = ['north_south', 'east_west']  # Simplified to two phases
        demands = {}
        
        for direction in directions:
            demands[direction] = self.calculate_traffic_demand(detections, tracks, direction)
            
        # Store demand history
        current_time = time.time()
        for direction, demand in demands.items():
            self.demand_history[direction].append((current_time, demand))
            
        # Apply optimization method
        if method in self.optimization_methods:
            optimized_timing = self.optimization_methods[method](demands)
        else:
            optimized_timing = self._adaptive_optimization(demands)
            
        # Update current timing
        self.current_timing.update(optimized_timing)
        
        return optimized_timing
    
    def calculate_performance_metrics(self, demands: Dict[str, TrafficDemand]) -> Dict:
        """Calculate signal performance metrics"""
        
        total_vehicles = sum(d.vehicle_count for d in demands.values())
        total_waiting_time = sum(d.waiting_time * d.vehicle_count for d in demands.values())
        avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
        
        total_queue_length = sum(d.queue_length for d in demands.values())
        avg_speed = np.mean([d.average_speed for d in demands.values() if d.average_speed > 0])
        
        # Calculate efficiency metrics
        throughput = total_vehicles  # vehicles processed
        delay = avg_waiting_time
        level_of_service = self._calculate_level_of_service(avg_waiting_time, avg_speed)
        
        metrics = {
            'throughput': throughput,
            'average_delay': delay,
            'total_queue_length': total_queue_length,
            'average_speed': avg_speed,
            'level_of_service': level_of_service,
            'timestamp': time.time()
        }
        
        self.performance_metrics.append(metrics)
        return metrics
    
    def _calculate_level_of_service(self, delay: float, speed: float) -> str:
        """Calculate Level of Service (LOS) based on delay and speed"""
        
        if delay <= 10 and speed >= 30:
            return "A"  # Free flow
        elif delay <= 20 and speed >= 20:
            return "B"  # Reasonably free flow
        elif delay <= 35 and speed >= 15:
            return "C"  # Stable flow
        elif delay <= 55 and speed >= 10:
            return "D"  # Approaching unstable flow
        elif delay <= 80 and speed >= 5:
            return "E"  # Unstable flow
        else:
            return "F"  # Forced flow
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Generate recommendations for signal optimization"""
        
        recommendations = []
        
        if len(self.performance_metrics) < 10:
            return [{"type": "info", "message": "Collecting data for analysis"}]
            
        recent_metrics = list(self.performance_metrics)[-10:]
        avg_delay = np.mean([m['average_delay'] for m in recent_metrics])
        avg_throughput = np.mean([m['throughput'] for m in recent_metrics])
        
        if avg_delay > 45:
            recommendations.append({
                "type": "critical",
                "message": "High average delay detected",
                "action": "Consider cycle time reduction or phase rebalancing"
            })
            
        if avg_throughput < 20:
            recommendations.append({
                "type": "warning", 
                "message": "Low throughput detected",
                "action": "Optimize green time allocation"
            })
            
        # Trend analysis
        if len(recent_metrics) >= 5:
            delay_trend = np.polyfit(range(5), [m['average_delay'] for m in recent_metrics[-5:]], 1)[0]
            if delay_trend > 2:
                recommendations.append({
                    "type": "warning",
                    "message": "Delay is increasing",
                    "action": "Monitor and adjust signal timing"
                })
                
        return recommendations
