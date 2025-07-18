"""
Advanced Traffic Congestion Detection System
Analyzes traffic density, speed patterns, and congestion levels
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time

@dataclass
class CongestionMetrics:
    """Metrics for traffic congestion analysis"""
    density: float  # vehicles per unit area
    average_speed: float  # km/h
    congestion_level: str  # "low", "medium", "high", "severe"
    flow_rate: float  # vehicles per minute
    occupancy_ratio: float  # percentage of road occupied
    timestamp: float

class CongestionDetector:
    """Advanced traffic congestion detection and analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.congestion_config = config.get('congestion', {})
        
        # Thresholds
        self.density_threshold = self.congestion_config.get('density_threshold', 0.7)
        self.speed_threshold = self.congestion_config.get('speed_threshold', 5)  # km/h
        self.time_window = self.congestion_config.get('time_window', 30)  # seconds
        
        # Historical data storage
        self.speed_history = deque(maxlen=100)
        self.density_history = deque(maxlen=100)
        self.flow_history = deque(maxlen=100)
        
        # Zone definitions
        self.analysis_zones = self._define_analysis_zones()
        
    def _define_analysis_zones(self) -> List[Dict]:
        """Define road zones for congestion analysis"""
        return [
            {"name": "intersection", "bounds": [(200, 200), (600, 400)]},
            {"name": "lane1", "bounds": [(100, 250), (700, 300)]},
            {"name": "lane2", "bounds": [(100, 350), (700, 400)]},
        ]
    
    def calculate_density(self, detections: List, zone_area: float) -> float:
        """Calculate vehicle density in vehicles per square meter"""
        if zone_area == 0:
            return 0.0
        return len(detections) / zone_area
    
    def calculate_occupancy_ratio(self, detections: List, zone_bounds: Tuple) -> float:
        """Calculate percentage of road area occupied by vehicles"""
        if not detections:
            return 0.0
            
        total_vehicle_area = 0
        zone_area = self._calculate_zone_area(zone_bounds)
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            vehicle_area = (x2 - x1) * (y2 - y1)
            total_vehicle_area += vehicle_area
            
        return min(total_vehicle_area / zone_area, 1.0) if zone_area > 0 else 0.0
    
    def _calculate_zone_area(self, zone_bounds: Tuple) -> float:
        """Calculate area of analysis zone"""
        (x1, y1), (x2, y2) = zone_bounds
        return abs(x2 - x1) * abs(y2 - y1)
    
    def analyze_speed_patterns(self, tracks: List) -> Dict:
        """Analyze speed patterns for congestion indicators"""
        if not tracks:
            return {"average_speed": 0, "speed_variance": 0, "slow_vehicles": 0}
            
        speeds = [track.get('speed', 0) for track in tracks if track.get('speed', 0) > 0]
        
        if not speeds:
            return {"average_speed": 0, "speed_variance": 0, "slow_vehicles": 0}
            
        avg_speed = np.mean(speeds)
        speed_variance = np.var(speeds)
        slow_vehicles = sum(1 for speed in speeds if speed < self.speed_threshold)
        
        return {
            "average_speed": avg_speed,
            "speed_variance": speed_variance,
            "slow_vehicles": slow_vehicles,
            "speed_distribution": np.histogram(speeds, bins=5)[0].tolist()
        }
    
    def calculate_flow_rate(self, vehicle_counts: List, time_window: float) -> float:
        """Calculate traffic flow rate (vehicles per minute)"""
        if not vehicle_counts or time_window <= 0:
            return 0.0
            
        total_vehicles = sum(vehicle_counts)
        return (total_vehicles / time_window) * 60  # vehicles per minute
    
    def detect_congestion_level(self, metrics: CongestionMetrics) -> str:
        """Determine congestion level based on multiple metrics"""
        congestion_score = 0
        
        # Density factor (0-3 points)
        if metrics.density > self.density_threshold * 1.5:
            congestion_score += 3
        elif metrics.density > self.density_threshold:
            congestion_score += 2
        elif metrics.density > self.density_threshold * 0.5:
            congestion_score += 1
            
        # Speed factor (0-3 points)
        if metrics.average_speed < self.speed_threshold * 0.5:
            congestion_score += 3
        elif metrics.average_speed < self.speed_threshold:
            congestion_score += 2
        elif metrics.average_speed < self.speed_threshold * 2:
            congestion_score += 1
            
        # Occupancy factor (0-2 points)
        if metrics.occupancy_ratio > 0.8:
            congestion_score += 2
        elif metrics.occupancy_ratio > 0.6:
            congestion_score += 1
            
        # Determine level
        if congestion_score >= 7:
            return "severe"
        elif congestion_score >= 5:
            return "high"
        elif congestion_score >= 3:
            return "medium"
        else:
            return "low"
    
    def analyze_congestion(self, detections: List, tracks: List, 
                          frame_area: float) -> CongestionMetrics:
        """Comprehensive congestion analysis"""
        current_time = time.time()
        
        # Calculate basic metrics
        density = self.calculate_density(detections, frame_area)
        speed_analysis = self.analyze_speed_patterns(tracks)
        avg_speed = speed_analysis['average_speed']
        
        # Calculate occupancy
        frame_bounds = ((0, 0), (800, 600))  # Default frame size
        occupancy_ratio = self.calculate_occupancy_ratio(detections, frame_bounds)
        
        # Calculate flow rate
        vehicle_count = len(detections)
        self.flow_history.append(vehicle_count)
        flow_rate = self.calculate_flow_rate(list(self.flow_history), self.time_window)
        
        # Create metrics object
        metrics = CongestionMetrics(
            density=density,
            average_speed=avg_speed,
            congestion_level="",  # Will be set below
            flow_rate=flow_rate,
            occupancy_ratio=occupancy_ratio,
            timestamp=current_time
        )
        
        # Determine congestion level
        metrics.congestion_level = self.detect_congestion_level(metrics)
        
        # Update history
        self.density_history.append(density)
        self.speed_history.append(avg_speed)
        
        return metrics
    
    def get_congestion_trends(self) -> Dict:
        """Analyze congestion trends over time"""
        if len(self.density_history) < 2:
            return {"trend": "insufficient_data"}
            
        # Calculate trends
        density_trend = np.polyfit(range(len(self.density_history)), 
                                 list(self.density_history), 1)[0]
        speed_trend = np.polyfit(range(len(self.speed_history)), 
                               list(self.speed_history), 1)[0]
        
        return {
            "density_trend": "increasing" if density_trend > 0 else "decreasing",
            "speed_trend": "increasing" if speed_trend > 0 else "decreasing",
            "trend_strength": abs(density_trend),
            "prediction": self._predict_congestion_change(density_trend, speed_trend)
        }
    
    def _predict_congestion_change(self, density_trend: float, speed_trend: float) -> str:
        """Predict if congestion will improve or worsen"""
        if density_trend > 0.1 and speed_trend < -0.5:
            return "worsening_rapidly"
        elif density_trend > 0.05 or speed_trend < -0.2:
            return "worsening"
        elif density_trend < -0.05 and speed_trend > 0.2:
            return "improving"
        else:
            return "stable"
    
    def generate_alerts(self, metrics: CongestionMetrics) -> List[Dict]:
        """Generate congestion alerts and recommendations"""
        alerts = []
        
        if metrics.congestion_level == "severe":
            alerts.append({
                "level": "critical",
                "message": "Severe congestion detected",
                "recommendation": "Consider traffic light optimization or alternate routes"
            })
        elif metrics.congestion_level == "high":
            alerts.append({
                "level": "warning", 
                "message": "High congestion levels",
                "recommendation": "Monitor closely and prepare interventions"
            })
            
        if metrics.average_speed < 2:
            alerts.append({
                "level": "warning",
                "message": "Traffic nearly at standstill",
                "recommendation": "Emergency response may be needed"
            })
            
        if metrics.occupancy_ratio > 0.9:
            alerts.append({
                "level": "info",
                "message": "Road capacity near maximum",
                "recommendation": "Consider reducing vehicle entry"
            })
            
        return alerts
