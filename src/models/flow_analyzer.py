"""
Traffic Flow Analysis System
Analyzes traffic patterns, counts vehicles, and measures flow characteristics
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import time

class TrafficFlowAnalyzer:
    """
    Analyzes traffic flow patterns and statistics
    """
    
    def __init__(self, counting_lines: List[List[List[int]]] = None, fps: float = 30):
        """
        Initialize traffic flow analyzer
        
        Args:
            counting_lines: List of counting lines defined as [[x1, y1], [x2, y2]]
            fps: Video frame rate for time-based calculations
        """
        self.counting_lines = counting_lines or []
        self.fps = fps
        
        # Counters for each line
        self.line_counters = {}
        self.crossed_tracks = {}  # Track which objects have crossed which lines
        
        # Flow statistics
        self.flow_data = defaultdict(list)
        self.vehicle_counts = defaultdict(int)
        self.speed_data = defaultdict(list)
        
        # Time-based analysis
        self.time_window = 60  # seconds
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize line counters
        for i, line in enumerate(self.counting_lines):
            self.line_counters[i] = {
                'total': 0,
                'vehicles': defaultdict(int),
                'direction_1': 0,  # crossing from side 1 to side 2
                'direction_2': 0,  # crossing from side 2 to side 1
                'hourly_counts': deque(maxlen=60)  # Last 60 minutes
            }
            self.crossed_tracks[i] = set()
    
    def analyze_frame(self, tracks: List[Dict], frame_shape: Tuple[int, int]) -> Dict:
        """
        Analyze traffic flow for current frame
        
        Args:
            tracks: List of confirmed tracks from tracker
            frame_shape: (height, width) of the frame
            
        Returns:
            Flow analysis results
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Process line crossings
        for track in tracks:
            self._check_line_crossings(track)
        
        # Calculate flow statistics
        flow_stats = self._calculate_flow_statistics(tracks, frame_shape)
        
        # Update time-based counters
        if self.frame_count % (self.fps * 60) == 0:  # Every minute
            self._update_hourly_counts()
        
        return flow_stats
    
    def _check_line_crossings(self, track: Dict):
        """Check if track has crossed any counting lines"""
        track_id = track['track_id']
        current_pos = track['center']
        
        for line_idx, line in enumerate(self.counting_lines):
            line_key = f"{line_idx}_{track_id}"
            
            # Skip if already counted for this line
            if track_id in self.crossed_tracks[line_idx]:
                continue
            
            # Check if track crosses the line
            if self._point_crosses_line(current_pos, line):
                self._record_crossing(track, line_idx)
                self.crossed_tracks[line_idx].add(track_id)
    
    def _point_crosses_line(self, point: List[int], line: List[List[int]]) -> bool:
        """
        Check if point crosses a line (simplified crossing detection)
        
        Args:
            point: [x, y] coordinates
            line: [[x1, y1], [x2, y2]] line endpoints
            
        Returns:
            True if crossing detected
        """
        x, y = point
        x1, y1 = line[0]
        x2, y2 = line[1]
        
        # Calculate distance from point to line
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return False
        
        distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_length
        
        # Consider crossing if point is very close to line (within 10 pixels)
        return distance < 10
    
    def _record_crossing(self, track: Dict, line_idx: int):
        """Record a line crossing event"""
        class_name = track['class_name']
        
        # Update counters
        self.line_counters[line_idx]['total'] += 1
        self.line_counters[line_idx]['vehicles'][class_name] += 1
        
        # Determine direction (simplified - based on position relative to line)
        line = self.counting_lines[line_idx]
        track_pos = track['center']
        
        # Calculate which side of the line the track is on
        side = self._get_line_side(track_pos, line)
        if side > 0:
            self.line_counters[line_idx]['direction_1'] += 1
        else:
            self.line_counters[line_idx]['direction_2'] += 1
        
        print(f"Vehicle {track['track_id']} ({class_name}) crossed line {line_idx}")
    
    def _get_line_side(self, point: List[int], line: List[List[int]]) -> float:
        """Determine which side of line a point is on"""
        x, y = point
        x1, y1 = line[0]
        x2, y2 = line[1]
        
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    
    def _calculate_flow_statistics(self, tracks: List[Dict], frame_shape: Tuple[int, int]) -> Dict:
        """Calculate comprehensive flow statistics"""
        height, width = frame_shape
        frame_area = height * width
        
        # Basic counts
        total_vehicles = len([t for t in tracks if t['class_id'] in [2, 3, 5, 7]])
        total_pedestrians = len([t for t in tracks if t['class_id'] == 0])
        
        # Vehicle type breakdown
        vehicle_types = defaultdict(int)
        for track in tracks:
            if track['class_id'] in [2, 3, 5, 7]:
                vehicle_types[track['class_name']] += 1
        
        # Density calculation
        total_vehicle_area = sum(t['area'] for t in tracks if t['class_id'] in [2, 3, 5, 7])
        density = total_vehicle_area / frame_area if frame_area > 0 else 0
        
        # Speed analysis (if available)
        speeds = []
        for track in tracks:
            if hasattr(track, 'speed') and track.get('speed', 0) > 0:
                speeds.append(track['speed'])
        
        avg_speed = np.mean(speeds) if speeds else 0
        speed_variance = np.var(speeds) if speeds else 0
        
        # Congestion level
        congestion_level = self._assess_congestion_level(density, avg_speed, total_vehicles)
        
        # Flow rate (vehicles per minute)
        elapsed_time = max(self.frame_count / self.fps / 60, 1/60)  # minutes
        total_counted = sum(counter['total'] for counter in self.line_counters.values())
        flow_rate = total_counted / elapsed_time if total_counted > 0 else 0
        
        return {
            'timestamp': time.time(),
            'frame_number': self.frame_count,
            'counts': {
                'total_vehicles': total_vehicles,
                'total_pedestrians': total_pedestrians,
                'vehicle_types': dict(vehicle_types)
            },
            'density': {
                'vehicle_density': density,
                'congestion_level': congestion_level
            },
            'speed': {
                'average_speed': avg_speed,
                'speed_variance': speed_variance,
                'speed_distribution': speeds
            },
            'flow': {
                'flow_rate': flow_rate,  # vehicles per minute
                'line_crossings': {i: counter['total'] for i, counter in self.line_counters.items()}
            },
            'line_statistics': self.get_line_statistics()
        }
    
    def _assess_congestion_level(self, density: float, avg_speed: float, vehicle_count: int) -> str:
        """Assess traffic congestion level"""
        # Thresholds (adjustable based on requirements)
        if density > 0.3 or avg_speed < 10 or vehicle_count > 20:
            return "Heavy"
        elif density > 0.2 or avg_speed < 20 or vehicle_count > 10:
            return "Moderate"
        elif density > 0.1 or avg_speed < 40 or vehicle_count > 5:
            return "Light"
        else:
            return "Free Flow"
    
    def get_line_statistics(self) -> Dict:
        """Get detailed statistics for each counting line"""
        stats = {}
        
        for line_idx, counter in self.line_counters.items():
            stats[line_idx] = {
                'total_crossings': counter['total'],
                'vehicle_breakdown': dict(counter['vehicles']),
                'direction_1_count': counter['direction_1'],
                'direction_2_count': counter['direction_2'],
                'net_flow': counter['direction_1'] - counter['direction_2'],
                'coordinates': self.counting_lines[line_idx]
            }
        
        return stats
    
    def _update_hourly_counts(self):
        """Update hourly counting statistics"""
        current_minute_total = sum(counter['total'] for counter in self.line_counters.values())
        
        for counter in self.line_counters.values():
            counter['hourly_counts'].append(current_minute_total)
    
    def get_hourly_flow_rate(self, line_idx: Optional[int] = None) -> float:
        """Get vehicles per hour for a specific line or overall"""
        if line_idx is not None and line_idx in self.line_counters:
            counts = self.line_counters[line_idx]['hourly_counts']
        else:
            # Overall flow rate
            all_counts = []
            for counter in self.line_counters.values():
                all_counts.extend(counter['hourly_counts'])
            counts = all_counts
        
        if not counts:
            return 0.0
        
        # Calculate vehicles per hour based on recent data
        total_vehicles = sum(counts)
        time_period_hours = len(counts) / 60  # minutes to hours
        
        return total_vehicles / time_period_hours if time_period_hours > 0 else 0.0
    
    def add_counting_line(self, line: List[List[int]]):
        """Add a new counting line"""
        line_idx = len(self.counting_lines)
        self.counting_lines.append(line)
        
        self.line_counters[line_idx] = {
            'total': 0,
            'vehicles': defaultdict(int),
            'direction_1': 0,
            'direction_2': 0,
            'hourly_counts': deque(maxlen=60)
        }
        self.crossed_tracks[line_idx] = set()
    
    def draw_counting_lines(self, frame: np.ndarray) -> np.ndarray:
        """Draw counting lines on frame for visualization"""
        annotated_frame = frame.copy()
        
        for i, line in enumerate(self.counting_lines):
            color = (0, 255, 255)  # Yellow
            thickness = 3
            
            # Draw line
            cv2.line(annotated_frame, tuple(line[0]), tuple(line[1]), color, thickness)
            
            # Draw line number and count
            mid_point = [(line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2]
            text = f"Line {i}: {self.line_counters[i]['total']}"
            
            cv2.putText(annotated_frame, text, tuple(mid_point), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def generate_flow_report(self) -> Dict:
        """Generate comprehensive flow analysis report"""
        total_time = self.frame_count / self.fps / 3600  # hours
        
        report = {
            'analysis_duration': total_time,
            'total_frames': self.frame_count,
            'counting_lines': len(self.counting_lines),
            'line_statistics': self.get_line_statistics(),
            'hourly_flow_rates': {
                i: self.get_hourly_flow_rate(i) for i in range(len(self.counting_lines))
            },
            'overall_flow_rate': self.get_hourly_flow_rate(),
            'peak_analysis': self._analyze_peak_hours(),
            'summary': {
                'total_vehicles_counted': sum(counter['total'] for counter in self.line_counters.values()),
                'average_vehicles_per_hour': self.get_hourly_flow_rate(),
                'busiest_line': self._get_busiest_line()
            }
        }
        
        return report
    
    def _analyze_peak_hours(self) -> Dict:
        """Analyze peak traffic hours"""
        # This is a simplified analysis - in practice, you'd analyze by actual time
        peak_data = {
            'peak_hour_estimate': 'Analysis requires longer observation period',
            'traffic_pattern': 'Continuous monitoring needed for pattern detection'
        }
        
        return peak_data
    
    def _get_busiest_line(self) -> Dict:
        """Find the busiest counting line"""
        if not self.line_counters:
            return {'line_id': None, 'count': 0}
        
        busiest_line = max(self.line_counters.items(), key=lambda x: x[1]['total'])
        
        return {
            'line_id': busiest_line[0],
            'count': busiest_line[1]['total'],
            'coordinates': self.counting_lines[busiest_line[0]]
        }

if __name__ == "__main__":
    # Example usage
    counting_lines = [
        [[100, 300], [500, 300]],  # Horizontal line
        [[300, 100], [300, 500]]   # Vertical line
    ]
    
    analyzer = TrafficFlowAnalyzer(counting_lines)
    
    # Simulate some tracks
    sample_tracks = [
        {'track_id': 1, 'center': [250, 250], 'class_id': 2, 'class_name': 'car', 'area': 2000},
        {'track_id': 2, 'center': [350, 350], 'class_id': 3, 'class_name': 'motorcycle', 'area': 1000}
    ]
    
    # Analyze frame
    results = analyzer.analyze_frame(sample_tracks, (600, 800))
    print("Flow analysis results:", results)
    
    # Generate report
    report = analyzer.generate_flow_report()
    print("Flow report:", report)
