"""
Traffic Visualization Tools
Handles drawing and visualization of traffic analysis results
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import colorsys

class TrafficVisualizer:
    """
    Comprehensive visualization tools for traffic analysis
    """
    
    def __init__(self):
        """Initialize traffic visualizer"""
        self.colors = self._generate_colors(50)  # Generate 50 distinct colors
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Color scheme
        self.vehicle_color = (0, 255, 0)      # Green
        self.pedestrian_color = (255, 0, 0)   # Blue
        self.traffic_light_color = (0, 0, 255) # Red
        self.counting_line_color = (0, 255, 255) # Yellow
        self.text_color = (255, 255, 255)     # White
        self.track_colors = {}  # Track ID to color mapping
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # Choose color based on object type
            if class_id in [2, 3, 5, 7]:  # Vehicles
                color = self.vehicle_color
            elif class_id == 0:  # Pedestrians
                color = self.pedestrian_color
            elif class_id == 9:  # Traffic lights
                color = self.traffic_light_color
            else:
                color = (255, 255, 0)  # Cyan for others
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, self.thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            
            # Background for text
            cv2.rectangle(annotated_frame,
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label,
                       (bbox[0], bbox[1] - 5),
                       self.font, self.font_scale,
                       self.text_color, self.thickness)
        
        return annotated_frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """
        Draw tracking information on frame
        
        Args:
            frame: Input frame
            tracks: List of tracks
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            center = track['center']
            class_name = track['class_name']
            
            # Get consistent color for this track
            if track_id not in self.track_colors:
                color_idx = track_id % len(self.colors)
                self.track_colors[track_id] = self.colors[color_idx]
            
            color = self.track_colors[track_id]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, self.thickness)
            
            # Draw center point
            cv2.circle(annotated_frame, tuple(center), 5, color, -1)
            
            # Draw track ID
            label = f"ID:{track_id} ({class_name})"
            cv2.putText(annotated_frame, label,
                       (bbox[0], bbox[1] - 10),
                       self.font, self.font_scale,
                       color, self.thickness)
            
            # Draw trajectory if available
            if hasattr(track, 'trajectory') and len(track.get('trajectory', [])) > 1:
                trajectory = track['trajectory']
                for i in range(1, len(trajectory)):
                    cv2.line(annotated_frame, 
                            tuple(trajectory[i-1]), 
                            tuple(trajectory[i]), 
                            color, 2)
        
        return annotated_frame
    
    def draw_counting_lines(self, frame: np.ndarray, counting_lines: List[List[List[int]]], 
                           line_stats: Optional[Dict] = None) -> np.ndarray:
        """
        Draw counting lines and statistics
        
        Args:
            frame: Input frame
            counting_lines: List of counting lines
            line_stats: Optional statistics for each line
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for i, line in enumerate(counting_lines):
            color = self.counting_line_color
            thickness = 4
            
            # Draw line
            cv2.line(annotated_frame, tuple(line[0]), tuple(line[1]), color, thickness)
            
            # Draw line number
            mid_point = [(line[0][0] + line[1][0]) // 2, (line[0][1] + line[1][1]) // 2]
            
            # Prepare label
            if line_stats and i in line_stats:
                count = line_stats[i].get('total_crossings', 0)
                label = f"Line {i}: {count}"
            else:
                label = f"Line {i}"
            
            # Background for text
            label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]
            cv2.rectangle(annotated_frame,
                         (mid_point[0] - 5, mid_point[1] - label_size[1] - 5),
                         (mid_point[0] + label_size[0] + 5, mid_point[1] + 5),
                         color, -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, 
                       tuple(mid_point), 
                       self.font, self.font_scale, 
                       (0, 0, 0), self.thickness)
        
        return annotated_frame
    
    def draw_statistics_panel(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Draw statistics panel on frame
        
        Args:
            frame: Input frame
            stats: Statistics dictionary
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Panel properties
        panel_width = 300
        panel_height = 200
        panel_x = frame.shape[1] - panel_width - 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(annotated_frame,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        
        cv2.rectangle(annotated_frame,
                     (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (255, 255, 255), 2)
        
        # Draw statistics
        y_offset = panel_y + 25
        line_height = 20
        
        # Title
        cv2.putText(annotated_frame, "Traffic Statistics", 
                   (panel_x + 10, y_offset),
                   self.font, 0.7, (255, 255, 255), 2)
        y_offset += line_height + 10
        
        # Extract statistics
        flow_stats = stats.get('flow_stats', {})
        counts = flow_stats.get('counts', {})
        density = flow_stats.get('density', {})
        
        # Vehicle count
        vehicle_count = counts.get('total_vehicles', 0)
        cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", 
                   (panel_x + 10, y_offset),
                   self.font, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        # Pedestrian count
        pedestrian_count = counts.get('total_pedestrians', 0)
        cv2.putText(annotated_frame, f"Pedestrians: {pedestrian_count}", 
                   (panel_x + 10, y_offset),
                   self.font, 0.5, (255, 0, 0), 1)
        y_offset += line_height
        
        # Density
        vehicle_density = density.get('vehicle_density', 0)
        cv2.putText(annotated_frame, f"Density: {vehicle_density:.3f}", 
                   (panel_x + 10, y_offset),
                   self.font, 0.5, (255, 255, 0), 1)
        y_offset += line_height
        
        # Congestion level
        congestion = density.get('congestion_level', 'Unknown')
        color = self._get_congestion_color(congestion)
        cv2.putText(annotated_frame, f"Congestion: {congestion}", 
                   (panel_x + 10, y_offset),
                   self.font, 0.5, color, 1)
        y_offset += line_height
        
        # Active tracks
        active_tracks = stats.get('tracks', 0)
        cv2.putText(annotated_frame, f"Active Tracks: {active_tracks}", 
                   (panel_x + 10, y_offset),
                   self.font, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def _get_congestion_color(self, congestion_level: str) -> Tuple[int, int, int]:
        """Get color based on congestion level"""
        colors = {
            'Free Flow': (0, 255, 0),    # Green
            'Light': (0, 255, 255),      # Yellow
            'Moderate': (0, 165, 255),   # Orange
            'Heavy': (0, 0, 255),        # Red
            'Unknown': (255, 255, 255)   # White
        }
        return colors.get(congestion_level, (255, 255, 255))
    
    def draw_speed_information(self, frame: np.ndarray, tracks: List[Dict], 
                              pixel_per_meter: float = 10, fps: float = 30) -> np.ndarray:
        """
        Draw speed information for tracked objects
        
        Args:
            frame: Input frame
            tracks: List of tracks with speed information
            pixel_per_meter: Conversion factor
            fps: Frame rate
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for track in tracks:
            if 'speed' in track and track['speed'] > 0:
                bbox = track['bbox']
                speed = track['speed']
                
                # Draw speed label
                speed_label = f"{speed:.1f} km/h"
                cv2.putText(annotated_frame, speed_label,
                           (bbox[0], bbox[3] + 20),
                           self.font, 0.5, (255, 255, 0), 2)
        
        return annotated_frame
    
    def draw_complete_analysis(self, frame: np.ndarray, detections: List[Dict], 
                              tracks: List[Dict], flow_stats: Dict, 
                              counting_lines: List[List[List[int]]]) -> np.ndarray:
        """
        Draw complete traffic analysis visualization
        
        Args:
            frame: Input frame
            detections: Object detections
            tracks: Object tracks
            flow_stats: Flow analysis statistics
            counting_lines: Counting lines
            
        Returns:
            Fully annotated frame
        """
        # Start with original frame
        annotated_frame = frame.copy()
        
        # Draw counting lines first (so they appear behind objects)
        line_stats = flow_stats.get('line_statistics', {})
        annotated_frame = self.draw_counting_lines(annotated_frame, counting_lines, line_stats)
        
        # Draw tracks (with trajectories)
        annotated_frame = self.draw_tracks(annotated_frame, tracks)
        
        # Draw speed information
        annotated_frame = self.draw_speed_information(annotated_frame, tracks)
        
        # Draw statistics panel
        stats = {
            'flow_stats': flow_stats,
            'tracks': len(tracks),
            'detections': len(detections)
        }
        annotated_frame = self.draw_statistics_panel(annotated_frame, stats)
        
        # Draw frame number and timestamp
        frame_info = f"Frame: {flow_stats.get('frame_number', 0)}"
        cv2.putText(annotated_frame, frame_info,
                   (10, frame.shape[0] - 10),
                   self.font, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def create_traffic_heatmap(self, detection_history: List[List[Dict]], 
                              frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create traffic density heatmap from detection history
        
        Args:
            detection_history: List of detections for each frame
            frame_shape: (height, width) of frames
            
        Returns:
            Heatmap as numpy array
        """
        height, width = frame_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate detections
        for frame_detections in detection_history:
            for detection in frame_detections:
                if detection['class_id'] in [2, 3, 5, 7]:  # Vehicles only
                    center = detection['center']
                    x, y = center
                    
                    # Add Gaussian blur around detection center
                    if 0 <= x < width and 0 <= y < height:
                        y_start = max(0, y - 25)
                        y_end = min(height, y + 25)
                        x_start = max(0, x - 25)
                        x_end = min(width, x + 25)
                        
                        for i in range(y_start, y_end):
                            for j in range(x_start, x_end):
                                distance = np.sqrt((i - y)**2 + (j - x)**2)
                                if distance <= 25:
                                    weight = np.exp(-(distance**2) / (2 * 10**2))
                                    heatmap[i, j] += weight
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to color heatmap
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        return heatmap_color
    
    def plot_traffic_statistics(self, flow_data: List[Dict], save_path: Optional[str] = None):
        """
        Plot traffic statistics over time
        
        Args:
            flow_data: List of flow statistics over time
            save_path: Optional path to save the plot
        """
        if not flow_data:
            print("No flow data available for plotting")
            return
        
        # Extract data for plotting
        timestamps = [data['timestamp'] for data in flow_data]
        vehicle_counts = [data['counts']['total_vehicles'] for data in flow_data]
        pedestrian_counts = [data['counts']['total_pedestrians'] for data in flow_data]
        densities = [data['density']['vehicle_density'] for data in flow_data]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Traffic Analysis Statistics', fontsize=16)
        
        # Vehicle counts over time
        axes[0, 0].plot(timestamps, vehicle_counts, 'g-', linewidth=2)
        axes[0, 0].set_title('Vehicle Count Over Time')
        axes[0, 0].set_ylabel('Number of Vehicles')
        axes[0, 0].grid(True)
        
        # Pedestrian counts over time
        axes[0, 1].plot(timestamps, pedestrian_counts, 'b-', linewidth=2)
        axes[0, 1].set_title('Pedestrian Count Over Time')
        axes[0, 1].set_ylabel('Number of Pedestrians')
        axes[0, 1].grid(True)
        
        # Traffic density over time
        axes[1, 0].plot(timestamps, densities, 'r-', linewidth=2)
        axes[1, 0].set_title('Traffic Density Over Time')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True)
        
        # Vehicle type distribution (if available)
        if flow_data and 'vehicle_types' in flow_data[-1]['counts']:
            vehicle_types = flow_data[-1]['counts']['vehicle_types']
            if vehicle_types:
                axes[1, 1].pie(vehicle_types.values(), labels=vehicle_types.keys(), autopct='%1.1f%%')
                axes[1, 1].set_title('Vehicle Type Distribution')
            else:
                axes[1, 1].text(0.5, 0.5, 'No vehicle type data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'No vehicle type data', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Traffic statistics plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()

if __name__ == "__main__":
    # Example usage
    visualizer = TrafficVisualizer()
    
    # Create sample frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Sample detections
    detections = [
        {'bbox': [100, 100, 150, 150], 'confidence': 0.9, 'class_id': 2, 'class_name': 'car'},
        {'bbox': [200, 200, 230, 250], 'confidence': 0.8, 'class_id': 0, 'class_name': 'person'}
    ]
    
    # Sample tracks
    tracks = [
        {'track_id': 1, 'bbox': [100, 100, 150, 150], 'center': [125, 125], 'class_name': 'car'},
        {'track_id': 2, 'bbox': [200, 200, 230, 250], 'center': [215, 225], 'class_name': 'person'}
    ]
    
    # Sample flow stats
    flow_stats = {
        'counts': {'total_vehicles': 1, 'total_pedestrians': 1},
        'density': {'vehicle_density': 0.1, 'congestion_level': 'Light'},
        'frame_number': 1
    }
    
    # Sample counting lines
    counting_lines = [[[100, 300], [500, 300]]]
    
    # Draw complete analysis
    result = visualizer.draw_complete_analysis(frame, detections, tracks, flow_stats, counting_lines)
    
    print("TrafficVisualizer test completed successfully!")
