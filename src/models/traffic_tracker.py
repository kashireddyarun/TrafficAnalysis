"""
Traffic Tracking System
Implements object tracking for traffic analysis using Kalman filters and IoU matching
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import uuid

class TrafficTracker:
    """
    Multi-object tracker for traffic analysis
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize traffic tracker
        
        Args:
            max_age: Maximum frames to keep a track without detection
            min_hits: Minimum hits before confirming a track
            iou_threshold: IoU threshold for data association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.track_id_counter = 0
        
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of detections from current frame
            
        Returns:
            List of confirmed tracks with IDs
        """
        self.frame_count += 1
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # Match detections to existing tracks
        matched_tracks, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for match in matched_tracks:
            track_idx, det_idx = match
            self.tracks[track_idx].update(detections[det_idx])
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = TrafficTrack(detections[det_idx], self._get_next_id())
            self.tracks.append(new_track)
        
        # Mark unmatched tracks for deletion
        for track_idx in unmatched_trks:
            self.tracks[track_idx].mark_missed()
        
        # Remove dead tracks
        self.tracks = [track for track in self.tracks if not track.is_dead(self.max_age)]
        
        # Return confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.is_confirmed(self.min_hits):
                track_data = track.get_state()
                track_data['track_id'] = track.track_id
                track_data['age'] = track.age
                track_data['hits'] = track.hits
                confirmed_tracks.append(track_data)
        
        return confirmed_tracks
    
    def _associate_detections_to_tracks(self, detections: List[Dict], tracks: List['TrafficTrack']) -> Tuple[List[Tuple], List[int], List[int]]:
        """
        Associate detections to existing tracks using IoU
        """
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for d_idx, detection in enumerate(detections):
            for t_idx, track in enumerate(tracks):
                iou_matrix[d_idx, t_idx] = self._calculate_iou(
                    detection['bbox'], track.get_predicted_bbox()
                )
        
        # Use Hungarian algorithm for optimal assignment
        if min(iou_matrix.shape) > 0:
            detection_indices, track_indices = linear_sum_assignment(-iou_matrix)
            
            matches = []
            for d_idx, t_idx in zip(detection_indices, track_indices):
                if iou_matrix[d_idx, t_idx] > self.iou_threshold:
                    matches.append((t_idx, d_idx))
            
            matched_track_indices = [match[0] for match in matches]
            matched_detection_indices = [match[1] for match in matches]
            
            unmatched_detections = [i for i in range(len(detections)) 
                                  if i not in matched_detection_indices]
            unmatched_tracks = [i for i in range(len(tracks)) 
                              if i not in matched_track_indices]
            
            return matches, unmatched_detections, unmatched_tracks
        else:
            return [], list(range(len(detections))), list(range(len(tracks)))
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _get_next_id(self) -> int:
        """Get next unique track ID"""
        self.track_id_counter += 1
        return self.track_id_counter
    
    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len([track for track in self.tracks if track.is_confirmed(self.min_hits)])
    
    def get_track_statistics(self) -> Dict:
        """Get tracking statistics"""
        confirmed_tracks = [track for track in self.tracks if track.is_confirmed(self.min_hits)]
        
        stats = {
            'total_tracks': len(confirmed_tracks),
            'vehicle_tracks': len([t for t in confirmed_tracks if t.detection['class_id'] in [2, 3, 5, 7]]),
            'pedestrian_tracks': len([t for t in confirmed_tracks if t.detection['class_id'] == 0]),
            'average_track_age': np.mean([t.age for t in confirmed_tracks]) if confirmed_tracks else 0
        }
        
        return stats

class TrafficTrack:
    """
    Individual track for a detected object
    """
    
    def __init__(self, detection: Dict, track_id: int):
        """
        Initialize a new track
        
        Args:
            detection: Initial detection
            track_id: Unique track identifier
        """
        self.track_id = track_id
        self.detection = detection
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        
        # Initialize Kalman filter for position tracking
        self.kf = self._init_kalman_filter(detection['bbox'])
        
        # Track history for speed calculation
        self.position_history = [detection['center']]
        self.time_history = [0]  # Frame numbers
        
    def _init_kalman_filter(self, bbox: List[int]) -> KalmanFilter:
        """Initialize Kalman filter for tracking"""
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State vector: [x, y, w, h, vx, vy, vw, vh]
        # x, y: center coordinates
        # w, h: width and height
        # vx, vy, vw, vh: velocities
        
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0]).reshape((8, 1))
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe position and size)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        kf.Q = np.eye(8) * 0.1
        
        # Measurement noise covariance
        kf.R = np.eye(4) * 10
        
        # Initial covariance
        kf.P = np.eye(8) * 100
        
        return kf
    
    def predict(self):
        """Predict next state using Kalman filter"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
    
    def update(self, detection: Dict):
        """Update track with new detection"""
        self.detection = detection
        self.hits += 1
        self.time_since_update = 0
        
        # Update Kalman filter
        bbox = detection['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        measurement = np.array([cx, cy, w, h]).reshape((4, 1))
        self.kf.update(measurement)
        
        # Update position history
        self.position_history.append(detection['center'])
        self.time_history.append(self.age)
        
        # Keep only recent history (last 30 frames)
        if len(self.position_history) > 30:
            self.position_history = self.position_history[-30:]
            self.time_history = self.time_history[-30:]
    
    def mark_missed(self):
        """Mark track as missed (no detection in current frame)"""
        self.time_since_update += 1
    
    def is_dead(self, max_age: int) -> bool:
        """Check if track should be deleted"""
        return self.time_since_update > max_age
    
    def is_confirmed(self, min_hits: int) -> bool:
        """Check if track is confirmed (enough hits)"""
        return self.hits >= min_hits
    
    def get_predicted_bbox(self) -> List[int]:
        """Get predicted bounding box from Kalman filter"""
        state = self.kf.x
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        
        x1 = int(cx - w/2)
        y1 = int(cy - h/2)
        x2 = int(cx + w/2)
        y2 = int(cy + h/2)
        
        return [x1, y1, x2, y2]
    
    def get_state(self) -> Dict:
        """Get current track state"""
        bbox = self.get_predicted_bbox()
        center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)]
        
        state = {
            'bbox': bbox,
            'center': center,
            'confidence': self.detection['confidence'],
            'class_id': self.detection['class_id'],
            'class_name': self.detection['class_name'],
            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        }
        
        return state
    
    def calculate_speed(self, fps: float = 30, pixel_per_meter: float = 10) -> float:
        """
        Calculate object speed based on position history
        
        Args:
            fps: Video frame rate
            pixel_per_meter: Conversion factor from pixels to meters
            
        Returns:
            Speed in km/h
        """
        if len(self.position_history) < 2:
            return 0.0
        
        # Calculate distance traveled
        pos1 = self.position_history[-2]
        pos2 = self.position_history[-1]
        
        pixel_distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        meter_distance = pixel_distance / pixel_per_meter
        
        # Calculate time difference
        time_diff = (self.time_history[-1] - self.time_history[-2]) / fps
        
        if time_diff <= 0:
            return 0.0
        
        # Speed in m/s, convert to km/h
        speed_ms = meter_distance / time_diff
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def get_trajectory(self) -> List[Tuple[int, int]]:
        """Get trajectory points for visualization"""
        return self.position_history.copy()

if __name__ == "__main__":
    # Example usage with dummy detections
    tracker = TrafficTracker()
    
    # Simulate some detections
    frame1_detections = [
        {'bbox': [100, 100, 150, 150], 'confidence': 0.9, 'class_id': 2, 'class_name': 'car', 'center': [125, 125], 'area': 2500},
        {'bbox': [200, 200, 230, 250], 'confidence': 0.8, 'class_id': 0, 'class_name': 'person', 'center': [215, 225], 'area': 1500}
    ]
    
    frame2_detections = [
        {'bbox': [105, 105, 155, 155], 'confidence': 0.9, 'class_id': 2, 'class_name': 'car', 'center': [130, 130], 'area': 2500},
        {'bbox': [205, 205, 235, 255], 'confidence': 0.8, 'class_id': 0, 'class_name': 'person', 'center': [220, 230], 'area': 1500}
    ]
    
    # Update tracker
    tracks1 = tracker.update(frame1_detections)
    tracks2 = tracker.update(frame2_detections)
    
    print(f"Frame 1 tracks: {len(tracks1)}")
    print(f"Frame 2 tracks: {len(tracks2)}")
    print("Track statistics:", tracker.get_track_statistics())
