"""
YOLO-based Traffic Detection System
Handles object detection for traffic analysis using YOLOv8
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import yaml
import os

class TrafficDetector:
    """
    Main traffic detection class using YOLO model
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the traffic detector
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.model = self._load_model()
        self.device = self._get_device()
        
        # Class mappings for COCO dataset
        self.class_names = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench'
        }
        
        # Traffic-relevant classes
        self.vehicle_classes = self.config['classes']['vehicles']
        self.pedestrian_classes = self.config['classes']['pedestrians']
        self.traffic_light_classes = self.config['classes']['traffic_lights']
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Default configuration if config file is not found"""
        return {
            'model': {
                'name': 'yolov8n',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.5,
                'device': 'auto'
            },
            'classes': {
                'vehicles': [2, 3, 5, 7],
                'pedestrians': [0],
                'traffic_lights': [9]
            }
        }
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        model_name = self.config['model']['name']
        print(f"Loading YOLO model: {model_name}")
        return YOLO(f"{model_name}.pt")
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        device_config = self.config['model']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device_config
    
    def detect_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            List of detections with bounding boxes, confidence, and class
        """
        results = self.model(
            frame,
            conf=self.config['model']['confidence_threshold'],
            iou=self.config['model']['iou_threshold'],
            device=self.device
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for traffic-relevant classes
                    if self._is_traffic_relevant(class_id):
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, 'unknown'),
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        detections.append(detection)
        
        return detections
    
    def _is_traffic_relevant(self, class_id: int) -> bool:
        """Check if detected class is relevant for traffic analysis"""
        return (class_id in self.vehicle_classes or 
                class_id in self.pedestrian_classes or 
                class_id in self.traffic_light_classes)
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> List[List[Dict]]:
        """
        Process entire video for traffic detection
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save processed video
            
        Returns:
            List of detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects in frame
            detections = self.detect_frame(frame)
            all_detections.append(detections)
            
            # Draw detections on frame
            annotated_frame = self.draw_detections(frame, detections)
            
            if out:
                out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if out:
            out.release()
        
        print(f"Video processing complete. Total frames: {frame_count}")
        return all_detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
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
            
            # Choose color based on object type
            if detection['class_id'] in self.vehicle_classes:
                color = (0, 255, 0)  # Green for vehicles
            elif detection['class_id'] in self.pedestrian_classes:
                color = (255, 0, 0)  # Blue for pedestrians
            elif detection['class_id'] in self.traffic_light_classes:
                color = (0, 0, 255)  # Red for traffic lights
            else:
                color = (255, 255, 0)  # Cyan for others
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(annotated_frame,
                         (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            
            cv2.putText(annotated_frame, label,
                       (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (255, 255, 255), 2)
        
        return annotated_frame
    
    def get_vehicle_count(self, detections: List[Dict]) -> int:
        """Count vehicles in detections"""
        return sum(1 for det in detections 
                  if det['class_id'] in self.vehicle_classes)
    
    def get_pedestrian_count(self, detections: List[Dict]) -> int:
        """Count pedestrians in detections"""
        return sum(1 for det in detections 
                  if det['class_id'] in self.pedestrian_classes)
    
    def analyze_traffic_density(self, detections: List[Dict], frame_area: int) -> float:
        """
        Calculate traffic density as percentage of frame covered by vehicles
        
        Args:
            detections: List of detections
            frame_area: Total frame area in pixels
            
        Returns:
            Traffic density (0.0 to 1.0)
        """
        total_vehicle_area = sum(det['area'] for det in detections 
                               if det['class_id'] in self.vehicle_classes)
        return min(total_vehicle_area / frame_area, 1.0)

if __name__ == "__main__":
    # Example usage
    detector = TrafficDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect_frame(frame)
        annotated_frame = detector.draw_detections(frame, detections)
        
        # Display counts
        vehicle_count = detector.get_vehicle_count(detections)
        pedestrian_count = detector.get_pedestrian_count(detections)
        
        cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Pedestrians: {pedestrian_count}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imshow('Traffic Detection', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
