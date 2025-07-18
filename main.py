"""
Main Traffic Analysis Application
Entry point for the traffic management system using YOLO
"""

import argparse
import cv2
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.yolo_detector import TrafficDetector
from src.models.traffic_tracker import TrafficTracker
from src.models.flow_analyzer import TrafficFlowAnalyzer
from src.utils.video_processor import VideoProcessor
from src.utils.visualization import TrafficVisualizer

class TrafficAnalysisSystem:
    """
    Complete traffic analysis system integrating detection, tracking, and flow analysis
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the traffic analysis system"""
        print("Initializing Traffic Analysis System...")
        
        # Initialize components
        self.detector = TrafficDetector(config_path)
        self.tracker = TrafficTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        
        # Initialize flow analyzer with default counting lines
        default_lines = [
            [[100, 300], [500, 300]],  # Horizontal counting line
            [[300, 100], [300, 500]]   # Vertical counting line
        ]
        self.flow_analyzer = TrafficFlowAnalyzer(counting_lines=default_lines)
        self.visualizer = TrafficVisualizer()
        
        # Statistics
        self.total_frames_processed = 0
        self.start_time = time.time()
        
        print("✓ Traffic Analysis System initialized successfully!")
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """
        Process a video file for traffic analysis
        
        Args:
            video_path: Path to input video
            output_path: Path to save processed video
            display: Whether to display real-time results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        # Processing loop
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, stats = self.process_frame(frame)
                
                # Display statistics
                if frame_count % 30 == 0:  # Every second
                    self._print_frame_stats(frame_count, total_frames, stats)
                
                # Save frame
                if out:
                    out.write(processed_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Traffic Analysis', processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Processing stopped by user")
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)  # Pause
                
                frame_count += 1
                self.total_frames_processed += 1
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Generate final report
        self._generate_final_report()
    
    def process_camera(self, camera_id: int = 0, display: bool = True):
        """
        Process live camera feed
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            display: Whether to display real-time results
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        print(f"Starting live camera analysis (Camera ID: {camera_id})")
        print("Press 'q' to quit, 'space' to pause, 'r' to reset counters")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame from camera")
                    break
                
                # Process frame
                processed_frame, stats = self.process_frame(frame)
                
                # Display frame
                if display:
                    cv2.imshow('Live Traffic Analysis', processed_frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == ord('q'):
                        break
                    elif key == ord(' '):
                        cv2.waitKey(0)  # Pause
                    elif key == ord('r'):
                        self._reset_counters()
                        print("Counters reset")
                
                self.total_frames_processed += 1
        
        except KeyboardInterrupt:
            print("\nLive analysis stopped")
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
    
    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, statistics)
        """
        # Step 1: Object detection
        detections = self.detector.detect_frame(frame)
        
        # Step 2: Object tracking
        tracks = self.tracker.update(detections)
        
        # Step 3: Flow analysis
        flow_stats = self.flow_analyzer.analyze_frame(tracks, frame.shape[:2])
        
        # Step 4: Visualization
        processed_frame = self.visualizer.draw_complete_analysis(
            frame, detections, tracks, flow_stats, self.flow_analyzer.counting_lines
        )
        
        # Compile statistics
        stats = {
            'detections': len(detections),
            'tracks': len(tracks),
            'flow_stats': flow_stats,
            'tracker_stats': self.tracker.get_track_statistics()
        }
        
        return processed_frame, stats
    
    def _print_frame_stats(self, frame_count: int, total_frames: int, stats: dict):
        """Print frame processing statistics"""
        progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
        
        print(f"Frame {frame_count:6d} | "
              f"Progress: {progress:5.1f}% | "
              f"Detections: {stats['detections']:2d} | "
              f"Tracks: {stats['tracks']:2d} | "
              f"Vehicles: {stats['flow_stats']['counts']['total_vehicles']:2d}")
    
    def _reset_counters(self):
        """Reset all counters and statistics"""
        self.tracker = TrafficTracker(max_age=30, min_hits=3, iou_threshold=0.3)
        self.flow_analyzer = TrafficFlowAnalyzer(counting_lines=self.flow_analyzer.counting_lines)
        self.total_frames_processed = 0
        self.start_time = time.time()
    
    def _generate_final_report(self):
        """Generate and print final analysis report"""
        processing_time = time.time() - self.start_time
        fps_average = self.total_frames_processed / processing_time if processing_time > 0 else 0
        
        print("\n" + "="*60)
        print("TRAFFIC ANALYSIS REPORT")
        print("="*60)
        print(f"Total frames processed: {self.total_frames_processed}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {fps_average:.2f}")
        
        # Flow analysis report
        flow_report = self.flow_analyzer.generate_flow_report()
        print(f"\nFlow Analysis:")
        print(f"  Total vehicles counted: {flow_report['summary']['total_vehicles_counted']}")
        print(f"  Average vehicles/hour: {flow_report['summary']['average_vehicles_per_hour']:.1f}")
        
        if flow_report['summary']['busiest_line']['line_id'] is not None:
            busiest = flow_report['summary']['busiest_line']
            print(f"  Busiest line: Line {busiest['line_id']} ({busiest['count']} vehicles)")
        
        # Line statistics
        print(f"\nCounting Line Statistics:")
        for line_id, stats in flow_report['line_statistics'].items():
            print(f"  Line {line_id}: {stats['total_crossings']} crossings")
            print(f"    Direction 1: {stats['direction_1_count']}")
            print(f"    Direction 2: {stats['direction_2_count']}")
            print(f"    Net flow: {stats['net_flow']}")
        
        print("="*60)

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Traffic Analysis System using YOLO")
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source: file path or camera ID (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (optional)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = TrafficAnalysisSystem(args.config)
        
        # Determine source type
        if args.source.isdigit():
            # Camera source
            camera_id = int(args.source)
            system.process_camera(camera_id, display=not args.no_display)
        else:
            # Video file source
            if not os.path.exists(args.source):
                print(f"Error: Video file not found: {args.source}")
                return
            
            system.process_video(args.source, args.output, display=not args.no_display)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
