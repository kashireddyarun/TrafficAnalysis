"""
Real-time Traffic Processing Application
Integrated application for real-time traffic analysis and management
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
from datetime import datetime

# Import traffic system components
from src.models.yolo_detector import TrafficDetector
from src.models.traffic_tracker import TrafficTracker
from src.models.flow_analyzer import TrafficFlowAnalyzer
from src.traffic_system.congestion_detector import CongestionDetector
from src.traffic_system.signal_optimizer import SignalOptimizer
from src.traffic_system.analytics import TrafficAnalytics
from src.utils.config import load_config
from src.utils.visualization import TrafficVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeTrafficProcessor:
    """Real-time traffic processing and management system"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the real-time traffic processor"""
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.detector = TrafficDetector(self.config)
        self.tracker = TrafficTracker(self.config)
        self.flow_analyzer = TrafficFlowAnalyzer(self.config)
        self.congestion_detector = CongestionDetector(self.config)
        self.signal_optimizer = SignalOptimizer(self.config)
        self.analytics = TrafficAnalytics(self.config)
        self.visualizer = TrafficVisualizer(self.config)
        
        # Processing parameters
        self.fps = self.config.get('processing', {}).get('target_fps', 30)
        self.frame_skip = self.config.get('processing', {}).get('frame_skip', 1)
        self.output_enabled = self.config.get('output', {}).get('enabled', True)
        
        # Performance monitoring
        self.frame_count = 0
        self.processing_times = []
        self.last_analytics_update = time.time()
        self.analytics_interval = 300  # 5 minutes
        
        # Status tracking
        self.is_running = False
        self.current_metrics = {}
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Process a single frame through the complete pipeline"""
        
        start_time = time.time()
        
        # 1. Object Detection
        detections = self.detector.detect_frame(frame)
        
        # 2. Object Tracking
        tracks = self.tracker.update_tracks(detections)
        
        # 3. Traffic Flow Analysis
        flow_data = self.flow_analyzer.analyze_frame(tracks, detections)
        
        # 4. Congestion Detection
        congestion_metrics = self.congestion_detector.analyze_congestion(
            detections, tracks, frame.shape[0] * frame.shape[1]
        )
        
        # 5. Signal Optimization (if enabled)
        if self.config.get('traffic_management', {}).get('signal_optimization', {}).get('enabled', False):
            signal_timing = self.signal_optimizer.optimize_signals(detections, tracks)
        else:
            signal_timing = {}
            
        # 6. Data Logging for Analytics
        if time.time() - self.last_analytics_update > self.analytics_interval:
            self._update_analytics(detections, flow_data, congestion_metrics)
            self.last_analytics_update = time.time()
            
        # 7. Visualization
        processed_frame = self.visualizer.draw_complete_analysis(
            frame, detections, tracks, flow_data, congestion_metrics
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            
        # Compile metrics
        frame_metrics = {
            'detections': detections,
            'tracks': tracks,
            'flow_data': flow_data,
            'congestion_metrics': congestion_metrics,
            'signal_timing': signal_timing,
            'processing_time': processing_time,
            'fps': 1.0 / processing_time if processing_time > 0 else 0,
            'frame_number': self.frame_count
        }
        
        self.current_metrics = frame_metrics
        self.frame_count += 1
        
        return processed_frame, frame_metrics
        
    def _update_analytics(self, detections: List, flow_data: Dict, 
                         congestion_metrics) -> None:
        """Update analytics database with current data"""
        
        try:
            # Log detections
            for detection in detections:
                self.analytics.log_vehicle_detection(detection)
                
            # Log flow data
            flow_summary = {
                'vehicle_count': len(detections),
                'average_speed': np.mean([t.get('speed', 0) for t in flow_data.get('tracks', [])]),
                'peak_speed': max([t.get('speed', 0) for t in flow_data.get('tracks', [])], default=0),
                'congestion_level': congestion_metrics.congestion_level,
                'flow_rate': flow_data.get('flow_rate', 0)
            }
            self.analytics.log_traffic_flow(flow_summary)
            
        except Exception as e:
            logger.error(f"Error updating analytics: {e}")
            
    def process_video_stream(self, source: str, output_path: Optional[str] = None) -> None:
        """Process video stream (file, camera, or network stream)"""
        
        logger.info(f"Starting video processing from source: {source}")
        
        # Open video source
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
            
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
            
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps} FPS")
        
        # Setup video writer if output is enabled
        video_writer = None
        if output_path and self.output_enabled:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        self.is_running = True
        frame_time = 1.0 / self.fps
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames if needed
                if self.frame_count % self.frame_skip != 0:
                    self.frame_count += 1
                    continue
                    
                # Process frame
                processed_frame, metrics = self.process_frame(frame)
                
                # Write output video
                if video_writer:
                    video_writer.write(processed_frame)
                    
                # Display frame (optional)
                if self.config.get('display', {}).get('enabled', False):
                    cv2.imshow('Traffic Analysis', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                # Performance monitoring
                if self.frame_count % 100 == 0:
                    avg_processing_time = np.mean(self.processing_times[-50:])
                    current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    logger.info(f"Frame {self.frame_count}: {current_fps:.1f} FPS, "
                              f"Processing time: {avg_processing_time*1000:.1f}ms")
                    
                # Frame rate control
                time.sleep(max(0, frame_time - metrics['processing_time']))
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
        finally:
            self.stop_processing()
            cap.release()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
    def process_batch_videos(self, video_paths: List[str], output_dir: str) -> None:
        """Process multiple videos in batch mode"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
            
            # Generate output path
            video_name = Path(video_path).stem
            output_file = output_path / f"{video_name}_processed.mp4"
            
            try:
                self.process_video_stream(video_path, str(output_file))
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                continue
                
            # Reset counters for next video
            self.frame_count = 0
            self.processing_times.clear()
            
    def get_real_time_status(self) -> Dict:
        """Get current real-time processing status"""
        
        if not self.processing_times:
            return {'status': 'not_running'}
            
        avg_processing_time = np.mean(self.processing_times[-10:])
        current_fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        status = {
            'status': 'running' if self.is_running else 'stopped',
            'frame_count': self.frame_count,
            'current_fps': current_fps,
            'average_processing_time': avg_processing_time * 1000,  # in ms
            'metrics': self.current_metrics,
            'uptime': time.time() - (getattr(self, 'start_time', time.time()))
        }
        
        return status
        
    def generate_live_analytics(self) -> Dict:
        """Generate live analytics dashboard data"""
        
        dashboard_data = self.analytics.get_performance_dashboard_data()
        
        # Add real-time processing metrics
        if self.current_metrics:
            dashboard_data.update({
                'real_time_fps': self.current_metrics.get('fps', 0),
                'active_tracks': len(self.current_metrics.get('tracks', [])),
                'current_detections': len(self.current_metrics.get('detections', [])),
                'processing_latency': self.current_metrics.get('processing_time', 0) * 1000
            })
            
        return dashboard_data
        
    def stop_processing(self) -> None:
        """Stop the real-time processing"""
        self.is_running = False
        logger.info("Processing stopped")
        
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.stop_processing()
        # Additional cleanup if needed
        
def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='Real-time Traffic Analysis System')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (file path, camera index, or stream URL)')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Configuration file path')
    parser.add_argument('--batch', nargs='+', help='Process multiple videos in batch mode')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for batch processing')
    parser.add_argument('--analytics', action='store_true',
                       help='Generate analytics report after processing')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RealTimeTrafficProcessor(args.config)
    
    try:
        if args.batch:
            # Batch processing mode
            processor.process_batch_videos(args.batch, args.output_dir)
        else:
            # Single video processing
            processor.process_video_stream(args.source, args.output)
            
        # Generate analytics report if requested
        if args.analytics:
            report = processor.analytics.generate_comprehensive_report('daily')
            report_json = processor.analytics.export_report(report, 'json')
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"traffic_report_{timestamp}.json"
            with open(report_path, 'w') as f:
                f.write(report_json)
            logger.info(f"Analytics report saved to: {report_path}")
            
    except KeyboardInterrupt:
        logger.info("Processing interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()
