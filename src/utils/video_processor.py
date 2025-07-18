"""
Video Processing Utilities
Handles video input/output, frame processing, and format conversions
"""

import cv2
import numpy as np
import os
from typing import Generator, Tuple, Optional, Union, List
from pathlib import Path
import imageio

class VideoProcessor:
    """
    Utility class for video processing operations
    """
    
    def __init__(self):
        """Initialize video processor"""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    def read_video_frames(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """
        Generator to read video frames one by one
        
        Args:
            video_path: Path to video file
            
        Yields:
            Video frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video properties and information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
        }
        
        cap.release()
        return info
    
    def resize_video(self, input_path: str, output_path: str, 
                    target_size: Tuple[int, int], maintain_aspect: bool = True):
        """
        Resize video to target dimensions
        
        Args:
            input_path: Input video path
            output_path: Output video path
            target_size: (width, height) target size
            maintain_aspect: Whether to maintain aspect ratio
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get original properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate new dimensions
        if maintain_aspect:
            aspect_ratio = original_width / original_height
            target_width, target_height = target_size
            
            if target_width / target_height > aspect_ratio:
                new_width = int(target_height * aspect_ratio)
                new_height = target_height
            else:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
        else:
            new_width, new_height = target_size
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, (new_width, new_height))
        
        print(f"Resizing video from {original_width}x{original_height} to {new_width}x{new_height}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame
                resized_frame = cv2.resize(frame, (new_width, new_height))
                out.write(resized_frame)
        
        finally:
            cap.release()
            out.release()
        
        print(f"Video resized and saved to: {output_path}")
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 1, max_frames: Optional[int] = None):
        """
        Extract frames from video as images
        
        Args:
            video_path: Input video path
            output_dir: Directory to save frames
            frame_interval: Extract every nth frame
            max_frames: Maximum number of frames to extract
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(output_dir, f"frame_{extracted_count:06d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    extracted_count += 1
                    
                    if max_frames and extracted_count >= max_frames:
                        break
                
                frame_count += 1
        
        finally:
            cap.release()
        
        print(f"Extracted {extracted_count} frames to {output_dir}")
    
    def create_video_from_frames(self, frames_dir: str, output_path: str, 
                                fps: float = 30, frame_pattern: str = "frame_*.jpg"):
        """
        Create video from a sequence of frames
        
        Args:
            frames_dir: Directory containing frames
            output_path: Output video path
            fps: Frames per second
            frame_pattern: Pattern to match frame files
        """
        frame_files = sorted(Path(frames_dir).glob(frame_pattern))
        
        if not frame_files:
            raise ValueError(f"No frames found in {frames_dir} with pattern {frame_pattern}")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width = first_frame.shape[:2]
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating video from {len(frame_files)} frames...")
        
        try:
            for frame_file in frame_files:
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    out.write(frame)
        
        finally:
            out.release()
        
        print(f"Video created: {output_path}")
    
    def crop_video(self, input_path: str, output_path: str, 
                   crop_box: Tuple[int, int, int, int]):
        """
        Crop video to specified region
        
        Args:
            input_path: Input video path
            output_path: Output video path
            crop_box: (x, y, width, height) crop region
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        x, y, w, h = crop_box
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        print(f"Cropping video to region ({x}, {y}, {w}, {h})")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Crop frame
                cropped_frame = frame[y:y+h, x:x+w]
                out.write(cropped_frame)
        
        finally:
            cap.release()
            out.release()
        
        print(f"Cropped video saved to: {output_path}")
    
    def convert_video_format(self, input_path: str, output_path: str, 
                           codec: str = 'mp4v', fps: Optional[float] = None):
        """
        Convert video to different format/codec
        
        Args:
            input_path: Input video path
            output_path: Output video path
            codec: Video codec (e.g., 'mp4v', 'XVID', 'H264')
            fps: Target FPS (None to keep original)
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get original properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        target_fps = fps or original_fps
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        print(f"Converting video format: {codec} at {target_fps} FPS")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
        
        finally:
            cap.release()
            out.release()
        
        print(f"Converted video saved to: {output_path}")
    
    def stabilize_video(self, input_path: str, output_path: str):
        """
        Simple video stabilization using optical flow
        
        Args:
            input_path: Input video path
            output_path: Output video path
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize transformation accumulator
        transforms = []
        
        print("Analyzing video for stabilization...")
        
        try:
            while True:
                ret, curr_frame = cap.read()
                if not ret:
                    break
                
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, 
                    cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, 
                                          qualityLevel=0.01, minDistance=30),
                    None
                )
                
                # Extract good points
                good_new = flow[0][flow[1].flatten() == 1]
                good_old = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, 
                                                 qualityLevel=0.01, minDistance=30)[flow[1].flatten() == 1]
                
                if len(good_new) > 10 and len(good_old) > 10:
                    # Estimate transformation
                    transform = cv2.estimateRigidTransform(good_old, good_new, False)
                    if transform is not None:
                        transforms.append(transform)
                    else:
                        transforms.append(np.eye(2, 3, dtype=np.float32))
                else:
                    transforms.append(np.eye(2, 3, dtype=np.float32))
                
                prev_gray = curr_gray.copy()
        
        finally:
            cap.release()
        
        # Apply smoothed transforms
        cap = cv2.VideoCapture(input_path)
        smoothed_transforms = self._smooth_transforms(transforms)
        
        print("Applying stabilization...")
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx < len(smoothed_transforms):
                    transform = smoothed_transforms[frame_idx]
                    stabilized_frame = cv2.warpAffine(frame, transform, (width, height))
                    out.write(stabilized_frame)
                else:
                    out.write(frame)
                
                frame_idx += 1
        
        finally:
            cap.release()
            out.release()
        
        print(f"Stabilized video saved to: {output_path}")
    
    def _smooth_transforms(self, transforms: List[np.ndarray], 
                          smoothing_radius: int = 30) -> List[np.ndarray]:
        """Apply smoothing to transformation matrices"""
        smoothed = []
        
        for i in range(len(transforms)):
            start_idx = max(0, i - smoothing_radius)
            end_idx = min(len(transforms), i + smoothing_radius + 1)
            
            window_transforms = transforms[start_idx:end_idx]
            avg_transform = np.mean(window_transforms, axis=0)
            smoothed.append(avg_transform)
        
        return smoothed
    
    def add_text_overlay(self, input_path: str, output_path: str, 
                        text: str, position: Tuple[int, int] = (10, 30),
                        font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255)):
        """
        Add text overlay to video
        
        Args:
            input_path: Input video path
            output_path: Output video path
            text: Text to overlay
            position: (x, y) position for text
            font_scale: Font size scale
            color: (B, G, R) text color
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Adding text overlay: '{text}'")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add text overlay
                cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, color, 2, cv2.LINE_AA)
                
                out.write(frame)
        
        finally:
            cap.release()
            out.release()
        
        print(f"Video with text overlay saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    processor = VideoProcessor()
    
    # Example: Get video info
    # info = processor.get_video_info("sample_video.mp4")
    # print("Video info:", info)
    
    print("VideoProcessor initialized successfully!")
    print("Available methods:")
    print("- read_video_frames()")
    print("- get_video_info()")
    print("- resize_video()")
    print("- extract_frames()")
    print("- create_video_from_frames()")
    print("- crop_video()")
    print("- convert_video_format()")
    print("- stabilize_video()")
    print("- add_text_overlay()")
