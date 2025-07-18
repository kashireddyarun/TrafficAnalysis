# Traffic Analysis ML Model using YOLO

A comprehensive traffic management system using YOLO (You Only Look Once) for real-time traffic analysis and management.

## Features

- **Object Detection**: Detect vehicles, pedestrians, traffic lights, and road signs
- **Traffic Flow Analysis**: Count vehicles, estimate speed, and analyze traffic patterns
- **Congestion Detection**: Real-time traffic density analysis
- **Traffic Light Management**: Intelligent signal timing optimization
- **Multi-lane Monitoring**: Support for multiple lanes and intersections
- **Real-time Processing**: Live video feed analysis
- **Analytics Dashboard**: Comprehensive traffic statistics and visualization

## Project Structure

```
DeepfakeModel/
├── src/
│   ├── models/
│   │   ├── yolo_detector.py       # YOLO model implementation
│   │   ├── traffic_tracker.py     # Vehicle tracking system
│   │   └── flow_analyzer.py       # Traffic flow analysis
│   ├── utils/
│   │   ├── video_processor.py     # Video processing utilities
│   │   ├── visualization.py       # Visualization tools
│   │   └── config.py              # Configuration management
│   ├── traffic_system/
│   │   ├── congestion_detector.py # Traffic congestion analysis
│   │   ├── signal_optimizer.py    # Traffic light optimization
│   │   └── analytics.py           # Traffic analytics
│   └── app/
│       ├── streamlit_app.py       # Web dashboard
│       └── real_time_processor.py # Real-time processing
├── data/
│   ├── sample_videos/             # Sample traffic videos
│   └── models/                    # Pre-trained model weights
├── config/
│   └── config.yaml                # Configuration file
├── notebooks/
│   └── traffic_analysis_demo.ipynb # Demo notebook
├── tests/
│   └── test_traffic_system.py     # Unit tests
├── requirements.txt
├── README.md
└── main.py                        # Main application entry point
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Traffic Detection

```python
from src.models.yolo_detector import TrafficDetector

detector = TrafficDetector()
results = detector.detect_traffic("path/to/video.mp4")
```

### Real-time Analysis

```python
python main.py --source 0  # Use webcam
python main.py --source "traffic_video.mp4"  # Use video file
```

### Web Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

## Configuration

Edit `config/config.yaml` to customize:

- Detection thresholds
- Tracking parameters
- Analytics settings
- Output preferences

## Model Performance

- **Detection Speed**: ~30-60 FPS on GPU
- **Accuracy**: >95% for vehicle detection
- **Supported Objects**: Cars, trucks, buses, motorcycles, bicycles, pedestrians
- **Multi-class Support**: Traffic lights, road signs, lane markings

## License

MIT License
