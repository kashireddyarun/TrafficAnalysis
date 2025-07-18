"""
Streamlit Web Dashboard for Traffic Analysis
Real-time traffic monitoring and analytics dashboard
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import sys
from typing import List, Dict
import threading
import queue

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.yolo_detector import TrafficDetector
from src.models.traffic_tracker import TrafficTracker
from src.models.flow_analyzer import TrafficFlowAnalyzer
from src.utils.visualization import TrafficVisualizer

class TrafficDashboard:
    """
    Streamlit-based traffic analysis dashboard
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        self.detector = None
        self.tracker = None
        self.flow_analyzer = None
        self.visualizer = TrafficVisualizer()
        
        # Data storage
        self.traffic_data = []
        self.is_processing = False
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Initialize session state
        if 'traffic_stats' not in st.session_state:
            st.session_state.traffic_stats = []
        if 'total_vehicles' not in st.session_state:
            st.session_state.total_vehicles = 0
        if 'total_pedestrians' not in st.session_state:
            st.session_state.total_pedestrians = 0

def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Traffic Analysis Dashboard",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize dashboard
    dashboard = TrafficDashboard()
    
    # Main title
    st.title("🚗 Traffic Analysis & Management System")
    st.markdown("Real-time traffic monitoring using YOLO object detection")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🔧 Controls")
        
        # System initialization
        if st.button("🚀 Initialize System"):
            initialize_system(dashboard)
        
        # Input source selection
        st.subheader("📹 Input Source")
        input_type = st.radio(
            "Select input type:",
            ["Webcam", "Video File", "IP Camera"]
        )
        
        if input_type == "Webcam":
            camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0)
            source = camera_id
        elif input_type == "Video File":
            uploaded_file = st.file_uploader("Choose video file", type=['mp4', 'avi', 'mov'])
            source = uploaded_file
        else:
            ip_url = st.text_input("IP Camera URL", placeholder="http://192.168.1.100:8080/video")
            source = ip_url
        
        # Processing controls
        st.subheader("⚙️ Processing Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, max_value=1.0, value=0.5, step=0.1
        )
        
        show_trajectories = st.checkbox("Show Trajectories", value=True)
        show_counting_lines = st.checkbox("Show Counting Lines", value=True)
        show_statistics = st.checkbox("Show Statistics Panel", value=True)
        
        # Counting lines configuration
        st.subheader("📊 Counting Lines")
        num_lines = st.number_input("Number of counting lines", min_value=0, max_value=5, value=2)
        
        counting_lines = []
        for i in range(num_lines):
            st.write(f"Line {i+1}:")
            col1, col2 = st.columns(2)
            with col1:
                x1 = st.number_input(f"X1_{i}", value=100, key=f"x1_{i}")
                y1 = st.number_input(f"Y1_{i}", value=300 + i*100, key=f"y1_{i}")
            with col2:
                x2 = st.number_input(f"X2_{i}", value=500, key=f"x2_{i}")
                y2 = st.number_input(f"Y2_{i}", value=300 + i*100, key=f"y2_{i}")
            counting_lines.append([[x1, y1], [x2, y2]])
        
        # Start/Stop processing
        col1, col2 = st.columns(2)
        with col1:
            start_processing = st.button("▶️ Start", key="start_btn")
        with col2:
            stop_processing = st.button("⏹️ Stop", key="stop_btn")
        
        if start_processing and dashboard.detector is not None:
            dashboard.flow_analyzer = TrafficFlowAnalyzer(counting_lines=counting_lines)
            dashboard.is_processing = True
            st.success("Processing started!")
        
        if stop_processing:
            dashboard.is_processing = False
            st.success("Processing stopped!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎥 Live Video Feed")
        video_placeholder = st.empty()
        
        # Process video if system is running
        if dashboard.is_processing and source is not None:
            process_video_stream(dashboard, source, video_placeholder, 
                               show_trajectories, show_counting_lines, show_statistics)
    
    with col2:
        st.subheader("📈 Real-time Statistics")
        
        # Current statistics
        stats_container = st.container()
        with stats_container:
            if st.session_state.traffic_stats:
                latest_stats = st.session_state.traffic_stats[-1]
                
                # Key metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("🚗 Vehicles", latest_stats.get('vehicle_count', 0))
                    st.metric("🚶 Pedestrians", latest_stats.get('pedestrian_count', 0))
                with col2:
                    st.metric("📊 Density", f"{latest_stats.get('density', 0):.3f}")
                    st.metric("🚦 Congestion", latest_stats.get('congestion', 'Unknown'))
                
                # Traffic flow chart
                create_flow_chart()
                
                # Vehicle type distribution
                create_vehicle_distribution_chart()
            else:
                st.info("Start processing to see statistics")
    
    # Bottom section - Historical data and analytics
    st.subheader("📊 Analytics Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Traffic Flow", "🚗 Vehicle Types", "📍 Heatmap", "📋 Reports"])
    
    with tab1:
        create_traffic_flow_analytics()
    
    with tab2:
        create_vehicle_type_analytics()
    
    with tab3:
        create_heatmap_visualization()
    
    with tab4:
        create_reports_section()

def initialize_system(dashboard):
    """Initialize the traffic analysis system"""
    try:
        with st.spinner("Initializing traffic analysis system..."):
            dashboard.detector = TrafficDetector()
            dashboard.tracker = TrafficTracker()
            dashboard.flow_analyzer = TrafficFlowAnalyzer()
        st.success("✅ System initialized successfully!")
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")

def process_video_stream(dashboard, source, video_placeholder, 
                        show_trajectories, show_counting_lines, show_statistics):
    """Process video stream and display results"""
    if isinstance(source, str) and source.startswith('http'):
        # IP Camera
        cap = cv2.VideoCapture(source)
    elif isinstance(source, int):
        # Webcam
        cap = cv2.VideoCapture(source)
    else:
        # Uploaded file
        if source is not None:
            # Save uploaded file temporarily
            temp_file = f"temp_video_{int(time.time())}.mp4"
            with open(temp_file, "wb") as f:
                f.write(source.read())
            cap = cv2.VideoCapture(temp_file)
        else:
            st.error("No video source selected")
            return
    
    if not cap.isOpened():
        st.error("Failed to open video source")
        return
    
    frame_count = 0
    while dashboard.is_processing:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, stats = process_single_frame(dashboard, frame)
        
        # Update session state
        update_session_stats(stats)
        
        # Display frame
        video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        
        frame_count += 1
        time.sleep(0.033)  # ~30 FPS
    
    cap.release()
    
    # Clean up temporary file if exists
    if 'temp_file' in locals() and os.path.exists(temp_file):
        os.remove(temp_file)

def process_single_frame(dashboard, frame):
    """Process a single frame through the pipeline"""
    # Object detection
    detections = dashboard.detector.detect_frame(frame)
    
    # Object tracking
    tracks = dashboard.tracker.update(detections)
    
    # Flow analysis
    flow_stats = dashboard.flow_analyzer.analyze_frame(tracks, frame.shape[:2])
    
    # Visualization
    processed_frame = dashboard.visualizer.draw_complete_analysis(
        frame, detections, tracks, flow_stats, dashboard.flow_analyzer.counting_lines
    )
    
    # Compile statistics
    stats = {
        'timestamp': datetime.now(),
        'vehicle_count': flow_stats['counts']['total_vehicles'],
        'pedestrian_count': flow_stats['counts']['total_pedestrians'],
        'density': flow_stats['density']['vehicle_density'],
        'congestion': flow_stats['density']['congestion_level'],
        'active_tracks': len(tracks),
        'detections': len(detections),
        'flow_rate': flow_stats['flow']['flow_rate']
    }
    
    return processed_frame, stats

def update_session_stats(stats):
    """Update session state with new statistics"""
    st.session_state.traffic_stats.append(stats)
    
    # Keep only last 100 entries to prevent memory issues
    if len(st.session_state.traffic_stats) > 100:
        st.session_state.traffic_stats = st.session_state.traffic_stats[-100:]
    
    # Update totals
    st.session_state.total_vehicles += stats['vehicle_count']
    st.session_state.total_pedestrians += stats['pedestrian_count']

def create_flow_chart():
    """Create real-time traffic flow chart"""
    if len(st.session_state.traffic_stats) > 1:
        df = pd.DataFrame(st.session_state.traffic_stats[-20:])  # Last 20 data points
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['vehicle_count'],
            mode='lines+markers',
            name='Vehicles',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['pedestrian_count'],
            mode='lines+markers',
            name='Pedestrians',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title="Real-time Traffic Flow",
            xaxis_title="Time",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_vehicle_distribution_chart():
    """Create vehicle type distribution chart"""
    if st.session_state.traffic_stats:
        # Mock data for vehicle types (in real implementation, this would come from flow_stats)
        vehicle_types = {'Car': 60, 'Truck': 20, 'Motorcycle': 15, 'Bus': 5}
        
        fig = go.Figure(data=[go.Pie(
            labels=list(vehicle_types.keys()),
            values=list(vehicle_types.values()),
            hole=0.3
        )])
        
        fig.update_layout(
            title="Vehicle Type Distribution",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_traffic_flow_analytics():
    """Create traffic flow analytics section"""
    if st.session_state.traffic_stats:
        df = pd.DataFrame(st.session_state.traffic_stats)
        
        # Time series analysis
        fig = px.line(df, x='timestamp', y=['vehicle_count', 'pedestrian_count'],
                     title="Traffic Count Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Density analysis
        fig2 = px.line(df, x='timestamp', y='density',
                      title="Traffic Density Over Time")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Statistics summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Vehicles", f"{df['vehicle_count'].mean():.1f}")
        with col2:
            st.metric("Peak Vehicles", df['vehicle_count'].max())
        with col3:
            st.metric("Average Density", f"{df['density'].mean():.3f}")
    else:
        st.info("No data available. Start processing to collect analytics.")

def create_vehicle_type_analytics():
    """Create vehicle type analytics section"""
    st.write("Vehicle type analytics would be displayed here")
    st.info("This section will show detailed vehicle classification statistics")

def create_heatmap_visualization():
    """Create traffic heatmap visualization"""
    st.write("Traffic density heatmap would be displayed here")
    st.info("This section will show spatial traffic distribution")

def create_reports_section():
    """Create reports section"""
    st.subheader("📋 Traffic Analysis Reports")
    
    if st.button("Generate Current Report"):
        if st.session_state.traffic_stats:
            # Generate report
            df = pd.DataFrame(st.session_state.traffic_stats)
            
            report = f"""
            # Traffic Analysis Report
            
            **Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            ## Summary Statistics
            - **Total Frames Analyzed:** {len(df)}
            - **Average Vehicles per Frame:** {df['vehicle_count'].mean():.1f}
            - **Average Pedestrians per Frame:** {df['pedestrian_count'].mean():.1f}
            - **Peak Vehicle Count:** {df['vehicle_count'].max()}
            - **Average Traffic Density:** {df['density'].mean():.3f}
            
            ## Traffic Patterns
            - **Busiest Period:** {df.loc[df['vehicle_count'].idxmax(), 'timestamp']}
            - **Congestion Analysis:** Most common level: {df['congestion'].mode().iloc[0] if not df.empty else 'N/A'}
            
            ## Recommendations
            Based on the analysis, consider:
            1. Traffic signal optimization during peak hours
            2. Additional monitoring for high-density periods
            3. Pedestrian safety measures in busy areas
            """
            
            st.markdown(report)
            
            # Download button for report
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        else:
            st.warning("No data available for report generation")
    
    # Export data
    if st.button("📊 Export Raw Data"):
        if st.session_state.traffic_stats:
            df = pd.DataFrame(st.session_state.traffic_stats)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
