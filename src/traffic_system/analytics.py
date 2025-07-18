"""
Comprehensive Traffic Analytics System
Advanced analytics and reporting for traffic management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import sqlite3
import os

@dataclass
class AnalyticsReport:
    """Comprehensive analytics report structure"""
    timestamp: datetime
    period: str  # 'hourly', 'daily', 'weekly'
    vehicle_statistics: Dict
    traffic_flow: Dict
    congestion_analysis: Dict
    signal_performance: Dict
    environmental_impact: Dict
    recommendations: List[str]

class TrafficAnalytics:
    """Advanced traffic analytics and reporting system"""
    
    def __init__(self, config: Dict, db_path: str = "traffic_analytics.db"):
        self.config = config
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Data storage
        self.hourly_data = defaultdict(list)
        self.daily_summaries = deque(maxlen=30)  # Keep 30 days
        self.weekly_summaries = deque(maxlen=12)  # Keep 12 weeks
        
        # Analytics parameters
        self.analytics_config = config.get('analytics', {})
        self.reporting_interval = self.analytics_config.get('reporting_interval', 3600)  # 1 hour
        
        # Performance thresholds
        self.thresholds = {
            'congestion_density': 0.7,
            'low_speed_limit': 10,  # km/h
            'high_delay_threshold': 60,  # seconds
            'efficiency_threshold': 0.8
        }
        
    def _init_database(self):
        """Initialize SQLite database for analytics storage"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else '.', exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vehicle_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                vehicle_type TEXT,
                confidence REAL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                speed REAL,
                direction TEXT,
                lane TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                hour INTEGER,
                vehicle_count INTEGER,
                average_speed REAL,
                peak_speed REAL,
                congestion_level TEXT,
                flow_rate REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                intersection TEXT,
                cycle_time REAL,
                green_time_ns REAL,
                green_time_ew REAL,
                throughput INTEGER,
                average_delay REAL,
                level_of_service TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def log_vehicle_detection(self, detection: Dict):
        """Log individual vehicle detection to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO vehicle_detections 
            (timestamp, vehicle_type, confidence, x1, y1, x2, y2, speed, direction, lane)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            detection.get('class', 'unknown'),
            detection.get('confidence', 0.0),
            detection.get('bbox', [0, 0, 0, 0])[0],
            detection.get('bbox', [0, 0, 0, 0])[1],
            detection.get('bbox', [0, 0, 0, 0])[2],
            detection.get('bbox', [0, 0, 0, 0])[3],
            detection.get('speed', 0.0),
            detection.get('direction', 'unknown'),
            detection.get('lane', 'unknown')
        ))
        
        conn.commit()
        conn.close()
        
    def log_traffic_flow(self, flow_data: Dict):
        """Log traffic flow data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        cursor.execute('''
            INSERT INTO traffic_flow 
            (timestamp, hour, vehicle_count, average_speed, peak_speed, congestion_level, flow_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            now,
            now.hour,
            flow_data.get('vehicle_count', 0),
            flow_data.get('average_speed', 0.0),
            flow_data.get('peak_speed', 0.0),
            flow_data.get('congestion_level', 'unknown'),
            flow_data.get('flow_rate', 0.0)
        ))
        
        conn.commit()
        conn.close()
        
    def calculate_vehicle_statistics(self, time_period: str = 'hourly') -> Dict:
        """Calculate comprehensive vehicle statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Define time window
        if time_period == 'hourly':
            time_filter = datetime.now() - timedelta(hours=1)
        elif time_period == 'daily':
            time_filter = datetime.now() - timedelta(days=1)
        elif time_period == 'weekly':
            time_filter = datetime.now() - timedelta(weeks=1)
        else:
            time_filter = datetime.now() - timedelta(hours=1)
            
        # Query vehicle data
        query = '''
            SELECT vehicle_type, COUNT(*) as count, AVG(speed) as avg_speed, 
                   AVG(confidence) as avg_confidence
            FROM vehicle_detections 
            WHERE timestamp >= ?
            GROUP BY vehicle_type
        '''
        
        df = pd.read_sql_query(query, conn, params=[time_filter])
        conn.close()
        
        if df.empty:
            return {
                'total_vehicles': 0,
                'vehicle_types': {},
                'average_speed': 0.0,
                'detection_confidence': 0.0
            }
            
        # Calculate statistics
        total_vehicles = df['count'].sum()
        vehicle_types = dict(zip(df['vehicle_type'], df['count']))
        avg_speed = df['avg_speed'].mean()
        avg_confidence = df['avg_confidence'].mean()
        
        return {
            'total_vehicles': int(total_vehicles),
            'vehicle_types': vehicle_types,
            'average_speed': float(avg_speed),
            'detection_confidence': float(avg_confidence),
            'most_common_type': df.loc[df['count'].idxmax(), 'vehicle_type'] if not df.empty else 'none'
        }
        
    def analyze_traffic_patterns(self, time_period: str = 'daily') -> Dict:
        """Analyze traffic patterns and trends"""
        conn = sqlite3.connect(self.db_path)
        
        if time_period == 'daily':
            time_filter = datetime.now() - timedelta(days=1)
        elif time_period == 'weekly':
            time_filter = datetime.now() - timedelta(weeks=1)
        else:
            time_filter = datetime.now() - timedelta(days=1)
            
        # Hourly traffic flow
        query = '''
            SELECT hour, AVG(vehicle_count) as avg_count, AVG(average_speed) as avg_speed,
                   AVG(flow_rate) as avg_flow_rate
            FROM traffic_flow 
            WHERE timestamp >= ?
            GROUP BY hour
            ORDER BY hour
        '''
        
        df = pd.read_sql_query(query, conn, params=[time_filter])
        conn.close()
        
        if df.empty:
            return {'peak_hours': [], 'traffic_pattern': 'insufficient_data'}
            
        # Identify peak hours
        peak_threshold = df['avg_count'].quantile(0.8)
        peak_hours = df[df['avg_count'] >= peak_threshold]['hour'].tolist()
        
        # Classify traffic pattern
        morning_peak = any(6 <= hour <= 9 for hour in peak_hours)
        evening_peak = any(17 <= hour <= 20 for hour in peak_hours)
        
        if morning_peak and evening_peak:
            pattern = 'bi_modal'
        elif morning_peak or evening_peak:
            pattern = 'single_peak'
        else:
            pattern = 'distributed'
            
        return {
            'peak_hours': peak_hours,
            'traffic_pattern': pattern,
            'hourly_averages': df.to_dict('records'),
            'peak_hour_volume': float(df['avg_count'].max()),
            'off_peak_volume': float(df['avg_count'].min())
        }
        
    def calculate_congestion_metrics(self) -> Dict:
        """Calculate comprehensive congestion metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Recent congestion data
        time_filter = datetime.now() - timedelta(hours=24)
        query = '''
            SELECT congestion_level, COUNT(*) as frequency, AVG(average_speed) as avg_speed
            FROM traffic_flow 
            WHERE timestamp >= ?
            GROUP BY congestion_level
        '''
        
        df = pd.read_sql_query(query, conn, params=[time_filter])
        conn.close()
        
        if df.empty:
            return {'congestion_frequency': {}, 'overall_level': 'unknown'}
            
        # Calculate congestion frequency
        total_records = df['frequency'].sum()
        congestion_freq = {
            row['congestion_level']: (row['frequency'] / total_records) * 100
            for _, row in df.iterrows()
        }
        
        # Determine overall congestion level
        severe_pct = congestion_freq.get('severe', 0)
        high_pct = congestion_freq.get('high', 0)
        
        if severe_pct > 20:
            overall_level = 'critical'
        elif high_pct + severe_pct > 30:
            overall_level = 'high'
        elif congestion_freq.get('medium', 0) > 40:
            overall_level = 'moderate'
        else:
            overall_level = 'low'
            
        return {
            'congestion_frequency': congestion_freq,
            'overall_level': overall_level,
            'average_speeds_by_level': dict(zip(df['congestion_level'], df['avg_speed']))
        }
        
    def evaluate_signal_performance(self) -> Dict:
        """Evaluate traffic signal performance"""
        conn = sqlite3.connect(self.db_path)
        
        time_filter = datetime.now() - timedelta(hours=24)
        query = '''
            SELECT AVG(throughput) as avg_throughput, AVG(average_delay) as avg_delay,
                   level_of_service, COUNT(*) as frequency
            FROM signal_performance 
            WHERE timestamp >= ?
            GROUP BY level_of_service
        '''
        
        df = pd.read_sql_query(query, conn, params=[time_filter])
        conn.close()
        
        if df.empty:
            return {'performance_grade': 'unknown', 'efficiency_score': 0.0}
            
        # Calculate performance metrics
        avg_throughput = df['avg_throughput'].mean()
        avg_delay = df['avg_delay'].mean()
        
        # Grade distribution
        grade_distribution = dict(zip(df['level_of_service'], df['frequency']))
        
        # Calculate efficiency score (0-100)
        good_grades = ['A', 'B', 'C']
        total_frequency = df['frequency'].sum()
        good_frequency = df[df['level_of_service'].isin(good_grades)]['frequency'].sum()
        efficiency_score = (good_frequency / total_frequency) * 100 if total_frequency > 0 else 0
        
        return {
            'average_throughput': float(avg_throughput),
            'average_delay': float(avg_delay),
            'grade_distribution': grade_distribution,
            'efficiency_score': float(efficiency_score),
            'performance_grade': 'excellent' if efficiency_score > 80 else 
                               'good' if efficiency_score > 60 else
                               'poor'
        }
        
    def calculate_environmental_impact(self, vehicle_stats: Dict, traffic_flow: Dict) -> Dict:
        """Calculate environmental impact metrics"""
        
        # Emission factors (kg CO2 per vehicle per km)
        emission_factors = {
            'car': 0.12,
            'truck': 0.35,
            'bus': 0.28,
            'motorcycle': 0.08
        }
        
        total_emissions = 0
        fuel_consumption = 0
        
        # Calculate based on vehicle types and distances
        for vehicle_type, count in vehicle_stats.get('vehicle_types', {}).items():
            if vehicle_type in emission_factors:
                # Estimate distance traveled (simplified)
                avg_speed = vehicle_stats.get('average_speed', 30)  # km/h
                distance_per_vehicle = (avg_speed / 60) * 10  # Assume 10 minute observation
                
                emissions = count * distance_per_vehicle * emission_factors[vehicle_type]
                total_emissions += emissions
                
                # Fuel consumption (simplified calculation)
                fuel_per_km = emission_factors[vehicle_type] / 2.3  # rough conversion
                fuel_consumption += count * distance_per_vehicle * fuel_per_km
                
        # Traffic efficiency impact
        avg_speed = vehicle_stats.get('average_speed', 30)
        speed_efficiency = min(avg_speed / 50, 1.0)  # Optimal speed ~50 km/h
        
        congestion_penalty = 1.0
        if 'congestion_level' in traffic_flow:
            level = traffic_flow['congestion_level']
            if level == 'severe':
                congestion_penalty = 2.0
            elif level == 'high':
                congestion_penalty = 1.5
            elif level == 'medium':
                congestion_penalty = 1.2
                
        adjusted_emissions = total_emissions * congestion_penalty
        
        return {
            'total_co2_emissions': round(total_emissions, 2),  # kg
            'adjusted_emissions': round(adjusted_emissions, 2),  # kg with congestion factor
            'fuel_consumption': round(fuel_consumption, 2),  # liters
            'emission_efficiency': round(speed_efficiency, 2),
            'environmental_grade': 'excellent' if speed_efficiency > 0.8 else
                                 'good' if speed_efficiency > 0.6 else
                                 'poor'
        }
        
    def generate_recommendations(self, analytics_data: Dict) -> List[str]:
        """Generate actionable recommendations based on analytics"""
        recommendations = []
        
        # Vehicle statistics recommendations
        vehicle_stats = analytics_data.get('vehicle_statistics', {})
        if vehicle_stats.get('detection_confidence', 1.0) < 0.8:
            recommendations.append("Consider improving camera positioning or lighting for better detection accuracy")
            
        # Traffic flow recommendations
        traffic_patterns = analytics_data.get('traffic_flow', {})
        if traffic_patterns.get('traffic_pattern') == 'bi_modal':
            recommendations.append("Implement adaptive signal timing for morning and evening peak hours")
        elif len(traffic_patterns.get('peak_hours', [])) > 6:
            recommendations.append("Consider implementing time-of-day pricing or alternative route guidance")
            
        # Congestion recommendations
        congestion = analytics_data.get('congestion_analysis', {})
        if congestion.get('overall_level') == 'critical':
            recommendations.append("Immediate intervention required: consider emergency traffic management protocols")
        elif congestion.get('overall_level') == 'high':
            recommendations.append("Implement dynamic lane management or ramp metering")
            
        # Signal performance recommendations
        signal_perf = analytics_data.get('signal_performance', {})
        if signal_perf.get('efficiency_score', 100) < 60:
            recommendations.append("Optimize signal timing parameters - current efficiency is below acceptable levels")
        if signal_perf.get('average_delay', 0) > 45:
            recommendations.append("Reduce signal cycle times or implement coordinated signal control")
            
        # Environmental recommendations
        env_impact = analytics_data.get('environmental_impact', {})
        if env_impact.get('environmental_grade') == 'poor':
            recommendations.append("Implement eco-friendly traffic management to reduce emissions")
            recommendations.append("Consider promoting public transportation or carpooling initiatives")
            
        return recommendations
        
    def generate_comprehensive_report(self, period: str = 'daily') -> AnalyticsReport:
        """Generate comprehensive analytics report"""
        
        # Collect all analytics data
        vehicle_stats = self.calculate_vehicle_statistics(period)
        traffic_patterns = self.analyze_traffic_patterns(period)
        congestion_metrics = self.calculate_congestion_metrics()
        signal_performance = self.evaluate_signal_performance()
        environmental_impact = self.calculate_environmental_impact(vehicle_stats, traffic_patterns)
        
        analytics_data = {
            'vehicle_statistics': vehicle_stats,
            'traffic_flow': traffic_patterns,
            'congestion_analysis': congestion_metrics,
            'signal_performance': signal_performance,
            'environmental_impact': environmental_impact
        }
        
        recommendations = self.generate_recommendations(analytics_data)
        
        report = AnalyticsReport(
            timestamp=datetime.now(),
            period=period,
            vehicle_statistics=vehicle_stats,
            traffic_flow=traffic_patterns,
            congestion_analysis=congestion_metrics,
            signal_performance=signal_performance,
            environmental_impact=environmental_impact,
            recommendations=recommendations
        )
        
        return report
        
    def export_report(self, report: AnalyticsReport, format: str = 'json') -> str:
        """Export analytics report in specified format"""
        
        if format == 'json':
            report_dict = asdict(report)
            # Convert datetime to string for JSON serialization
            report_dict['timestamp'] = report.timestamp.isoformat()
            return json.dumps(report_dict, indent=2)
        elif format == 'csv':
            # Create a simplified CSV version
            csv_data = []
            csv_data.append(f"Traffic Analytics Report - {report.period}")
            csv_data.append(f"Generated: {report.timestamp}")
            csv_data.append("")
            csv_data.append("Vehicle Statistics:")
            for key, value in report.vehicle_statistics.items():
                csv_data.append(f"{key},{value}")
            # Add other sections...
            return "\n".join(csv_data)
        else:
            return str(report)
            
    def get_performance_dashboard_data(self) -> Dict:
        """Get data for real-time performance dashboard"""
        
        # Recent performance metrics
        vehicle_stats = self.calculate_vehicle_statistics('hourly')
        congestion = self.calculate_congestion_metrics()
        signal_perf = self.evaluate_signal_performance()
        
        # Real-time indicators
        dashboard_data = {
            'current_vehicle_count': vehicle_stats.get('total_vehicles', 0),
            'average_speed': vehicle_stats.get('average_speed', 0),
            'congestion_level': congestion.get('overall_level', 'unknown'),
            'signal_efficiency': signal_perf.get('efficiency_score', 0),
            'detection_accuracy': vehicle_stats.get('detection_confidence', 0) * 100,
            'environmental_grade': 'good',  # Simplified
            'alerts': self._generate_real_time_alerts(vehicle_stats, congestion, signal_perf)
        }
        
        return dashboard_data
        
    def _generate_real_time_alerts(self, vehicle_stats: Dict, 
                                  congestion: Dict, signal_perf: Dict) -> List[Dict]:
        """Generate real-time alerts for dashboard"""
        alerts = []
        
        if congestion.get('overall_level') == 'critical':
            alerts.append({
                'level': 'critical',
                'message': 'Critical congestion detected',
                'timestamp': datetime.now().isoformat()
            })
            
        if signal_perf.get('efficiency_score', 100) < 50:
            alerts.append({
                'level': 'warning',
                'message': 'Poor signal performance detected',
                'timestamp': datetime.now().isoformat()
            })
            
        if vehicle_stats.get('detection_confidence', 1.0) < 0.7:
            alerts.append({
                'level': 'info',
                'message': 'Detection accuracy below optimal',
                'timestamp': datetime.now().isoformat()
            })
            
        return alerts
