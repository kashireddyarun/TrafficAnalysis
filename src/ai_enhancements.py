"""
Advanced AI Features for Traffic Analysis
Next-generation enhancements using deep learning
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle
from collections import deque
import cv2

@dataclass
class PredictionResult:
    """Traffic prediction result structure"""
    predicted_count: int
    confidence: float
    time_horizon: int  # minutes
    prediction_type: str  # 'flow', 'congestion', 'incident'

class TrafficPredictor(nn.Module):
    """LSTM-based traffic flow prediction model"""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2):
        super(TrafficPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)  # Predict single value (vehicle count)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

class IncidentDetector:
    """AI-powered incident detection system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Historical data for anomaly detection
        self.speed_history = deque(maxlen=100)
        self.density_history = deque(maxlen=100)
        self.flow_history = deque(maxlen=100)
        
        # Thresholds for incident detection
        self.speed_drop_threshold = 0.3  # 30% speed drop
        self.density_spike_threshold = 2.0  # 200% density increase
        self.stagnation_threshold = 60  # seconds
        
        # Machine learning model for incident classification
        self.incident_classifier = self._load_incident_classifier()
        
    def _load_incident_classifier(self):
        """Load pre-trained incident classification model"""
        # In a real implementation, this would load a trained model
        # For now, return a placeholder
        return None
        
    def detect_incidents(self, current_metrics: Dict, tracks: List) -> List[Dict]:
        """Detect traffic incidents using AI"""
        incidents = []
        
        # Speed-based incident detection
        speed_incident = self._detect_speed_anomaly(current_metrics)
        if speed_incident:
            incidents.append(speed_incident)
            
        # Stopped vehicle detection
        stopped_incident = self._detect_stopped_vehicles(tracks)
        if stopped_incident:
            incidents.append(stopped_incident)
            
        # Pattern-based detection
        pattern_incident = self._detect_unusual_patterns(current_metrics)
        if pattern_incident:
            incidents.append(pattern_incident)
            
        return incidents
        
    def _detect_speed_anomaly(self, metrics: Dict) -> Optional[Dict]:
        """Detect sudden speed drops indicating incidents"""
        current_speed = metrics.get('average_speed', 0)
        self.speed_history.append(current_speed)
        
        if len(self.speed_history) < 10:
            return None
            
        recent_avg = np.mean(list(self.speed_history)[-10:])
        historical_avg = np.mean(list(self.speed_history)[:-10])
        
        if historical_avg > 0 and (recent_avg / historical_avg) < (1 - self.speed_drop_threshold):
            return {
                'type': 'speed_anomaly',
                'severity': 'high' if recent_avg < 5 else 'medium',
                'description': f'Sudden speed drop: {historical_avg:.1f} → {recent_avg:.1f} km/h',
                'confidence': 0.8,
                'location': 'main_area'
            }
            
        return None
        
    def _detect_stopped_vehicles(self, tracks: List) -> Optional[Dict]:
        """Detect vehicles stopped for extended periods"""
        stopped_vehicles = []
        
        for track in tracks:
            if track.get('speed', 0) < 2 and track.get('stationary_time', 0) > self.stagnation_threshold:
                stopped_vehicles.append(track)
                
        if len(stopped_vehicles) >= 3:  # Multiple stopped vehicles
            return {
                'type': 'stopped_vehicles',
                'severity': 'high',
                'description': f'{len(stopped_vehicles)} vehicles stopped for >60s',
                'confidence': 0.9,
                'affected_vehicles': len(stopped_vehicles)
            }
            
        return None
        
    def _detect_unusual_patterns(self, metrics: Dict) -> Optional[Dict]:
        """Detect unusual traffic patterns"""
        current_density = metrics.get('density', 0)
        self.density_history.append(current_density)
        
        if len(self.density_history) < 20:
            return None
            
        # Check for sudden density spikes
        recent_density = np.mean(list(self.density_history)[-5:])
        baseline_density = np.mean(list(self.density_history)[-20:-5])
        
        if baseline_density > 0 and (recent_density / baseline_density) > self.density_spike_threshold:
            return {
                'type': 'density_spike',
                'severity': 'medium',
                'description': f'Unusual traffic buildup detected',
                'confidence': 0.7
            }
            
        return None

class SmartTrafficController:
    """AI-powered adaptive traffic control system"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Reinforcement learning components
        self.state_size = 12  # traffic state features
        self.action_size = 8   # possible signal timing actions
        
        # Q-learning parameters
        self.q_table = np.random.uniform(low=-2, high=0, size=(1000, self.action_size))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.3
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.learning_enabled = config.get('ai', {}).get('learning_enabled', True)
        
    def get_traffic_state(self, metrics: Dict) -> np.ndarray:
        """Convert traffic metrics to state vector for AI"""
        state = np.array([
            metrics.get('vehicle_count', 0) / 50,  # Normalized vehicle count
            metrics.get('average_speed', 0) / 60,  # Normalized speed
            metrics.get('density', 0),             # Traffic density
            metrics.get('queue_length_ns', 0) / 20, # North-South queue
            metrics.get('queue_length_ew', 0) / 20, # East-West queue
            metrics.get('waiting_time_ns', 0) / 120, # North-South wait time
            metrics.get('waiting_time_ew', 0) / 120, # East-West wait time
            metrics.get('congestion_level_numeric', 0) / 4,  # Congestion (0-4)
            metrics.get('time_of_day', 12) / 24,   # Time factor
            metrics.get('weather_factor', 1.0),    # Weather impact
            metrics.get('incident_factor', 0),     # Incident presence
            metrics.get('special_event_factor', 0) # Special events
        ])
        
        return np.clip(state, 0, 1)  # Ensure values are between 0 and 1
        
    def state_to_index(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete index for Q-table"""
        # Simple discretization - in practice, use more sophisticated methods
        discretized = (state * 10).astype(int)
        hash_value = hash(tuple(discretized))
        return abs(hash_value) % len(self.q_table)
        
    def choose_action(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        state_index = self.state_to_index(state)
        
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.choice(self.action_size)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state_index])
            
    def action_to_signal_timing(self, action: int) -> Dict:
        """Convert action index to signal timing parameters"""
        # Define action space
        actions = {
            0: {'ns_green': 30, 'ew_green': 30},  # Balanced
            1: {'ns_green': 40, 'ew_green': 20},  # Favor NS
            2: {'ns_green': 20, 'ew_green': 40},  # Favor EW
            3: {'ns_green': 50, 'ew_green': 10},  # Heavily favor NS
            4: {'ns_green': 10, 'ew_green': 50},  # Heavily favor EW
            5: {'ns_green': 35, 'ew_green': 25},  # Slightly favor NS
            6: {'ns_green': 25, 'ew_green': 35},  # Slightly favor EW
            7: {'ns_green': 45, 'ew_green': 15}   # Strong NS preference
        }
        
        return actions.get(action, actions[0])
        
    def calculate_reward(self, previous_metrics: Dict, current_metrics: Dict) -> float:
        """Calculate reward for reinforcement learning"""
        # Reward components
        throughput_reward = current_metrics.get('throughput', 0) * 0.3
        delay_penalty = -current_metrics.get('average_delay', 0) * 0.5
        queue_penalty = -(current_metrics.get('total_queue_length', 0) * 0.2)
        
        # Efficiency bonus
        efficiency = current_metrics.get('efficiency_score', 0) / 100
        efficiency_bonus = efficiency * 2.0
        
        # Congestion penalty
        congestion_level = current_metrics.get('congestion_level_numeric', 0)
        congestion_penalty = -congestion_level * 0.8
        
        total_reward = (throughput_reward + delay_penalty + queue_penalty + 
                       efficiency_bonus + congestion_penalty)
        
        return total_reward
        
    def update_q_table(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray) -> None:
        """Update Q-table using Q-learning algorithm"""
        if not self.learning_enabled:
            return
            
        current_state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        
        # Q-learning update
        current_q = self.q_table[current_state_index, action]
        max_next_q = np.max(self.q_table[next_state_index])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[current_state_index, action] = new_q
        
        # Decay exploration rate
        self.exploration_rate = max(0.01, self.exploration_rate * 0.999)
        
    def optimize_intersection(self, current_metrics: Dict, 
                            previous_metrics: Optional[Dict] = None) -> Dict:
        """Main optimization function using AI"""
        
        # Get current state
        state = self.get_traffic_state(current_metrics)
        
        # Choose action
        action = self.choose_action(state)
        
        # Convert to signal timing
        signal_timing = self.action_to_signal_timing(action)
        
        # Update learning if we have previous data
        if previous_metrics and hasattr(self, 'previous_state') and hasattr(self, 'previous_action'):
            reward = self.calculate_reward(previous_metrics, current_metrics)
            self.update_q_table(self.previous_state, self.previous_action, reward, state)
            
        # Store for next iteration
        self.previous_state = state
        self.previous_action = action
        
        # Add AI metadata
        signal_timing.update({
            'ai_action': action,
            'ai_confidence': 1.0 - self.exploration_rate,
            'state_features': state.tolist(),
            'learning_enabled': self.learning_enabled
        })
        
        return signal_timing

class WeatherImpactAnalyzer:
    """Analyze weather impact on traffic patterns"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Weather impact factors
        self.weather_factors = {
            'clear': 1.0,
            'rain': 0.7,      # 30% reduction in capacity
            'heavy_rain': 0.5, # 50% reduction
            'snow': 0.4,       # 60% reduction
            'fog': 0.6,        # 40% reduction
            'wind': 0.9        # 10% reduction
        }
        
    def analyze_weather_impact(self, weather_condition: str, 
                             base_metrics: Dict) -> Dict:
        """Analyze how weather affects traffic"""
        
        factor = self.weather_factors.get(weather_condition, 1.0)
        
        adjusted_metrics = {
            'capacity_reduction': 1 - factor,
            'speed_reduction': 1 - factor,
            'safety_factor': factor,
            'visibility_impact': 0.8 if weather_condition in ['fog', 'heavy_rain'] else 1.0,
            'recommended_actions': self._get_weather_recommendations(weather_condition)
        }
        
        return adjusted_metrics
        
    def _get_weather_recommendations(self, weather: str) -> List[str]:
        """Get traffic management recommendations for weather conditions"""
        
        recommendations = {
            'rain': [
                "Increase signal timing for safer turns",
                "Activate variable message signs about wet conditions",
                "Reduce speed limits on highways"
            ],
            'heavy_rain': [
                "Implement emergency traffic protocols",
                "Increase police presence at major intersections",
                "Consider traffic restrictions on dangerous routes"
            ],
            'snow': [
                "Activate snow emergency routes",
                "Increase signal cycle times",
                "Deploy emergency response teams"
            ],
            'fog': [
                "Activate fog warning systems",
                "Reduce highway speed limits",
                "Increase signal visibility"
            ]
        }
        
        return recommendations.get(weather, ["Monitor conditions closely"])

# Integration function
def create_ai_enhanced_system(config: Dict) -> Dict:
    """Create AI-enhanced traffic management system"""
    
    return {
        'predictor': TrafficPredictor(),
        'incident_detector': IncidentDetector(config),
        'smart_controller': SmartTrafficController(config),
        'weather_analyzer': WeatherImpactAnalyzer(config)
    }
