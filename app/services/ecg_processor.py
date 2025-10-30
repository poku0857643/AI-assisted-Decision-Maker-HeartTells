import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from app.config.config import settings
import json
import pickle
import os

class ECGSignalProcessor:
    """Advanced ECG signal processing for sentiment recognition and personal profiling"""
    
    def __init__(self):
        self.sampling_rate = 512  # Apple Watch sampling rate
        self.window_size = 30  # seconds
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        self.sentiment_model = None
        self.personal_profile_model = None
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models or initialize new ones"""
        try:
            # Try to load existing models
            model_path = os.path.join(settings.EMBEDDING_STORAGE_PATH, "ecg_sentiment_model.h5")
            if os.path.exists(model_path):
                self.sentiment_model = tf.keras.models.load_model(model_path)
            else:
                self.sentiment_model = self._create_sentiment_model()
            
            # Load scaler and PCA if they exist
            scaler_path = os.path.join(settings.EMBEDDING_STORAGE_PATH, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            pca_path = os.path.join(settings.EMBEDDING_STORAGE_PATH, "pca.pkl")
            if os.path.exists(pca_path):
                with open(pca_path, 'rb') as f:
                    self.pca = pickle.load(f)
                    
        except Exception as e:
            print(f"Error loading models: {e}")
            self.sentiment_model = self._create_sentiment_model()
    
    def _create_sentiment_model(self) -> tf.keras.Model:
        """Create a deep learning model for sentiment recognition from ECG"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(150, 1)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Dropout(0.2),
            
            LSTM(100, return_sequences=True),
            LSTM(50),
            Dropout(0.3),
            
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),
            Dense(7, activation='softmax')  # 7 emotion classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def preprocess_ecg_signal(self, raw_ecg_data: Dict) -> np.ndarray:
        """Preprocess raw ECG signal from Apple Watch"""
        try:
            # Extract voltage measurements
            if 'voltage_measurements' in raw_ecg_data:
                signal_data = np.array(raw_ecg_data['voltage_measurements'])
            elif 'voltageValues' in raw_ecg_data:
                signal_data = np.array(raw_ecg_data['voltageValues'])
            else:
                raise ValueError("No voltage data found in ECG record")
            
            # Remove DC component
            signal_data = signal_data - np.mean(signal_data)
            
            # Apply bandpass filter (0.5-40 Hz for ECG)
            nyquist = self.sampling_rate / 2
            low = 0.5 / nyquist
            high = 40.0 / nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            
            # Normalize
            filtered_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
            
            return filtered_signal
            
        except Exception as e:
            print(f"Error preprocessing ECG signal: {e}")
            return np.array([])
    
    def extract_hrv_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract Heart Rate Variability features for sentiment analysis"""
        try:
            # Find R-peaks using simple peak detection
            peaks, _ = signal.find_peaks(ecg_signal, height=0.5, distance=int(0.6 * self.sampling_rate))
            
            if len(peaks) < 2:
                return {}
            
            # Calculate RR intervals
            rr_intervals = np.diff(peaks) / self.sampling_rate * 1000  # in milliseconds
            
            if len(rr_intervals) < 2:
                return {}
            
            # Time domain features
            features = {
                'mean_rr': np.mean(rr_intervals),
                'std_rr': np.std(rr_intervals),
                'rmssd': np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),
                'pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
                'heart_rate': 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
            }
            
            # Frequency domain features (if enough data)
            if len(rr_intervals) > 10:
                # Interpolate RR intervals for frequency analysis
                time_original = np.cumsum(rr_intervals)
                time_interp = np.arange(0, time_original[-1], 250)  # 4Hz interpolation
                rr_interp = np.interp(time_interp, time_original[:-1], rr_intervals)
                
                # Power spectral density
                freqs, psd = signal.welch(rr_interp, fs=4, nperseg=len(rr_interp)//4)
                
                # Frequency bands
                vlf_band = (freqs >= 0.003) & (freqs < 0.04)
                lf_band = (freqs >= 0.04) & (freqs < 0.15)
                hf_band = (freqs >= 0.15) & (freqs < 0.4)
                
                features.update({
                    'vlf_power': np.trapz(psd[vlf_band], freqs[vlf_band]),
                    'lf_power': np.trapz(psd[lf_band], freqs[lf_band]),
                    'hf_power': np.trapz(psd[hf_band], freqs[hf_band]),
                })
                
                # LF/HF ratio (autonomic balance indicator)
                if features['hf_power'] > 0:
                    features['lf_hf_ratio'] = features['lf_power'] / features['hf_power']
                else:
                    features['lf_hf_ratio'] = 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting HRV features: {e}")
            return {}
    
    def extract_morphological_features(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Extract ECG morphological features for personal sentiment profiling"""
        try:
            features = {}
            
            # Statistical features
            features.update({
                'signal_mean': np.mean(ecg_signal),
                'signal_std': np.std(ecg_signal),
                'signal_skewness': self._skewness(ecg_signal),
                'signal_kurtosis': self._kurtosis(ecg_signal),
                'signal_entropy': entropy(np.histogram(ecg_signal, bins=50)[0] + 1e-10)
            })
            
            # Spectral features
            freqs, psd = signal.welch(ecg_signal, fs=self.sampling_rate)
            features.update({
                'spectral_centroid': np.sum(freqs * psd) / np.sum(psd),
                'spectral_bandwidth': np.sqrt(np.sum(((freqs - features['spectral_centroid']) ** 2) * psd) / np.sum(psd)),
                'spectral_rolloff': freqs[np.where(np.cumsum(psd) >= 0.85 * np.sum(psd))[0][0]]
            })
            
            return features
            
        except Exception as e:
            print(f"Error extracting morphological features: {e}")
            return {}
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def create_embedding(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Create embedding vector from ECG signal"""
        try:
            # Extract all features
            hrv_features = self.extract_hrv_features(ecg_signal)
            morph_features = self.extract_morphological_features(ecg_signal)
            
            # Combine features
            all_features = {**hrv_features, **morph_features}
            
            # Convert to array
            feature_vector = np.array(list(all_features.values()))
            
            # Handle NaN values
            feature_vector = np.nan_to_num(feature_vector)
            
            # Normalize
            if hasattr(self.scaler, 'mean_'):
                feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]
            else:
                # Fit scaler if not already fitted
                feature_vector = self.scaler.fit_transform(feature_vector.reshape(1, -1))[0]
            
            # Apply PCA for dimensionality reduction
            if hasattr(self.pca, 'components_'):
                embedding = self.pca.transform(feature_vector.reshape(1, -1))[0]
            else:
                # Create a dummy embedding if PCA not fitted
                embedding = feature_vector[:50] if len(feature_vector) >= 50 else np.pad(feature_vector, (0, 50 - len(feature_vector)))
            
            return embedding
            
        except Exception as e:
            print(f"Error creating embedding: {e}")
            return np.zeros(50)  # Return zero embedding on error
    
    def predict_sentiment(self, ecg_signal: np.ndarray) -> Dict[str, float]:
        """Predict emotional sentiment from ECG signal"""
        try:
            # Prepare signal for CNN-LSTM model
            if len(ecg_signal) > 150:
                # Take a representative segment
                start_idx = len(ecg_signal) // 2 - 75
                signal_segment = ecg_signal[start_idx:start_idx + 150]
            else:
                # Pad or repeat signal to reach 150 samples
                signal_segment = np.pad(ecg_signal, (0, max(0, 150 - len(ecg_signal))), mode='constant')[:150]
            
            # Reshape for model input
            model_input = signal_segment.reshape(1, 150, 1)
            
            # Predict using the sentiment model
            if self.sentiment_model:
                predictions = self.sentiment_model.predict(model_input, verbose=0)[0]
                
                emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprised', 'calm']
                sentiment_scores = {
                    emotion: float(score) for emotion, score in zip(emotion_labels, predictions)
                }
                
                # Add confidence score
                sentiment_scores['confidence'] = float(np.max(predictions))
                sentiment_scores['dominant_emotion'] = emotion_labels[np.argmax(predictions)]
                
                return sentiment_scores
            else:
                # Fallback to simple HRV-based sentiment estimation
                hrv_features = self.extract_hrv_features(ecg_signal)
                return self._simple_sentiment_estimation(hrv_features)
                
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return {'neutral': 1.0, 'confidence': 0.0, 'dominant_emotion': 'neutral'}
    
    def _simple_sentiment_estimation(self, hrv_features: Dict[str, float]) -> Dict[str, float]:
        """Simple sentiment estimation based on HRV features"""
        try:
            if not hrv_features:
                return {'neutral': 1.0, 'confidence': 0.0, 'dominant_emotion': 'neutral'}
            
            hr = hrv_features.get('heart_rate', 70)
            rmssd = hrv_features.get('rmssd', 30)
            lf_hf = hrv_features.get('lf_hf_ratio', 1.0)
            
            # Simple rules based on physiological research
            if hr > 100:  # High heart rate
                if lf_hf > 2.0:  # High sympathetic activity
                    return {'angry': 0.6, 'fearful': 0.3, 'neutral': 0.1, 'confidence': 0.7, 'dominant_emotion': 'angry'}
                else:
                    return {'happy': 0.5, 'surprised': 0.3, 'neutral': 0.2, 'confidence': 0.6, 'dominant_emotion': 'happy'}
            elif hr < 60:  # Low heart rate
                if rmssd > 50:  # High parasympathetic activity
                    return {'calm': 0.7, 'neutral': 0.3, 'confidence': 0.8, 'dominant_emotion': 'calm'}
                else:
                    return {'sad': 0.6, 'neutral': 0.4, 'confidence': 0.5, 'dominant_emotion': 'sad'}
            else:
                return {'neutral': 0.8, 'calm': 0.2, 'confidence': 0.6, 'dominant_emotion': 'neutral'}
                
        except Exception as e:
            print(f"Error in simple sentiment estimation: {e}")
            return {'neutral': 1.0, 'confidence': 0.0, 'dominant_emotion': 'neutral'}
    
    def analyze_personal_patterns(self, user_ecg_history: List[Dict]) -> Dict[str, any]:
        """Analyze personal ECG patterns for personalized insights"""
        try:
            if not user_ecg_history:
                return {}
            
            embeddings = []
            sentiments = []
            timestamps = []
            
            for record in user_ecg_history:
                if record.get('raw_data'):
                    signal_data = self.preprocess_ecg_signal(record['raw_data'])
                    if len(signal_data) > 0:
                        embedding = self.create_embedding(signal_data)
                        sentiment = self.predict_sentiment(signal_data)
                        
                        embeddings.append(embedding)
                        sentiments.append(sentiment)
                        timestamps.append(record.get('recorded_at'))
            
            if not embeddings:
                return {}
            
            embeddings = np.array(embeddings)
            
            # Detect anomalies in ECG patterns
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = isolation_forest.fit_predict(embeddings)
            
            # Cluster analysis for pattern identification
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Personal pattern analysis
            personal_patterns = {
                'total_records': len(embeddings),
                'anomaly_count': np.sum(anomaly_scores == -1),
                'pattern_clusters': n_clusters,
                'dominant_emotions': self._analyze_emotion_patterns(sentiments),
                'stress_indicators': self._analyze_stress_patterns(sentiments),
                'temporal_patterns': self._analyze_temporal_patterns(sentiments, timestamps)
            }
            
            return personal_patterns
            
        except Exception as e:
            print(f"Error analyzing personal patterns: {e}")
            return {}
    
    def _analyze_emotion_patterns(self, sentiments: List[Dict]) -> Dict[str, float]:
        """Analyze dominant emotional patterns"""
        emotion_counts = {}
        for sentiment in sentiments:
            dominant = sentiment.get('dominant_emotion', 'neutral')
            emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
        
        total = len(sentiments)
        return {emotion: count/total for emotion, count in emotion_counts.items()}
    
    def _analyze_stress_patterns(self, sentiments: List[Dict]) -> Dict[str, float]:
        """Analyze stress-related patterns"""
        stress_emotions = ['angry', 'fearful', 'sad']
        calm_emotions = ['calm', 'neutral', 'happy']
        
        stress_count = sum(1 for s in sentiments if s.get('dominant_emotion') in stress_emotions)
        calm_count = sum(1 for s in sentiments if s.get('dominant_emotion') in calm_emotions)
        
        total = len(sentiments)
        return {
            'stress_ratio': stress_count / total if total > 0 else 0,
            'calm_ratio': calm_count / total if total > 0 else 0,
            'avg_confidence': np.mean([s.get('confidence', 0) for s in sentiments])
        }
    
    def _analyze_temporal_patterns(self, sentiments: List[Dict], timestamps: List) -> Dict[str, any]:
        """Analyze temporal emotion patterns"""
        if not timestamps:
            return {}
        
        try:
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(timestamps),
                'emotion': [s.get('dominant_emotion', 'neutral') for s in sentiments],
                'confidence': [s.get('confidence', 0) for s in sentiments]
            })
            
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            
            hourly_patterns = df.groupby('hour')['emotion'].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral').to_dict()
            weekly_patterns = df.groupby('day_of_week')['emotion'].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral').to_dict()
            
            return {
                'hourly_patterns': hourly_patterns,
                'weekly_patterns': weekly_patterns,
                'most_active_hour': df['hour'].mode().iloc[0] if len(df) > 0 else 12,
                'avg_daily_confidence': df.groupby('day_of_week')['confidence'].mean().to_dict()
            }
            
        except Exception as e:
            print(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def save_models(self):
        """Save trained models and preprocessors"""
        try:
            os.makedirs(settings.EMBEDDING_STORAGE_PATH, exist_ok=True)
            
            if self.sentiment_model:
                self.sentiment_model.save(os.path.join(settings.EMBEDDING_STORAGE_PATH, "ecg_sentiment_model.h5"))
            
            with open(os.path.join(settings.EMBEDDING_STORAGE_PATH, "scaler.pkl"), 'wb') as f:
                pickle.dump(self.scaler, f)
            
            with open(os.path.join(settings.EMBEDDING_STORAGE_PATH, "pca.pkl"), 'wb') as f:
                pickle.dump(self.pca, f)
                
        except Exception as e:
            print(f"Error saving models: {e}")