from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json
from datetime import datetime
from app.services.ecg_processor import ECGSignalProcessor
from app.config.config import settings

class DecisionType(Enum):
    BINARY = "binary"  # Yes/No decisions
    MULTIPLE_CHOICE = "multiple_choice"  # A, B, C, D options
    SCALED = "scaled"  # 1-10 scale decisions
    OPEN_ENDED = "open_ended"  # Free form decisions

class EmotionalState(Enum):
    CALM = "calm"
    STRESSED = "stressed"
    EXCITED = "excited"
    CONFUSED = "confused"
    CONFIDENT = "confident"
    ANXIOUS = "anxious"

@dataclass
class Decision:
    id: str
    question: str
    options: List[str]
    decision_type: DecisionType
    user_response: Any
    confidence_level: float
    timestamp: datetime
    context: Dict[str, Any] = None

@dataclass
class ECGContext:
    sentiment_analysis: Dict[str, float]
    stress_indicators: Dict[str, float]
    personal_patterns: Dict[str, Any]
    emotional_state: EmotionalState
    physiological_markers: Dict[str, float]

@dataclass
class IntegratedDecisionResult:
    decision: Decision
    ecg_context: ECGContext
    recommendation: str
    confidence_score: float
    reasoning: str
    alternative_suggestions: List[str]
    risk_assessment: Dict[str, float]

class DecisionIntegrationEngine:
    """Integrates ECG sentiment analysis with decision-making process"""
    
    def __init__(self):
        self.ecg_processor = ECGSignalProcessor()
        self.decision_history: List[IntegratedDecisionResult] = []
        self.personal_decision_patterns: Dict[str, Any] = {}
    
    def analyze_decision_context(self, ecg_data: Dict, decisions: List[Decision]) -> List[IntegratedDecisionResult]:
        """Analyze decisions in the context of ECG emotional state"""
        try:
            # Process ECG signal
            ecg_signal = self.ecg_processor.preprocess_ecg_signal(ecg_data)
            
            if len(ecg_signal) == 0:
                return self._create_fallback_results(decisions)
            
            # Extract ECG context
            ecg_context = self._extract_ecg_context(ecg_signal)
            
            # Analyze each decision
            results = []
            for decision in decisions:
                result = self._analyze_single_decision(decision, ecg_context)
                results.append(result)
                self.decision_history.append(result)
            
            return results
            
        except Exception as e:
            print(f"Error analyzing decision context: {e}")
            return self._create_fallback_results(decisions)
    
    def _extract_ecg_context(self, ecg_signal: np.ndarray) -> ECGContext:
        """Extract comprehensive context from ECG signal"""
        try:
            # Get sentiment analysis
            sentiment_analysis = self.ecg_processor.predict_sentiment(ecg_signal)
            
            # Extract HRV features for stress indicators
            hrv_features = self.ecg_processor.extract_hrv_features(ecg_signal)
            
            # Determine emotional state
            emotional_state = self._determine_emotional_state(sentiment_analysis, hrv_features)
            
            # Calculate stress indicators
            stress_indicators = self._calculate_stress_indicators(hrv_features, sentiment_analysis)
            
            # Get physiological markers
            physiological_markers = self._extract_physiological_markers(hrv_features)
            
            return ECGContext(
                sentiment_analysis=sentiment_analysis,
                stress_indicators=stress_indicators,
                personal_patterns={},  # Will be filled from user history
                emotional_state=emotional_state,
                physiological_markers=physiological_markers
            )
            
        except Exception as e:
            print(f"Error extracting ECG context: {e}")
            return self._create_default_context()
    
    def _determine_emotional_state(self, sentiment: Dict[str, float], hrv: Dict[str, float]) -> EmotionalState:
        """Determine overall emotional state from sentiment and HRV data"""
        try:
            dominant_emotion = sentiment.get('dominant_emotion', 'neutral')
            confidence = sentiment.get('confidence', 0)
            heart_rate = hrv.get('heart_rate', 70)
            rmssd = hrv.get('rmssd', 30)
            lf_hf_ratio = hrv.get('lf_hf_ratio', 1.0)
            
            # High confidence in emotional classification
            if confidence > 0.7:
                if dominant_emotion in ['angry', 'fearful']:
                    return EmotionalState.STRESSED
                elif dominant_emotion == 'happy':
                    return EmotionalState.EXCITED if heart_rate > 80 else EmotionalState.CONFIDENT
                elif dominant_emotion == 'calm':
                    return EmotionalState.CALM
                elif dominant_emotion == 'sad':
                    return EmotionalState.ANXIOUS
            
            # Physiological indicators
            if heart_rate > 100 or lf_hf_ratio > 3.0:
                return EmotionalState.STRESSED
            elif heart_rate < 60 and rmssd > 50:
                return EmotionalState.CALM
            elif heart_rate > 85 and lf_hf_ratio > 1.5:
                return EmotionalState.EXCITED
            elif rmssd < 20 or lf_hf_ratio > 2.5:
                return EmotionalState.ANXIOUS
            else:
                return EmotionalState.CONFIDENT
                
        except Exception as e:
            print(f"Error determining emotional state: {e}")
            return EmotionalState.CONFIDENT
    
    def _calculate_stress_indicators(self, hrv: Dict[str, float], sentiment: Dict[str, float]) -> Dict[str, float]:
        """Calculate various stress indicators"""
        try:
            heart_rate = hrv.get('heart_rate', 70)
            rmssd = hrv.get('rmssd', 30)
            lf_hf_ratio = hrv.get('lf_hf_ratio', 1.0)
            stress_emotions = sentiment.get('angry', 0) + sentiment.get('fearful', 0) + sentiment.get('sad', 0)
            
            # Normalize stress indicators (0-1 scale)
            hr_stress = min(max((heart_rate - 60) / 40, 0), 1)  # 60-100 bpm range
            hrv_stress = min(max((2.5 - lf_hf_ratio) / 2.5, 0), 1)  # Inverted LF/HF ratio
            rmssd_stress = min(max((50 - rmssd) / 50, 0), 1)  # Lower RMSSD = higher stress
            emotion_stress = min(stress_emotions, 1)
            
            overall_stress = (hr_stress + hrv_stress + rmssd_stress + emotion_stress) / 4
            
            return {
                'overall_stress': overall_stress,
                'physiological_stress': (hr_stress + hrv_stress + rmssd_stress) / 3,
                'emotional_stress': emotion_stress,
                'autonomic_balance': 1 - abs(lf_hf_ratio - 1.0) / 3.0,  # Closer to 1.0 is better
                'stress_level': 'high' if overall_stress > 0.7 else 'medium' if overall_stress > 0.4 else 'low'
            }
            
        except Exception as e:
            print(f"Error calculating stress indicators: {e}")
            return {'overall_stress': 0.5, 'stress_level': 'medium'}
    
    def _extract_physiological_markers(self, hrv: Dict[str, float]) -> Dict[str, float]:
        """Extract key physiological markers for decision analysis"""
        return {
            'heart_rate': hrv.get('heart_rate', 70),
            'hrv_score': hrv.get('rmssd', 30),
            'autonomic_balance': hrv.get('lf_hf_ratio', 1.0),
            'cardiac_coherence': hrv.get('pnn50', 10),
            'recovery_state': min(hrv.get('rmssd', 30) / 50, 1.0)
        }
    
    def _analyze_single_decision(self, decision: Decision, ecg_context: ECGContext) -> IntegratedDecisionResult:
        """Analyze a single decision with ECG context"""
        try:
            # Assess decision quality based on emotional state
            quality_assessment = self._assess_decision_quality(decision, ecg_context)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(decision, ecg_context, quality_assessment)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(decision, ecg_context, quality_assessment)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(decision, ecg_context, quality_assessment)
            
            # Generate alternative suggestions
            alternatives = self._generate_alternatives(decision, ecg_context)
            
            # Assess risks
            risk_assessment = self._assess_decision_risks(decision, ecg_context)
            
            return IntegratedDecisionResult(
                decision=decision,
                ecg_context=ecg_context,
                recommendation=recommendation,
                confidence_score=confidence_score,
                reasoning=reasoning,
                alternative_suggestions=alternatives,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            print(f"Error analyzing single decision: {e}")
            return self._create_fallback_result(decision, ecg_context)
    
    def _assess_decision_quality(self, decision: Decision, ecg_context: ECGContext) -> Dict[str, float]:
        """Assess the quality of decision-making based on emotional state"""
        emotional_state = ecg_context.emotional_state
        stress_level = ecg_context.stress_indicators.get('overall_stress', 0.5)
        confidence = decision.confidence_level
        
        # Quality factors
        emotional_clarity = 1.0
        if emotional_state == EmotionalState.STRESSED:
            emotional_clarity = 0.3
        elif emotional_state == EmotionalState.ANXIOUS:
            emotional_clarity = 0.4
        elif emotional_state == EmotionalState.CONFUSED:
            emotional_clarity = 0.2
        elif emotional_state in [EmotionalState.CALM, EmotionalState.CONFIDENT]:
            emotional_clarity = 0.9
        
        stress_factor = 1.0 - stress_level
        confidence_factor = confidence
        
        overall_quality = (emotional_clarity + stress_factor + confidence_factor) / 3
        
        return {
            'overall_quality': overall_quality,
            'emotional_clarity': emotional_clarity,
            'stress_factor': stress_factor,
            'confidence_factor': confidence_factor,
            'recommendation_strength': 'strong' if overall_quality > 0.7 else 'moderate' if overall_quality > 0.4 else 'weak'
        }
    
    def _generate_recommendation(self, decision: Decision, ecg_context: ECGContext, quality: Dict[str, float]) -> str:
        """Generate personalized recommendation based on analysis"""
        emotional_state = ecg_context.emotional_state
        stress_level = ecg_context.stress_indicators.get('stress_level', 'medium')
        quality_level = quality.get('recommendation_strength', 'moderate')
        
        if quality_level == 'weak':
            if emotional_state == EmotionalState.STRESSED:
                return "Consider postponing this decision until you're in a calmer state. Your current stress levels may impact decision quality."
            elif emotional_state == EmotionalState.ANXIOUS:
                return "Take time to process your emotions before finalizing this decision. Consider seeking additional input."
            else:
                return "Your current emotional state suggests revisiting this decision when you feel more centered."
        
        elif quality_level == 'moderate':
            if stress_level == 'high':
                return "Proceed with caution. Your elevated stress levels suggest taking extra time to consider all options."
            else:
                return "Your decision-making state is reasonable. Consider the alternatives provided for a well-rounded choice."
        
        else:  # Strong recommendation
            if emotional_state == EmotionalState.CONFIDENT:
                return "You're in an optimal state for decision-making. Trust your judgment while staying open to new information."
            elif emotional_state == EmotionalState.CALM:
                return "Your calm and centered state is ideal for making thoughtful decisions. Proceed with confidence."
            else:
                return "You're in a good state for decision-making. Your choice appears well-considered."
    
    def _calculate_confidence_score(self, decision: Decision, ecg_context: ECGContext, quality: Dict[str, float]) -> float:
        """Calculate overall confidence in the decision recommendation"""
        decision_confidence = decision.confidence_level
        ecg_confidence = ecg_context.sentiment_analysis.get('confidence', 0.5)
        quality_score = quality.get('overall_quality', 0.5)
        
        # Weight the components
        weighted_confidence = (
            decision_confidence * 0.4 +
            ecg_confidence * 0.3 +
            quality_score * 0.3
        )
        
        return min(max(weighted_confidence, 0.0), 1.0)
    
    def _generate_reasoning(self, decision: Decision, ecg_context: ECGContext, quality: Dict[str, float]) -> str:
        """Generate detailed reasoning for the recommendation"""
        emotional_state = ecg_context.emotional_state.value
        dominant_emotion = ecg_context.sentiment_analysis.get('dominant_emotion', 'neutral')
        stress_level = ecg_context.stress_indicators.get('stress_level', 'medium')
        heart_rate = ecg_context.physiological_markers.get('heart_rate', 70)
        
        reasoning_parts = [
            f"Based on your current emotional state ({emotional_state}) and dominant emotion ({dominant_emotion}),",
            f"combined with {stress_level} stress levels and a heart rate of {heart_rate:.0f} bpm,"
        ]
        
        if quality['overall_quality'] > 0.7:
            reasoning_parts.append("you appear to be in an optimal state for decision-making.")
        elif quality['overall_quality'] > 0.4:
            reasoning_parts.append("you're in a reasonable but not ideal state for complex decisions.")
        else:
            reasoning_parts.append("your current state may compromise decision quality.")
        
        if ecg_context.stress_indicators.get('overall_stress', 0) > 0.7:
            reasoning_parts.append("High stress levels detected may lead to impulsive or suboptimal choices.")
        
        return " ".join(reasoning_parts)
    
    def _generate_alternatives(self, decision: Decision, ecg_context: ECGContext) -> List[str]:
        """Generate alternative suggestions based on emotional state"""
        emotional_state = ecg_context.emotional_state
        alternatives = []
        
        if emotional_state == EmotionalState.STRESSED:
            alternatives.extend([
                "Take a 5-10 minute break to practice deep breathing",
                "Postpone the decision for 30 minutes to an hour",
                "Discuss the decision with a trusted friend or advisor"
            ])
        elif emotional_state == EmotionalState.ANXIOUS:
            alternatives.extend([
                "Write down pros and cons for each option",
                "Consider the worst-case scenario and how you'd handle it",
                "Break the decision into smaller, manageable parts"
            ])
        elif emotional_state == EmotionalState.EXCITED:
            alternatives.extend([
                "Channel your energy into thorough analysis",
                "Consider long-term implications beyond immediate excitement",
                "Verify your decision aligns with your core values"
            ])
        else:
            alternatives.extend([
                "Review the decision from multiple perspectives",
                "Consider timing and implementation strategies",
                "Document your reasoning for future reference"
            ])
        
        return alternatives
    
    def _assess_decision_risks(self, decision: Decision, ecg_context: ECGContext) -> Dict[str, float]:
        """Assess various risk factors associated with the decision"""
        emotional_state = ecg_context.emotional_state
        stress_level = ecg_context.stress_indicators.get('overall_stress', 0.5)
        
        risk_factors = {
            'emotional_bias_risk': 0.5,
            'stress_induced_error_risk': stress_level,
            'impulsivity_risk': 0.3,
            'analysis_paralysis_risk': 0.2
        }
        
        if emotional_state == EmotionalState.STRESSED:
            risk_factors['emotional_bias_risk'] = 0.8
            risk_factors['impulsivity_risk'] = 0.7
        elif emotional_state == EmotionalState.EXCITED:
            risk_factors['impulsivity_risk'] = 0.6
            risk_factors['emotional_bias_risk'] = 0.6
        elif emotional_state == EmotionalState.ANXIOUS:
            risk_factors['analysis_paralysis_risk'] = 0.7
            risk_factors['emotional_bias_risk'] = 0.7
        elif emotional_state == EmotionalState.CALM:
            risk_factors = {k: v * 0.5 for k, v in risk_factors.items()}
        
        # Overall risk assessment
        overall_risk = sum(risk_factors.values()) / len(risk_factors)
        risk_factors['overall_risk'] = overall_risk
        risk_factors['risk_level'] = 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low'
        
        return risk_factors
    
    def _create_fallback_results(self, decisions: List[Decision]) -> List[IntegratedDecisionResult]:
        """Create fallback results when ECG processing fails"""
        default_context = self._create_default_context()
        return [self._create_fallback_result(decision, default_context) for decision in decisions]
    
    def _create_fallback_result(self, decision: Decision, ecg_context: ECGContext) -> IntegratedDecisionResult:
        """Create a fallback result for a single decision"""
        return IntegratedDecisionResult(
            decision=decision,
            ecg_context=ecg_context,
            recommendation="Unable to analyze ECG data. Consider your emotional state and stress levels when making this decision.",
            confidence_score=0.5,
            reasoning="ECG analysis unavailable. Base decision on conscious self-assessment.",
            alternative_suggestions=["Take time to reflect on your current state", "Consider seeking input from others"],
            risk_assessment={'overall_risk': 0.5, 'risk_level': 'medium'}
        )
    
    def _create_default_context(self) -> ECGContext:
        """Create default ECG context when processing fails"""
        return ECGContext(
            sentiment_analysis={'neutral': 1.0, 'confidence': 0.0, 'dominant_emotion': 'neutral'},
            stress_indicators={'overall_stress': 0.5, 'stress_level': 'medium'},
            personal_patterns={},
            emotional_state=EmotionalState.CONFIDENT,
            physiological_markers={'heart_rate': 70, 'hrv_score': 30}
        )
    
    def get_decision_history_summary(self, user_id: int) -> Dict[str, Any]:
        """Get summary of user's decision-making patterns"""
        user_decisions = [result for result in self.decision_history 
                         if hasattr(result.decision, 'user_id') and result.decision.user_id == user_id]
        
        if not user_decisions:
            return {}
        
        # Analyze patterns
        emotional_states = [result.ecg_context.emotional_state.value for result in user_decisions]
        confidence_scores = [result.confidence_score for result in user_decisions]
        stress_levels = [result.ecg_context.stress_indicators.get('overall_stress', 0.5) for result in user_decisions]
        
        return {
            'total_decisions': len(user_decisions),
            'average_confidence': np.mean(confidence_scores),
            'average_stress_level': np.mean(stress_levels),
            'most_common_emotional_state': max(set(emotional_states), key=emotional_states.count),
            'decision_quality_trend': self._calculate_quality_trend(user_decisions),
            'recommendations': self._generate_personal_recommendations(user_decisions)
        }
    
    def _calculate_quality_trend(self, decisions: List[IntegratedDecisionResult]) -> str:
        """Calculate trend in decision quality over time"""
        if len(decisions) < 3:
            return "insufficient_data"
        
        recent_quality = np.mean([d.confidence_score for d in decisions[-5:]])
        earlier_quality = np.mean([d.confidence_score for d in decisions[:-5]])
        
        if recent_quality > earlier_quality + 0.1:
            return "improving"
        elif recent_quality < earlier_quality - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_personal_recommendations(self, decisions: List[IntegratedDecisionResult]) -> List[str]:
        """Generate personalized recommendations based on decision history"""
        recommendations = []
        
        avg_stress = np.mean([d.ecg_context.stress_indicators.get('overall_stress', 0.5) for d in decisions])
        if avg_stress > 0.7:
            recommendations.append("Consider stress management techniques before important decisions")
        
        emotional_states = [d.ecg_context.emotional_state for d in decisions]
        if emotional_states.count(EmotionalState.STRESSED) / len(emotional_states) > 0.3:
            recommendations.append("Practice relaxation techniques to improve decision-making quality")
        
        avg_confidence = np.mean([d.confidence_score for d in decisions])
        if avg_confidence < 0.5:
            recommendations.append("Consider taking more time to analyze decisions or seeking additional input")
        
        return recommendations