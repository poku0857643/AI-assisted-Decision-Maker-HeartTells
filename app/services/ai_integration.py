import openai
from typing import Dict, List, Any, Optional
import json
from dataclasses import asdict
from app.services.decision_engine import IntegratedDecisionResult, Decision, ECGContext
from app.config.config import settings

class ChatGPTIntegrationService:
    """Service for integrating ECG analysis and decision-making with ChatGPT"""
    
    def __init__(self):
        # Use the CHATGPT_API_KEY from settings (should be OpenAI API key)
        openai.api_key = settings.CHATGPT_API_KEY
        self.client = openai.OpenAI(api_key=settings.CHATGPT_API_KEY) if settings.CHATGPT_API_KEY else None
    
    def generate_comprehensive_analysis(self, 
                                      decision_results: List[IntegratedDecisionResult],
                                      user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive analysis combining ECG data and decisions using ChatGPT"""
        
        if not self.client:
            return self._generate_fallback_analysis(decision_results)
        
        try:
            # Prepare context for ChatGPT
            analysis_context = self._prepare_analysis_context(decision_results, user_context)
            
            # Generate primary analysis
            primary_analysis = self._generate_primary_analysis(analysis_context)
            
            # Generate actionable insights
            actionable_insights = self._generate_actionable_insights(analysis_context, primary_analysis)
            
            # Generate personalized recommendations
            personalized_recommendations = self._generate_personalized_recommendations(analysis_context)
            
            # Generate risk assessment
            risk_assessment = self._generate_risk_assessment(analysis_context)
            
            return {
                'primary_analysis': primary_analysis,
                'actionable_insights': actionable_insights,
                'personalized_recommendations': personalized_recommendations,
                'risk_assessment': risk_assessment,
                'emotional_intelligence_score': self._calculate_ei_score(decision_results),
                'decision_quality_metrics': self._calculate_decision_metrics(decision_results),
                'summary': self._generate_executive_summary(primary_analysis, actionable_insights)
            }
            
        except Exception as e:
            print(f"Error generating ChatGPT analysis: {e}")
            return self._generate_fallback_analysis(decision_results)
    
    def _prepare_analysis_context(self, 
                                decision_results: List[IntegratedDecisionResult],
                                user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Prepare structured context for ChatGPT analysis"""
        
        # Extract ECG patterns
        ecg_patterns = []
        decision_patterns = []
        
        for result in decision_results:
            # ECG context
            ecg_data = {
                'emotional_state': result.ecg_context.emotional_state.value,
                'dominant_emotion': result.ecg_context.sentiment_analysis.get('dominant_emotion'),
                'stress_level': result.ecg_context.stress_indicators.get('stress_level'),
                'overall_stress': result.ecg_context.stress_indicators.get('overall_stress'),
                'heart_rate': result.ecg_context.physiological_markers.get('heart_rate'),
                'autonomic_balance': result.ecg_context.physiological_markers.get('autonomic_balance')
            }
            ecg_patterns.append(ecg_data)
            
            # Decision context
            decision_data = {
                'question': result.decision.question,
                'decision_type': result.decision.decision_type.value,
                'user_confidence': result.decision.confidence_level,
                'recommendation': result.recommendation,
                'confidence_score': result.confidence_score,
                'risk_level': result.risk_assessment.get('risk_level')
            }
            decision_patterns.append(decision_data)
        
        return {
            'ecg_patterns': ecg_patterns,
            'decision_patterns': decision_patterns,
            'user_context': user_context or {},
            'session_summary': {
                'total_decisions': len(decision_results),
                'average_stress': sum(p['overall_stress'] for p in ecg_patterns) / len(ecg_patterns),
                'dominant_emotions': [p['dominant_emotion'] for p in ecg_patterns],
                'decision_confidence': sum(p['user_confidence'] for p in decision_patterns) / len(decision_patterns)
            }
        }
    
    def _generate_primary_analysis(self, context: Dict[str, Any]) -> str:
        """Generate primary analysis using ChatGPT"""
        try:
            prompt = self._create_primary_analysis_prompt(context)
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in psychophysiology and decision science, specializing in ECG-based emotional analysis and decision-making optimization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating primary analysis: {e}")
            return "Unable to generate detailed analysis. Please review the individual decision recommendations."
    
    def _create_primary_analysis_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for primary analysis"""
        ecg_summary = context['session_summary']
        emotions = ', '.join(set(ecg_summary['dominant_emotions']))
        
        prompt = f"""
        Analyze the following decision-making session that combines ECG-based emotional analysis with user decisions:

        SESSION OVERVIEW:
        - Total decisions made: {ecg_summary['total_decisions']}
        - Average stress level: {ecg_summary['average_stress']:.2f} (0-1 scale)
        - Detected emotions: {emotions}
        - Average user confidence: {ecg_summary['decision_confidence']:.2f} (0-1 scale)

        ECG PATTERNS:
        {json.dumps(context['ecg_patterns'], indent=2)}

        DECISION PATTERNS:
        {json.dumps(context['decision_patterns'], indent=2)}

        Please provide a comprehensive analysis that:
        1. Identifies patterns between emotional states and decision quality
        2. Assesses the coherence between physiological indicators and conscious decision-making
        3. Highlights any concerning patterns or positive trends
        4. Explains how the user's emotional state influenced their decision-making process
        5. Provides insights into their decision-making style and emotional intelligence

        Focus on actionable insights that can help improve future decision-making.
        """
        
        return prompt
    
    def _generate_actionable_insights(self, context: Dict[str, Any], primary_analysis: str) -> List[str]:
        """Generate specific actionable insights"""
        try:
            prompt = f"""
            Based on this ECG-decision analysis:
            {primary_analysis}
            
            And this session data:
            {json.dumps(context['session_summary'], indent=2)}
            
            Generate 5-7 specific, actionable insights that the user can implement to improve their decision-making. 
            Each insight should be:
            - Specific and concrete
            - Based on the physiological and emotional patterns observed
            - Implementable in daily life
            - Focused on improving decision quality and emotional regulation
            
            Format as a simple list.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a practical advisor specializing in translating psychophysiological insights into actionable daily practices."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.6
            )
            
            # Parse response into list
            insights_text = response.choices[0].message.content
            insights = [insight.strip() for insight in insights_text.split('\n') if insight.strip() and not insight.strip().startswith('#')]
            return insights[:7]  # Limit to 7 insights
            
        except Exception as e:
            print(f"Error generating actionable insights: {e}")
            return self._generate_fallback_insights(context)
    
    def _generate_personalized_recommendations(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate personalized recommendations by category"""
        try:
            prompt = f"""
            Based on this user's ECG and decision-making patterns:
            {json.dumps(context, indent=2)}
            
            Generate personalized recommendations in these categories:
            1. Stress Management (3-4 recommendations)
            2. Decision-Making Strategies (3-4 recommendations)  
            3. Emotional Regulation (3-4 recommendations)
            4. Lifestyle Optimization (2-3 recommendations)
            
            Each recommendation should be:
            - Tailored to the observed patterns
            - Scientifically grounded
            - Practically implementable
            - Specific to their emotional and physiological profile
            
            Format as JSON with categories as keys and lists of recommendations as values.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a personalized wellness coach with expertise in psychophysiology and behavioral science."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.6
            )
            
            # Try to parse as JSON, fallback to structured text parsing
            try:
                recommendations = json.loads(response.choices[0].message.content)
                return recommendations
            except json.JSONDecodeError:
                return self._parse_recommendations_from_text(response.choices[0].message.content)
            
        except Exception as e:
            print(f"Error generating personalized recommendations: {e}")
            return self._generate_fallback_recommendations()
    
    def _generate_risk_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        try:
            avg_stress = context['session_summary']['average_stress']
            emotions = context['session_summary']['dominant_emotions']
            
            # Calculate risk factors
            stress_risk = "high" if avg_stress > 0.7 else "medium" if avg_stress > 0.4 else "low"
            emotional_volatility = len(set(emotions)) / len(emotions) if emotions else 0
            
            prompt = f"""
            Assess the decision-making risks for a user with these patterns:
            - Average stress level: {avg_stress:.2f}
            - Emotional patterns: {emotions}
            - Emotional volatility: {emotional_volatility:.2f}
            
            Provide risk assessment for:
            1. Cognitive bias risks
            2. Stress-induced decision errors
            3. Emotional decision-making risks
            4. Long-term decision pattern concerns
            
            For each risk category, provide:
            - Risk level (low/medium/high)
            - Specific concerns based on the data
            - Mitigation strategies
            
            Format as structured text.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a risk assessment specialist focusing on decision-making and emotional regulation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.5
            )
            
            return {
                'overall_risk_level': stress_risk,
                'detailed_assessment': response.choices[0].message.content,
                'key_risk_factors': self._extract_risk_factors(context),
                'mitigation_priority': self._determine_mitigation_priority(context)
            }
            
        except Exception as e:
            print(f"Error generating risk assessment: {e}")
            return {'overall_risk_level': 'medium', 'detailed_assessment': 'Risk assessment unavailable'}
    
    def _generate_executive_summary(self, primary_analysis: str, actionable_insights: List[str]) -> str:
        """Generate executive summary"""
        try:
            prompt = f"""
            Create a concise executive summary (3-4 sentences) based on:
            
            Analysis: {primary_analysis[:500]}...
            Key Insights: {'; '.join(actionable_insights[:3])}
            
            The summary should highlight the most important findings and immediate action items.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an executive communications specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.5
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            return "Analysis completed. Review detailed recommendations for decision-making improvements."
    
    def _calculate_ei_score(self, decision_results: List[IntegratedDecisionResult]) -> float:
        """Calculate emotional intelligence score based on decision patterns"""
        if not decision_results:
            return 0.5
        
        # Factors for EI score
        stress_management = 1 - sum(r.ecg_context.stress_indicators.get('overall_stress', 0.5) for r in decision_results) / len(decision_results)
        emotional_awareness = sum(r.ecg_context.sentiment_analysis.get('confidence', 0.5) for r in decision_results) / len(decision_results)
        decision_quality = sum(r.confidence_score for r in decision_results) / len(decision_results)
        
        ei_score = (stress_management * 0.4 + emotional_awareness * 0.3 + decision_quality * 0.3)
        return min(max(ei_score, 0.0), 1.0)
    
    def _calculate_decision_metrics(self, decision_results: List[IntegratedDecisionResult]) -> Dict[str, float]:
        """Calculate decision quality metrics"""
        if not decision_results:
            return {}
        
        return {
            'average_confidence': sum(r.confidence_score for r in decision_results) / len(decision_results),
            'stress_impact': sum(r.ecg_context.stress_indicators.get('overall_stress', 0.5) for r in decision_results) / len(decision_results),
            'emotional_alignment': sum(1 for r in decision_results if r.ecg_context.emotional_state.value in ['calm', 'confident']) / len(decision_results),
            'decision_consistency': self._calculate_consistency(decision_results)
        }
    
    def _calculate_consistency(self, decision_results: List[IntegratedDecisionResult]) -> float:
        """Calculate decision consistency score"""
        if len(decision_results) < 2:
            return 1.0
        
        confidence_scores = [r.confidence_score for r in decision_results]
        consistency = 1.0 - (max(confidence_scores) - min(confidence_scores))
        return max(consistency, 0.0)
    
    def _generate_fallback_analysis(self, decision_results: List[IntegratedDecisionResult]) -> Dict[str, Any]:
        """Generate fallback analysis when ChatGPT is unavailable"""
        return {
            'primary_analysis': "ChatGPT integration unavailable. Basic analysis: Review individual decision recommendations and ECG insights.",
            'actionable_insights': self._generate_fallback_insights({'decision_results': decision_results}),
            'personalized_recommendations': self._generate_fallback_recommendations(),
            'risk_assessment': {'overall_risk_level': 'medium', 'detailed_assessment': 'Unable to assess risks without AI analysis'},
            'emotional_intelligence_score': self._calculate_ei_score(decision_results),
            'decision_quality_metrics': self._calculate_decision_metrics(decision_results),
            'summary': "Analysis completed with limited AI integration. Review individual recommendations."
        }
    
    def _generate_fallback_insights(self, context: Dict[str, Any]) -> List[str]:
        """Generate basic insights without ChatGPT"""
        return [
            "Monitor your stress levels before making important decisions",
            "Take breaks when feeling overwhelmed or anxious",
            "Consider your emotional state when evaluating options",
            "Practice mindfulness to improve decision-making clarity",
            "Seek input from others when confidence is low"
        ]
    
    def _generate_fallback_recommendations(self) -> Dict[str, List[str]]:
        """Generate basic recommendations without ChatGPT"""
        return {
            'stress_management': [
                "Practice deep breathing exercises",
                "Take regular breaks during decision-making",
                "Use progressive muscle relaxation"
            ],
            'decision_strategies': [
                "List pros and cons for important decisions",
                "Set decision deadlines to avoid overthinking",
                "Consider long-term consequences"
            ],
            'emotional_regulation': [
                "Practice mindfulness meditation",
                "Keep an emotion diary",
                "Identify emotional triggers"
            ],
            'lifestyle': [
                "Maintain regular exercise routine",
                "Ensure adequate sleep"
            ]
        }
    
    def _parse_recommendations_from_text(self, text: str) -> Dict[str, List[str]]:
        """Parse recommendations from text format"""
        # Basic text parsing implementation
        categories = {}
        current_category = None
        
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line and not line.startswith('-'):
                current_category = line.split(':')[0].lower().replace(' ', '_')
                categories[current_category] = []
            elif line.startswith('-') and current_category:
                categories[current_category].append(line[1:].strip())
        
        return categories if categories else self._generate_fallback_recommendations()
    
    def _extract_risk_factors(self, context: Dict[str, Any]) -> List[str]:
        """Extract key risk factors from context"""
        risk_factors = []
        avg_stress = context['session_summary']['average_stress']
        
        if avg_stress > 0.7:
            risk_factors.append("High stress levels affecting decision quality")
        
        emotions = context['session_summary']['dominant_emotions']
        negative_emotions = [e for e in emotions if e in ['angry', 'fearful', 'sad', 'anxious']]
        if len(negative_emotions) > len(emotions) / 2:
            risk_factors.append("Predominantly negative emotional states")
        
        return risk_factors
    
    def _determine_mitigation_priority(self, context: Dict[str, Any]) -> str:
        """Determine priority for risk mitigation"""
        avg_stress = context['session_summary']['average_stress']
        
        if avg_stress > 0.8:
            return "immediate"
        elif avg_stress > 0.6:
            return "high"
        elif avg_stress > 0.4:
            return "medium"
        else:
            return "low"