from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.database import get_db
from app.database.models import ECGRecord, User
from app.auth.dependencies import get_current_user
from app.services.decision_engine import DecisionIntegrationEngine, Decision, DecisionType
from app.services.ai_integration import ChatGPTIntegrationService
from app.services.personal_profile import PersonalProfileLearningSystem

router = APIRouter(prefix="/decision-analysis", tags=["Decision Analysis"])

# Pydantic models for API
class DecisionInput(BaseModel):
    question: str
    options: List[str]
    decision_type: str  # "binary", "multiple_choice", "scaled", "open_ended"
    user_response: Any
    confidence_level: float = Field(ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = None

class DecisionAnalysisRequest(BaseModel):
    ecg_record_id: int
    decisions: List[DecisionInput]

class PersonalizedInsightsRequest(BaseModel):
    timeframe_days: int = Field(default=30, ge=1, le=365)
    include_recommendations: bool = True
    include_ai_analysis: bool = True

class DecisionAnalysisResponse(BaseModel):
    session_id: str
    analysis_results: List[Dict[str, Any]]
    ai_insights: Dict[str, Any]
    personal_recommendations: Dict[str, Any]
    summary: str

# Initialize services
decision_engine = DecisionIntegrationEngine()
ai_service = ChatGPTIntegrationService()
profile_system = PersonalProfileLearningSystem()

@router.post("/analyze", response_model=DecisionAnalysisResponse)
async def analyze_decisions_with_ecg(
    request: DecisionAnalysisRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze decisions in context of ECG emotional state"""
    
    # Get ECG record
    ecg_record = (
        db.query(ECGRecord)
        .filter(ECGRecord.id == request.ecg_record_id, ECGRecord.user_id == current_user.id)
        .first()
    )
    
    if not ecg_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ECG record not found"
        )
    
    if not ecg_record.raw_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ECG record has no raw data for analysis"
        )
    
    # Convert request decisions to Decision objects
    decisions = []
    for i, decision_input in enumerate(request.decisions):
        try:
            decision_type = DecisionType(decision_input.decision_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid decision type: {decision_input.decision_type}"
            )
        
        decision = Decision(
            id=f"decision_{i}_{ecg_record.id}",
            question=decision_input.question,
            options=decision_input.options,
            decision_type=decision_type,
            user_response=decision_input.user_response,
            confidence_level=decision_input.confidence_level,
            timestamp=datetime.now(),
            context=decision_input.context
        )
        decisions.append(decision)
    
    # Analyze decisions with ECG context
    decision_results = decision_engine.analyze_decision_context(ecg_record.raw_data, decisions)
    
    # Generate AI insights
    user_context = {
        'user_id': current_user.id,
        'session_time': datetime.now().isoformat(),
        'ecg_record_time': ecg_record.recorded_at.isoformat()
    }
    ai_insights = ai_service.generate_comprehensive_analysis(decision_results, user_context)
    
    # Get personal recommendations
    personal_recommendations = profile_system.get_personalized_recommendations(current_user.id)
    
    # Create session ID
    session_id = f"session_{current_user.id}_{int(datetime.now().timestamp())}"
    
    # Store processed data in ECG record
    processed_data = {
        'session_id': session_id,
        'decision_analysis': [
            {
                'decision': {
                    'question': result.decision.question,
                    'user_response': result.decision.user_response,
                    'confidence': result.decision.confidence_level
                },
                'recommendation': result.recommendation,
                'confidence_score': result.confidence_score,
                'emotional_context': {
                    'emotional_state': result.ecg_context.emotional_state.value,
                    'stress_level': result.ecg_context.stress_indicators.get('stress_level')
                }
            }
            for result in decision_results
        ],
        'ai_insights_summary': ai_insights.get('summary', ''),
        'emotional_intelligence_score': ai_insights.get('emotional_intelligence_score', 0.5)
    }
    
    ecg_record.processed_data = processed_data
    ecg_record.is_processed = True
    db.commit()
    
    # Format response
    analysis_results = []
    for result in decision_results:
        analysis_results.append({
            'decision': {
                'question': result.decision.question,
                'options': result.decision.options,
                'user_response': result.decision.user_response,
                'confidence': result.decision.confidence_level
            },
            'ecg_context': {
                'emotional_state': result.ecg_context.emotional_state.value,
                'dominant_emotion': result.ecg_context.sentiment_analysis.get('dominant_emotion'),
                'stress_level': result.ecg_context.stress_indicators.get('stress_level'),
                'physiological_markers': result.ecg_context.physiological_markers
            },
            'recommendation': result.recommendation,
            'confidence_score': result.confidence_score,
            'reasoning': result.reasoning,
            'alternatives': result.alternative_suggestions,
            'risk_assessment': result.risk_assessment
        })
    
    return DecisionAnalysisResponse(
        session_id=session_id,
        analysis_results=analysis_results,
        ai_insights=ai_insights,
        personal_recommendations=personal_recommendations,
        summary=ai_insights.get('summary', 'Analysis completed successfully')
    )

@router.get("/personal-insights")
async def get_personal_insights(
    request: PersonalizedInsightsRequest = Depends(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized insights based on user's ECG and decision history"""
    
    # Get user's ECG history
    cutoff_date = datetime.now() - timedelta(days=request.timeframe_days)
    ecg_records = (
        db.query(ECGRecord)
        .filter(
            ECGRecord.user_id == current_user.id,
            ECGRecord.recorded_at >= cutoff_date
        )
        .order_by(ECGRecord.recorded_at.desc())
        .all()
    )
    
    if not ecg_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ECG records found for the specified timeframe"
        )
    
    # Convert to list of dicts for processing
    ecg_history = []
    decision_history = []
    
    for record in ecg_records:
        ecg_data = {
            'recorded_at': record.recorded_at,
            'raw_data': record.raw_data,
            'processed_data': record.processed_data
        }
        ecg_history.append(ecg_data)
        
        # Extract decision history from processed data
        if record.processed_data and 'decision_analysis' in record.processed_data:
            # Note: This is simplified - in a real implementation, you'd store decision results separately
            pass
    
    # Analyze user patterns and update profile
    user_profile = profile_system.analyze_user_patterns(
        current_user.id, 
        ecg_history, 
        decision_history,  # Empty for now, would need separate decision storage
        request.timeframe_days
    )
    
    # Get recommendations
    recommendations = profile_system.get_personalized_recommendations(current_user.id)
    
    # Get learning insights
    insights = profile_system.learning_insights.get(current_user.id, [])
    
    response = {
        'user_profile': {
            'stress_reactivity': user_profile.stress_reactivity,
            'emotional_stability': user_profile.emotional_stability,
            'decision_confidence': user_profile.decision_confidence,
            'stress_recovery': user_profile.stress_recovery,
            'dominant_emotions': user_profile.dominant_emotions,
            'strengths': user_profile.strengths,
            'improvement_areas': user_profile.improvement_areas,
            'last_updated': user_profile.last_updated.isoformat()
        },
        'learning_insights': [
            {
                'type': insight.insight_type,
                'description': insight.description,
                'confidence': insight.confidence,
                'priority': insight.priority,
                'actionable_steps': insight.actionable_steps,
                'created_at': insight.created_at.isoformat()
            }
            for insight in insights
        ],
        'recommendations': recommendations,
        'statistics': {
            'total_ecg_records': len(ecg_records),
            'timeframe_days': request.timeframe_days,
            'analysis_date': datetime.now().isoformat()
        }
    }
    
    # Add AI analysis if requested
    if request.include_ai_analysis and ecg_history:
        # Create mock decision results for AI analysis
        # In real implementation, you'd have stored decision results
        ai_context = {
            'user_id': current_user.id,
            'profile_summary': user_profile.__dict__,
            'timeframe': request.timeframe_days
        }
        
        # Generate AI insights about the user's patterns
        ai_insights = {
            'pattern_analysis': f"Based on {len(ecg_records)} ECG records over {request.timeframe_days} days",
            'emotional_trends': user_profile.dominant_emotions,
            'improvement_suggestions': user_profile.improvement_areas,
            'strengths_analysis': user_profile.strengths
        }
        response['ai_insights'] = ai_insights
    
    return response

@router.get("/decision-history")
async def get_decision_history(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's decision-making history with analysis"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    ecg_records = (
        db.query(ECGRecord)
        .filter(
            ECGRecord.user_id == current_user.id,
            ECGRecord.recorded_at >= cutoff_date,
            ECGRecord.processed_data.isnot(None)
        )
        .order_by(ECGRecord.recorded_at.desc())
        .all()
    )
    
    decision_sessions = []
    for record in ecg_records:
        if record.processed_data and 'decision_analysis' in record.processed_data:
            session_data = {
                'session_id': record.processed_data.get('session_id'),
                'date': record.recorded_at.isoformat(),
                'decisions': record.processed_data['decision_analysis'],
                'emotional_intelligence_score': record.processed_data.get('emotional_intelligence_score'),
                'summary': record.processed_data.get('ai_insights_summary', '')
            }
            decision_sessions.append(session_data)
    
    # Calculate summary statistics
    if decision_sessions:
        total_decisions = sum(len(session['decisions']) for session in decision_sessions)
        avg_ei_score = sum(session.get('emotional_intelligence_score', 0.5) for session in decision_sessions) / len(decision_sessions)
        
        # Extract confidence scores
        all_confidence_scores = []
        for session in decision_sessions:
            for decision in session['decisions']:
                all_confidence_scores.append(decision.get('confidence_score', 0.5))
        
        avg_confidence = sum(all_confidence_scores) / len(all_confidence_scores) if all_confidence_scores else 0.5
    else:
        total_decisions = 0
        avg_ei_score = 0.5
        avg_confidence = 0.5
    
    return {
        'decision_sessions': decision_sessions,
        'summary_statistics': {
            'total_sessions': len(decision_sessions),
            'total_decisions': total_decisions,
            'average_emotional_intelligence': avg_ei_score,
            'average_decision_confidence': avg_confidence,
            'timeframe_days': days
        }
    }

@router.post("/compare-baseline")
async def compare_with_baseline(
    ecg_record_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compare current ECG metrics with user's baseline"""
    
    # Get ECG record
    ecg_record = (
        db.query(ECGRecord)
        .filter(ECGRecord.id == ecg_record_id, ECGRecord.user_id == current_user.id)
        .first()
    )
    
    if not ecg_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="ECG record not found"
        )
    
    # Extract current metrics from ECG record
    current_metrics = {}
    if ecg_record.processed_data:
        # Extract relevant metrics
        current_metrics = {
            'heart_rate': 70,  # Would extract from processed data
            'stress_level': 0.5,  # Would extract from processed data
            'hrv_score': 30  # Would extract from processed data
        }
    
    # Compare with baseline
    comparison = profile_system.compare_with_baseline(current_user.id, current_metrics)
    
    if 'error' in comparison:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=comparison['error']
        )
    
    return {
        'ecg_record_id': ecg_record_id,
        'comparison_date': datetime.now().isoformat(),
        'baseline_comparison': comparison,
        'recommendations': profile_system.get_personalized_recommendations(current_user.id)
    }