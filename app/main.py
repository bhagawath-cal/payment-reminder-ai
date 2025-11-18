"""
Payment Reminder AI Service - FastAPI
Using Supervised Machine Learning for Bill Categorization
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import joblib
import pickle
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Payment Reminder AI Service",
    description="AI-powered bill categorization and payment risk prediction",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# LOAD TRAINED MODEL
# ============================================

MODEL_PATH = Path("bill_categorization_model.pkl")
CATEGORIES_PATH = Path("categories.pkl")

try:
    # Load the trained scikit-learn pipeline
    categorization_model = joblib.load(MODEL_PATH)
    
    # Load categories list
    with open(CATEGORIES_PATH, 'rb') as f:
        categories_list = pickle.load(f)
    
    print("✅ Machine Learning model loaded successfully!")
    print(f"✅ Categories: {categories_list}")
except FileNotFoundError:
    print("⚠️ WARNING: ML model not found. Using rule-based system.")
    categorization_model = None
    categories_list = [
        "Utilities", "Housing", "Transportation", "Internet & Phone",
        "Insurance", "Subscriptions", "Food & Groceries", "Healthcare",
        "Education", "Entertainment", "Other"
    ]

# Initialize Agent
try:
    from .agent import PaymentReminderAgent
    agent = PaymentReminderAgent()
    print("✅ Payment Reminder Agent initialized successfully!")
except Exception as e:
    print(f"⚠️ WARNING: Agent initialization failed: {e}")
    agent = None

# ============================================
# PYDANTIC MODELS
# ============================================

class BillCategorizationRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    amount: Optional[float] = 0.0

class BillCategorizationResponse(BaseModel):
    category: str
    confidence: float
    reasoning: str
    all_probabilities: Optional[dict] = None

class PaymentPredictionRequest(BaseModel):
    billName: str
    amount: float
    dueDate: str
    category: str
    paymentHistory: List[dict] = []

class PaymentPredictionResponse(BaseModel):
    riskLevel: str
    latePaymentProbability: float
    recommendedReminderDate: str
    reasoning: str

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    model_loaded: bool
    agent_loaded: bool
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    use_memory: bool = True

class ChatResponse(BaseModel):
    response: str

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Payment Reminder AI Service",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "categorize": "/categorize-bill",
            "predict": "/predict-payment",
            "test": "/test-categorization",
            "categories": "/categories",
            "chat": "/chat",
            "tools": "/tools"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        service="Payment Reminder AI",
        version="2.0.0",
        model_loaded=categorization_model is not None,
        agent_loaded=agent is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/categories")
async def get_categories():
    """Get all available bill categories"""
    return {
        "categories": categories_list,
        "count": len(categories_list)
    }

@app.post("/test-categorization")
async def test_categorization():
    """Test the categorization system with sample bills"""
    
    test_bills = [
        {"name": "Netflix Premium", "description": "Streaming service", "amount": 19.99},
        {"name": "ComEd Electric Bill", "description": "Monthly electricity", "amount": 125.50},
        {"name": "Apartment Rent", "description": "Monthly rent payment", "amount": 1500.00},
        {"name": "Verizon Wireless", "description": "Cell phone bill", "amount": 85.00},
        {"name": "State Farm Insurance", "description": "Auto insurance", "amount": 150.00},
        {"name": "Spotify Premium", "description": "Music streaming", "amount": 10.99},
        {"name": "Whole Foods", "description": "Grocery shopping", "amount": 85.50},
        {"name": "CVS Pharmacy", "description": "Prescription pickup", "amount": 25.00},
    ]
    
    results = []
    
    for bill in test_bills:
        request = BillCategorizationRequest(**bill)
        
        if categorization_model is None:
            response = _fallback_categorization(request)
        else:
            try:
                text_input = f"{request.name} {request.description}".strip()
                predicted_category = categorization_model.predict([text_input])[0]
                probabilities = categorization_model.predict_proba([text_input])[0]
                confidence = float(max(probabilities))
                
                response = BillCategorizationResponse(
                    category=predicted_category,
                    confidence=confidence,
                    reasoning=f"ML classification with {confidence:.1%} confidence"
                )
            except:
                response = _fallback_categorization(request)
        
        results.append({
            "bill_name": bill["name"],
            "predicted_category": response.category,
            "confidence": response.confidence,
            "amount": bill["amount"]
        })
    
    return {
        "test_results": results,
        "total_tests": len(results),
        "status": "Success",
        "model_used": "ML Model" if categorization_model else "Rule-based"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Intelligent chat endpoint using Nuscale LLM
    Can answer questions about bills, payments, and provide assistance
    """
    
    if not agent:
        return ChatResponse(
            response="Chat agent is not available. Please check the agent configuration."
        )
    
    try:
        # Use the agent to process the message
        response_text = agent.chat(request.message, request.use_memory)
        return ChatResponse(response=response_text)
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return ChatResponse(
            response=f"I encountered an error: {str(e)}. Please try again."
        )

@app.post("/clear-chat-history")
async def clear_chat_history():
    """Clear conversation history"""
    if agent:
        agent.clear_history()
        return {"message": "Chat history cleared", "status": "success"}
    return {"message": "Agent not available", "status": "error"}

@app.get("/tools")
async def get_tools():
    """Get list of available tools"""
    if agent:
        return {"tools": agent.get_available_tools()}
    return {"tools": []}

@app.post("/categorize-bill", response_model=BillCategorizationResponse)
async def categorize_bill(request: BillCategorizationRequest):
    """
    Categorize a bill using supervised machine learning
    
    Uses a trained Multinomial Naive Bayes classifier with TF-IDF features
    """
    
    if categorization_model is None:
        # Fallback to rule-based if model not loaded
        return _fallback_categorization(request)
    
    try:
        # Combine name and description for better accuracy
        text_input = f"{request.name} {request.description}".strip()
        
        # Get prediction from ML model
        predicted_category = categorization_model.predict([text_input])[0]
        
        # Get probability scores for all categories
        probabilities = categorization_model.predict_proba([text_input])[0]
        
        # Get confidence (probability of predicted class)
        confidence = float(max(probabilities))
        
        # Create probability dictionary
        prob_dict = {
            cat: float(prob) 
            for cat, prob in zip(categories_list, probabilities)
        }
        
        # Sort by probability
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_probs[:3]
        
        # Generate reasoning
        reasoning = _generate_reasoning(
            request.name, 
            predicted_category, 
            confidence,
            top_3
        )
        
        return BillCategorizationResponse(
            category=predicted_category,
            confidence=confidence,
            reasoning=reasoning,
            all_probabilities=prob_dict
        )
        
    except Exception as e:
        print(f"Error in categorization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-payment", response_model=PaymentPredictionResponse)
async def predict_payment_risk(request: PaymentPredictionRequest):
    """
    Predict payment behavior and risk of late payment
    """
    
    try:
        # Parse due date
        due_date = datetime.fromisoformat(request.dueDate.replace('Z', '+00:00'))
        days_until_due = (due_date - datetime.now()).days
        
        # Calculate risk factors
        amount_risk = _calculate_amount_risk(request.amount)
        timing_risk = _calculate_timing_risk(days_until_due)
        category_risk = _calculate_category_risk(request.category)
        history_risk = _calculate_history_risk(request.paymentHistory)
        
        # Weighted risk calculation
        total_risk = (
            amount_risk * 0.3 +
            timing_risk * 0.3 +
            category_risk * 0.2 +
            history_risk * 0.2
        )
        
        # Determine risk level
        if total_risk < 0.3:
            risk_level = "LOW"
            reminder_days_before = 3
        elif total_risk < 0.6:
            risk_level = "MEDIUM"
            reminder_days_before = 5
        else:
            risk_level = "HIGH"
            reminder_days_before = 7
        
        # Calculate recommended reminder date
        reminder_date = due_date - timedelta(days=reminder_days_before)
        
        # Generate reasoning
        reasoning = _generate_risk_reasoning(
            request.billName,
            request.amount,
            days_until_due,
            risk_level,
            total_risk
        )
        
        return PaymentPredictionResponse(
            riskLevel=risk_level,
            latePaymentProbability=round(total_risk, 2),
            recommendedReminderDate=reminder_date.isoformat(),
            reasoning=reasoning
        )
        
    except Exception as e:
        print(f"Error in payment prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# HELPER FUNCTIONS
# ============================================

def _fallback_categorization(request: BillCategorizationRequest) -> BillCategorizationResponse:
    """Fallback rule-based categorization if ML model not available"""
    
    name_lower = request.name.lower()
    
    # Rule-based classification
    if any(word in name_lower for word in ['netflix', 'spotify', 'hulu', 'disney', 'subscription', 'prime', 'youtube']):
        category = "Subscriptions"
        confidence = 0.95
    elif any(word in name_lower for word in ['electric', 'gas', 'water', 'utility', 'power', 'energy', 'comed']):
        category = "Utilities"
        confidence = 0.85
    elif any(word in name_lower for word in ['rent', 'mortgage', 'hoa', 'lease', 'property', 'apartment']):
        category = "Housing"
        confidence = 0.90
    elif any(word in name_lower for word in ['internet', 'phone', 'wifi', 'mobile', 'cable', 'comcast', 'verizon', 'att', 'wireless']):
        category = "Internet & Phone"
        confidence = 0.90
    elif any(word in name_lower for word in ['insurance', 'geico', 'state farm', 'allstate', 'progressive']):
        category = "Insurance"
        confidence = 0.85
    elif any(word in name_lower for word in ['car', 'auto', 'vehicle', 'fuel', 'parking', 'uber', 'lyft']):
        category = "Transportation"
        confidence = 0.80
    elif any(word in name_lower for word in ['grocery', 'food', 'restaurant', 'walmart', 'costco', 'doordash', 'ubereats', 'whole foods']):
        category = "Food & Groceries"
        confidence = 0.80
    elif any(word in name_lower for word in ['doctor', 'hospital', 'medical', 'health', 'prescription', 'pharmacy', 'dental', 'cvs', 'walgreens']):
        category = "Healthcare"
        confidence = 0.85
    elif any(word in name_lower for word in ['tuition', 'school', 'college', 'university', 'education', 'course']):
        category = "Education"
        confidence = 0.80
    elif any(word in name_lower for word in ['movie', 'concert', 'game', 'entertainment', 'ticket', 'hobby']):
        category = "Entertainment"
        confidence = 0.75
    else:
        category = "Other"
        confidence = 0.60
    
    reasoning = f"Rule-based classification: '{category}' based on keyword matching"
    
    return BillCategorizationResponse(
        category=category,
        confidence=confidence,
        reasoning=reasoning
    )

def _generate_reasoning(name: str, category: str, confidence: float, top_3: list) -> str:
    """Generate human-readable reasoning for categorization"""
    
    reasoning_parts = []
    
    # Main prediction
    reasoning_parts.append(f"ML model classified as '{category}' ({confidence:.1%} confidence)")
    
    # Mention alternatives if confidence is not very high
    if confidence < 0.9 and len(top_3) > 1:
        alt_cat, alt_prob = top_3[1]
        reasoning_parts.append(f"Alternative: '{alt_cat}' ({alt_prob:.1%})")
    
    return ". ".join(reasoning_parts)

def _calculate_amount_risk(amount: float) -> float:
    if amount < 50:
        return 0.1
    elif amount < 200:
        return 0.3
    elif amount < 500:
        return 0.5
    else:
        return 0.7

def _calculate_timing_risk(days_until_due: int) -> float:
    if days_until_due < 0:
        return 1.0
    elif days_until_due < 3:
        return 0.8
    elif days_until_due < 7:
        return 0.5
    elif days_until_due < 14:
        return 0.3
    else:
        return 0.1

def _calculate_category_risk(category: str) -> float:
    high_priority = ["Utilities", "Housing", "Insurance"]
    medium_priority = ["Internet & Phone", "Healthcare", "Transportation"]
    
    if category in high_priority:
        return 0.7
    elif category in medium_priority:
        return 0.5
    else:
        return 0.3

def _calculate_history_risk(payment_history: List[dict]) -> float:
    if not payment_history:
        return 0.5
    
    late_count = sum(1 for p in payment_history if p.get('was_late', False) or p.get('daysPastDue', 0) > 0)
    total = len(payment_history)
    
    if total == 0:
        return 0.5
    
    late_rate = late_count / total
    return min(late_rate + 0.2, 1.0)

def _generate_risk_reasoning(name: str, amount: float, days_until_due: int, 
                             risk_level: str, probability: float) -> str:
    """Generate reasoning for payment risk prediction"""
    
    reasons = []
    
    if amount > 500:
        reasons.append(f"high amount (${amount:.2f})")
    elif amount < 50:
        reasons.append(f"low amount (${amount:.2f})")
    
    if days_until_due < 0:
        reasons.append("bill is overdue")
    elif days_until_due < 3:
        reasons.append("due very soon")
    elif days_until_due > 14:
        reasons.append("due date is far out")
    
    if risk_level == "LOW":
        base = f"Low risk ({probability:.0%}) of late payment"
    elif risk_level == "MEDIUM":
        base = f"Moderate risk ({probability:.0%}) of late payment"
    else:
        base = f"High risk ({probability:.0%}) of late payment"
    
    if reasons:
        return f"{base} due to {', '.join(reasons)}"
    else:
        return f"{base} based on analysis"

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*50)
    print("🤖 Payment Reminder AI Service Starting...")
    print("="*50)
    print(f"Model loaded: {categorization_model is not None}")
    print(f"Agent loaded: {agent is not None}")
    print(f"Categories: {len(categories_list)}")
    print(f"Available endpoints: /health, /categorize-bill, /predict-payment")
    print(f"                     /test-categorization, /categories, /chat, /tools")
    print("="*50 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)