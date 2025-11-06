from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.agent import PaymentReminderAgent

app = FastAPI(
    title="Payment Reminder AI Service",
    description="AI-powered assistant for payment reminders",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PaymentReminderAgent()

class ChatRequest(BaseModel):
    message: str
    use_memory: bool = True

class ChatResponse(BaseModel):
    response: str

@app.get("/")
async def root():
    return {
        "message": "Payment Reminder AI Service",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "tools": "/tools",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "UP",
        "service": "Payment Reminder AI",
        "version": "1.0.0"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - send messages to AI assistant
    """
    try:
        response = agent.chat(request.message, request.use_memory)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def get_tools():
    """
    Get list of available tools
    """
    return {
        "tools": agent.get_available_tools()
    }

@app.post("/clear-history")
async def clear_history():
    """
    Clear conversation history
    """
    agent.clear_history()
    return {"message": "Conversation history cleared"}

@app.get("/agent-info")
async def agent_info():
    """
    Get information about the AI agent
    """
    return {
        "name": "Payment Reminder AI Assistant",
        "description": "AI assistant for managing bills and payment reminders",
        "capabilities": [
            "View all bills",
            "Create new bills",
            "Get upcoming bills",
            "Calculate totals",
            "Natural language queries"
        ],
        "tools_count": len(agent.tools)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)