import requests
import os
from dotenv import load_dotenv
from app.tools.bill_tools import GetBillsTool, CreateBillTool, GetUpcomingBillsTool, CalculatorTool
from .database_client import BackendClient

load_dotenv()

NUSCALE_API = os.getenv('NUSCALE_API')

class PaymentReminderAgent:
    def __init__(self):
        # Fix: tools should be a dict, not a list with mixed types
        self.backend_client = BackendClient()
        
        self.tools = {
            "get_bills": GetBillsTool(),
            "create_bill": CreateBillTool(),
            "get_upcoming_bills": GetUpcomingBillsTool(),
            "calculator": CalculatorTool(),
        }
        
        # Additional tool definitions
        self.tool_functions = {
            "get_bills_due_soon": self.get_bills_due_soon,
            "get_overdue_bills": self.get_overdue_bills
        }
        
        self.conversation_history = []
    
    def chat(self, user_message: str, use_memory: bool = True) -> str:
        """
        Process user message and return AI response
        """
        if use_memory:
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
        
        # Check if we need to use tools
        tool_response = self._check_and_use_tools(user_message)
        
        if tool_response:
            context = f"Tool Response: {tool_response}\n\nUser Question: {user_message}"
        else:
            context = user_message
        
        # Call LLM
        try:
            ai_response = self._call_llm(context)
            
            if use_memory:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": ai_response
                })
            
            return ai_response
        except Exception as e:
            return f"Error: Unable to process request. {str(e)}"
    
    def _check_and_use_tools(self, message: str) -> str:
        """
        Determine if we need to use tools based on user message
        """
        message_lower = message.lower()
        
        # Check for bill queries
        if any(word in message_lower for word in ['bills due', 'upcoming bills', 'bills coming up', 'due soon']):
            return self.get_bills_due_soon()
        
        if any(word in message_lower for word in ['overdue', 'late bills', 'late payment', 'missed payment']):
            return self.get_overdue_bills()
        
        if any(word in message_lower for word in ['all bills', 'my bills', 'show bills', 'list bills']):
            try:
                tool = self.tools['get_bills']
                return tool._run()
            except Exception as e:
                return f"Error fetching bills: {str(e)}"
        
        # Check for bill creation
        if any(word in message_lower for word in ['create', 'add', 'new bill', 'remind me']):
            return "To create a bill, please provide: Bill Name, Amount, Due Date, and Category"
        
        # Check for calculations
        if any(word in message_lower for word in ['calculate', 'total', 'sum', '+', '-', '*', '/']):
            import re
            numbers = re.findall(r'\d+\.?\d*', message)
            if len(numbers) >= 2:
                expression = ' + '.join(numbers)
                try:
                    tool = self.tools['calculator']
                    return tool._run(expression)
                except Exception as e:
                    return f"Error calculating: {str(e)}"
        
        return None
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the Nuscale LLM API
        """
        if not NUSCALE_API:
            return "AI service is not configured. Please set NUSCALE_API environment variable."
        
        try:
            # Build messages including conversation history
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant for a payment reminder system. 
                    
Your capabilities:
- Help users manage their bills and payments
- Answer questions about payment schedules
- Provide information about upcoming and overdue bills
- Give financial advice about bill management
- Be friendly, concise, and helpful

When responding:
- Be conversational and natural
- Provide specific, actionable advice
- If tool data is provided, use it to give detailed answers
- Keep responses under 150 words unless more detail is needed"""
                }
            ]
            
            # Add conversation history if available (last 5 messages for context)
            if len(self.conversation_history) > 0:
                recent_history = self.conversation_history[-5:]  # Last 5 messages
                messages.extend(recent_history)
            
            # Add current prompt
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            response = requests.post(
                NUSCALE_API,
                json={
                    "model": "llama3.1:8b",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                elif 'message' in result:
                    return result['message']['content']
                elif 'response' in result:
                    return result['response']
                else:
                    return "I received your message but couldn't generate a response."
            else:
                return f"I'm having trouble connecting to my AI brain right now. (Error: {response.status_code})"
        
        except requests.exceptions.Timeout:
            return "I'm thinking really hard, but it's taking too long. Can you try asking again?"
        except requests.exceptions.ConnectionError:
            return "I can't connect to my AI service right now. Please check if the NUSCALE_API is accessible."
        except Exception as e:
            return f"Oops! Something went wrong: {str(e)}"
    
    def get_available_tools(self) -> list:
        """
        Return list of available tools
        """
        tools_list = []
        
        # Add tools from self.tools
        for name, tool in self.tools.items():
            tools_list.append({
                "name": name,
                "description": getattr(tool, 'description', f'Tool: {name}')
            })
        
        # Add custom tool functions
        tools_list.extend([
            {
                "name": "get_bills_due_soon",
                "description": "Get bills due in the next 7 days"
            },
            {
                "name": "get_overdue_bills",
                "description": "Get all overdue bills"
            }
        ])
        
        return tools_list
    
    def clear_history(self):
        """
        Clear conversation history
        """
        self.conversation_history = []
        
    def get_bills_due_soon(self, days: int = 7) -> str:
        """Tool to get bills due soon"""
        try:
            bills = self.backend_client.get_bills_due_soon(days)
            
            if not bills:
                return f"Good news! You have no bills due in the next {days} days."
            
            response = f"📅 You have {len(bills)} bill(s) due in the next {days} days:\n\n"
            for bill in bills:
                name = bill.get('billName') or bill.get('bill_name', 'Unknown')
                amount = bill.get('amount', 0)
                due_date = bill.get('dueDate') or bill.get('due_date', 'Unknown')
                category = bill.get('category', 'Unknown')
                
                response += f"• {name}: ${amount:.2f} due on {due_date} ({category})\n"
            
            return response
        except Exception as e:
            return f"I couldn't fetch your upcoming bills right now. Error: {str(e)}"
    
    def get_overdue_bills(self) -> str:
        """Tool to get overdue bills"""
        try:
            bills = self.backend_client.get_overdue_bills()
            
            if not bills:
                return "✅ Great job! You have no overdue bills."
            
            response = f"⚠️ You have {len(bills)} overdue bill(s):\n\n"
            for bill in bills:
                name = bill.get('billName') or bill.get('bill_name', 'Unknown')
                amount = bill.get('amount', 0)
                due_date = bill.get('dueDate') or bill.get('due_date', 'Unknown')
                
                response += f"• {name}: ${amount:.2f} was due on {due_date}\n"
            
            return response
        except Exception as e:
            return f"I couldn't fetch your overdue bills right now. Error: {str(e)}"