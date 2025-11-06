import requests
import os
from dotenv import load_dotenv
from app.tools.bill_tools import GetBillsTool, CreateBillTool, GetUpcomingBillsTool, CalculatorTool

load_dotenv()

NUSCALE_API = os.getenv('NUSCALE_API')

class PaymentReminderAgent:
    def __init__(self):
        self.tools = {
            "get_bills": GetBillsTool(),
            "create_bill": CreateBillTool(),
            "get_upcoming_bills": GetUpcomingBillsTool(),
            "calculator": CalculatorTool()
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
        
        
        tool_response = self._check_and_use_tools(user_message)
        
        if tool_response:
            context = f"Tool Response: {tool_response}\n\nUser Question: {user_message}"
        else:
            context = user_message
        
        
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
        
        
        if any(word in message_lower for word in ['bills', 'payment', 'due', 'owe', 'upcoming']):
            if 'upcoming' in message_lower or 'due' in message_lower:
                tool = self.tools['get_upcoming_bills']
                return tool._run()
            else:
                tool = self.tools['get_bills']
                return tool._run()
        
        
        if any(word in message_lower for word in ['create', 'add', 'new bill', 'remind me']):
            
            return "To create a bill, please provide: Bill Name, Amount, Due Date, and Category"
        
        
        if any(word in message_lower for word in ['calculate', 'total', 'sum', '+', '-', '*', '/']):
            
            import re
            numbers = re.findall(r'\d+\.?\d*', message)
            if len(numbers) >= 2:
                expression = ' + '.join(numbers)
                tool = self.tools['calculator']
                return tool._run(expression)
        
        return None
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the Nuscale LLM API
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant for a payment reminder system. Help users manage their bills, set reminders, and answer questions about their payments. Be friendly and concise."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = requests.post(
                NUSCALE_API,
                json={
                    "model": "llama3.1:8b",
                    "messages": messages
                },
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                return "I received your message but couldn't generate a response."
            else:
                return f"Error from AI service: {response.status_code}"
        
        except requests.exceptions.Timeout:
            return "The AI service is taking too long to respond. Please try again."
        except Exception as e:
            return f"Error calling AI service: {str(e)}"
    
    def get_available_tools(self) -> list:
        """
        Return list of available tools
        """
        return [
            {
                "name": tool.name,
                "description": tool.description
            }
            for tool in self.tools.values()
        ]
    
    def clear_history(self):
        """
        Clear conversation history
        """
        self.conversation_history = []