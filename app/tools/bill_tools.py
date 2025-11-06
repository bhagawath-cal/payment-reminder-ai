from langchain.tools import BaseTool
import requests
import os
from dotenv import load_dotenv

load_dotenv()

SPRING_BOOT_API = os.getenv('SPRING_BOOT_API', 'http://localhost:8080/api')

class GetBillsTool(BaseTool):
    name: str = "get_bills"
    description: str = "Get all bills from the payment reminder system. Use this when user asks about their bills, upcoming payments, or what they owe."
    
    def _run(self, query: str = "") -> str:
        try:
            response = requests.get(f"{SPRING_BOOT_API}/bills")
            if response.status_code == 200:
                bills = response.json()
                if not bills:
                    return "No bills found."
                
                result = "Here are your bills:\n"
                for bill in bills:
                    result += f"- {bill['billName']}: ${bill['amount']} due on {bill['dueDate']} (Status: {bill['status']})\n"
                return result
            else:
                return f"Error fetching bills: {response.status_code}"
        except Exception as e:
            return f"Error connecting to Spring Boot API: Make sure it's running on port 8080. Error: {str(e)}"

class CreateBillTool(BaseTool):
    name: str = "create_bill"
    description: str = """Create a new bill in the system. 
    Input should be in format: 'billName|amount|dueDate|category'
    Example: 'Electricity Bill|120.50|2025-11-15|Utilities'
    """
    
    def _run(self, bill_info: str) -> str:
        try:
            parts = bill_info.split('|')
            if len(parts) < 4:
                return "Error: Please provide bill information in format: 'billName|amount|dueDate|category'"
            
            bill_data = {
                "billName": parts[0].strip(),
                "amount": float(parts[1].strip()),
                "dueDate": parts[2].strip(),
                "category": parts[3].strip(),
                "status": "PENDING"
            }
            
            response = requests.post(f"{SPRING_BOOT_API}/bills", json=bill_data)
            if response.status_code == 200 or response.status_code == 201:
                return f"Successfully created bill: {bill_data['billName']} for ${bill_data['amount']}"
            else:
                return f"Error creating bill: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

class GetUpcomingBillsTool(BaseTool):
    name: str = "get_upcoming_bills"
    description: str = "Get bills that are due soon. Use this when user asks about upcoming payments or bills due this week/month."
    
    def _run(self, query: str = "") -> str:
        try:
            # For now, get all bills and filter (you can add specific endpoint later)
            response = requests.get(f"{SPRING_BOOT_API}/bills")
            if response.status_code == 200:
                bills = response.json()
                upcoming = [b for b in bills if b.get('status') == 'PENDING']
                
                if not upcoming:
                    return "No upcoming bills found."
                
                result = "Upcoming bills:\n"
                for bill in upcoming:
                    result += f"- {bill['billName']}: ${bill['amount']} due on {bill['dueDate']}\n"
                return result
            else:
                return f"Error fetching bills: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Perform mathematical calculations. Useful for calculating totals, differences, or any math operations."
    
    def _run(self, expression: str) -> str:
        try:
            # Safe eval for basic math
            result = eval(expression, {"__builtins__": {}}, {})
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating: {str(e)}"   