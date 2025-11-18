import requests
from typing import List, Dict
from datetime import datetime, timedelta

class BackendClient:
    def __init__(self, backend_url: str = "http://backend:8080"):
        self.backend_url = backend_url
        # Use the security credentials from backend
        self.auth = ("user", "966a88b3-904d-4635-a3e3-f4b4ac50b35a")
    
    def get_all_bills(self) -> List[Dict]:
        """Fetch all bills from backend"""
        try:
            response = requests.get(
                f"{self.backend_url}/api/bills",
                auth=self.auth,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching bills: {e}")
            return []
    
    def get_bills_due_soon(self, days: int = 7) -> List[Dict]:
        """Get bills due within the next N days"""
        try:
            bills = self.get_all_bills()
            today = datetime.now().date()
            future_date = today + timedelta(days=days)
            
            due_soon = []
            for bill in bills:
                # Parse due_date from bill
                due_date_str = bill.get('dueDate') or bill.get('due_date')
                status = bill.get('status', '').upper()
                
                if due_date_str and status == 'PENDING':
                    try:
                        # Handle different date formats
                        if 'T' in due_date_str:
                            due_date = datetime.fromisoformat(due_date_str.replace('Z', '+00:00')).date()
                        else:
                            due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
                        
                        if today <= due_date <= future_date:
                            due_soon.append(bill)
                    except Exception as e:
                        print(f"Error parsing date {due_date_str}: {e}")
            
            return due_soon
        except Exception as e:
            print(f"Error getting bills due soon: {e}")
            return []
    
    def get_overdue_bills(self) -> List[Dict]:
        """Get bills that are overdue"""
        try:
            bills = self.get_all_bills()
            today = datetime.now().date()
            
            overdue = []
            for bill in bills:
                due_date_str = bill.get('dueDate') or bill.get('due_date')
                status = bill.get('status', '').upper()
                
                if due_date_str and status == 'PENDING':
                    try:
                        if 'T' in due_date_str:
                            due_date = datetime.fromisoformat(due_date_str.replace('Z', '+00:00')).date()
                        else:
                            due_date = datetime.strptime(due_date_str, "%Y-%m-%d").date()
                        
                        if due_date < today:
                            overdue.append(bill)
                    except Exception as e:
                        print(f"Error parsing date {due_date_str}: {e}")
            
            return overdue
        except Exception as e:
            print(f"Error getting overdue bills: {e}")
            return []