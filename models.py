from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from decimal import Decimal
import datetime

# Initialize the database
db = SQLAlchemy()

class User(db.Model, UserMixin):
    """
    User model for the database, compatible with Auth0.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # This is the new, unique ID from Auth0 (e.g., 'sub')
    auth0_id = db.Column(db.String(200), unique=True, nullable=False)
    
    email = db.Column(db.String(150), unique=True, nullable=False)
    
    # --- THIS IS THE CHANGE ---
    # We now give 5.0 free credits by default
    credits = db.Column(db.Numeric(10, 4), default=Decimal('5.0'), nullable=False)
    
    def get_credit_balance(self):
        """Returns the credit balance as a float for calculations."""
        # --- FIX: Removed extra ')' from this line ---
        return float(self.credits)