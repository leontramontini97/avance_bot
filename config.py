import os

class Config:
    # Database configuration
    
    
    # Session and security configuration
    SESSION_TYPE = 'filesystem'
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')

    # API Keys and other configurations with defaults
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    

 