from slowapi import Limiter
from slowapi.util import get_remote_address

from dotenv import load_dotenv
from app.utils.utils import parse_list_from_env
load_dotenv()


rate_limits = dict(
    RATE_LIMIT_DEFAULT = parse_list_from_env("RATE_LIMIT_DEFAULT", ["1000 per day", "200 per hour"]),
    
    RATE_LIMIT_THOUGHTS_CREATE = parse_list_from_env("RATE_LIMIT_THOUGHTS_CREATE", [ "100 per minute"]),
    RATE_LIMIT_THOUGHTS_READ = parse_list_from_env("RATE_LIMIT_THOUGHTS_READ", [ "30 per minute"]),
    RATE_LIMIT_THOUGHTS_UPDATE = parse_list_from_env("RATE_LIMIT_THOUGHTS_UPDATE", [ "30 per minute"]),
    RATE_LIMIT_THOUGHTS_DELETE = parse_list_from_env("RATE_LIMIT_THOUGHTS_DELETE", [ "30 per minute"]),
    
    RATE_LIMIT_SESSIONS_READ = parse_list_from_env("RATE_LIMIT_SESSIONS_READ", [ "30 per minute"]),
    RATE_LIMIT_SESSION_DELETE = parse_list_from_env("RATE_LIMIT_SESSION_DELETE", [ "30 per minute"]),
    RATE_LIMIT_SESSION_HISTORY = parse_list_from_env("RATE_LIMIT_SESSION_HISTORY", [ "30 per minute"]),
    
    RATE_LIMIT_CHAT = parse_list_from_env("RATE_LIMIT_CHAT", [ "100 per minute"]),
    
    RATE_LIMIT_DISCOVER_ARTICLES = parse_list_from_env("RATE_LIMIT_DISCOVER_ARTICLES", [ "100 per minute"]),
    
    RATE_LIMIT_WEBHOOKS_CLERK_AUTH = parse_list_from_env("RATE_LIMIT_WEBHOOKS_CLERK_AUTH", ["60 per minute"]),
)

limiter = Limiter(key_func=get_remote_address, default_limits=rate_limits["RATE_LIMIT_DEFAULT"])
