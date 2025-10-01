"""
Authentication configuration for LangGraph Agent Server.

This module provides environment-based authentication switching between:
- noop: No authentication (allow all requests)  
- custom: Custom authentication integration

Set AUTH_TYPE environment variable to choose authentication mode.
"""

import os
import logging
from typing import Dict, Any
from langgraph_sdk import Auth

logger = logging.getLogger(__name__)

# Initialize LangGraph Auth instance
auth = Auth()

# Get authentication type from environment
AUTH_TYPE = os.getenv("AUTH_TYPE", "noop").lower()

if AUTH_TYPE == "noop":
    logger.info("Using noop authentication (no auth required)")
    
    @auth.authenticate
    async def authenticate(headers: Dict[str, str]) -> Auth.types.MinimalUserDict:
        """No-op authentication that allows all requests."""
        _ = headers  # Suppress unused warning
        return {
            "identity": "anonymous",
            "display_name": "Anonymous User", 
            "is_authenticated": True
        }

    @auth.on
    async def authorize(ctx: Auth.types.AuthContext, value: Dict[str, Any]) -> Dict[str, Any]:
        """No-op authorization that allows access to all resources."""
        _ = ctx, value  # Suppress unused warnings
        return {}  # Empty filter = no access restrictions
    
elif AUTH_TYPE == "custom":  
    logger.info("Using custom authentication with NextAuth JWT validation")
    
    # Import jwt here to avoid dependency when using noop mode
    try:
        import jwt
    except ImportError:
        logger.error("PyJWT not installed. Run: pip install pyjwt")
        raise
    
    # Get JWT secret from environment (use BrainCore backend secret)
    JWT_SECRET = os.getenv("SECRET_KEY", "mock-secret-key-for-migrations")
    
    @auth.authenticate
    async def authenticate(headers: Dict[str, str]) -> Auth.types.MinimalUserDict:
        """
        Validate NextAuth JWT tokens.

        This validates tokens from NextAuth.js used in the BrainCore frontend.
        """
        # Extract authorization header
        authorization = (
            headers.get("authorization") or
            headers.get("Authorization") or
            headers.get(b"authorization") or
            headers.get(b"Authorization")
        )

        # Handle bytes headers
        if isinstance(authorization, bytes):
            authorization = authorization.decode('utf-8')

        if not authorization:
            logger.warning("Missing Authorization header")
            # For development, allow anonymous access even in custom mode
            if os.getenv("ALLOW_ANONYMOUS", "false").lower() == "true":
                return {
                    "identity": "anonymous",
                    "display_name": "Anonymous User",
                    "is_authenticated": False
                }
            raise Auth.exceptions.HTTPException(
                status_code=401,
                detail="Authorization header required"
            )
        
        # Extract token from "Bearer <token>" format
        if not authorization.startswith("Bearer "):
            raise Auth.exceptions.HTTPException(
                status_code=401,
                detail="Invalid authorization format. Expected 'Bearer <token>'"
            )
        
        token = authorization.replace("Bearer ", "")
        
        # For development/testing with dummy tokens
        if token.startswith("anonymous") or token.startswith("noop-"):
            user_id = token.split("-", 1)[-1] if "-" in token else "anonymous"
            return {
                "identity": user_id,
                "display_name": f"Dev User ({user_id})",
                "is_authenticated": True
            }
        
        # Validate real JWT token from BrainCore backend
        if not JWT_SECRET:
            logger.error("JWT_SECRET not configured")
            # Fallback for development
            if os.getenv("NODE_ENV") == "development":
                return {
                    "identity": "dev-user",
                    "display_name": "Development User",
                    "is_authenticated": True
                }
            raise Auth.exceptions.HTTPException(
                status_code=500,
                detail="Authentication not properly configured"
            )
        
        try:
            # Debug: Log the token and secret being used
            logger.info(f"Token length: {len(token)}")
            logger.info(f"JWT_SECRET being used: {JWT_SECRET[:10]}..." if JWT_SECRET else "No JWT_SECRET")

            # Debug: Try to decode without verification to see payload
            try:
                unverified_payload = jwt.decode(token, options={"verify_signature": False})
                logger.info(f"Token payload (unverified): {unverified_payload}")
            except Exception as e:
                logger.warning(f"Could not decode token without verification: {e}")

            # Decode BrainCore backend JWT
            # BrainCore backend uses HS256
            payload = jwt.decode(
                token,
                JWT_SECRET,
                algorithms=["HS256"]
            )
            
            # Extract user information from BrainCore backend token
            # BrainCore uses different field names
            user_id = payload.get("user_id") or payload.get("sub") or payload.get("id")
            email = payload.get("email")
            name = payload.get("name") or payload.get("username")
            
            if not user_id:
                logger.error(f"No user ID in token payload: {payload}")
                raise Auth.exceptions.HTTPException(
                    status_code=401,
                    detail="Invalid token: missing user identity"
                )
            
            # Return user information for LangGraph
            return {
                "identity": str(user_id),
                "display_name": name or email or str(user_id),
                "email": email,
                "is_authenticated": True,
                "permissions": ["read", "write"],
                "metadata": {
                    # Add any additional metadata from token
                    "provider": payload.get("provider"),
                    "org_id": payload.get("orgId"),
                    "company_id": payload.get("companyId"),
                    "family_id": payload.get("familyId"),
                }
            }
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            raise Auth.exceptions.HTTPException(
                status_code=401,
                detail="Token expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            # In development, be more lenient
            if os.getenv("NODE_ENV") == "development":
                return {
                    "identity": "dev-user-invalid-token",
                    "display_name": "Dev User (Invalid Token)",
                    "is_authenticated": True
                }
            raise Auth.exceptions.HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}", exc_info=True)
            raise Auth.exceptions.HTTPException(
                status_code=500,
                detail="Authentication system error"
            )

    @auth.on
    async def authorize(ctx: Auth.types.AuthContext, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-tenant authorization with user-scoped access control.
        """
        try:
            # Get user identity from authentication context
            user_id = ctx.user.identity
            
            if not user_id:
                logger.error("Missing user identity in auth context")
                raise Auth.exceptions.HTTPException(
                    status_code=401,
                    detail="Invalid user identity"
                )
            
            # Create owner filter for resource access control
            owner_filter = {"owner": user_id}
            
            # Add owner information to metadata for create/update operations
            metadata = value.setdefault("metadata", {})
            metadata.update(owner_filter)
            
            # Return filter for database operations
            return owner_filter
            
        except Auth.exceptions.HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authorization error: {e}", exc_info=True)
            raise Auth.exceptions.HTTPException(
                status_code=500,
                detail="Authorization system error"
            )
            
else:
    raise ValueError(
        f"Unknown AUTH_TYPE: {AUTH_TYPE}. "
        f"Supported values: 'noop', 'custom'"
    )