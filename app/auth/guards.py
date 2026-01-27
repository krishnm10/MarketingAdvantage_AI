from fastapi import Depends, HTTPException, status
from app.auth.deps import get_current_user

def require_role(*allowed_roles):
    """
    Require a specific role or roles.
    Example: user=Depends(require_role("admin", "editor"))
    """
    async def role_checker(current_user=Depends(get_current_user)):
        role = current_user.get("role")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access forbidden: insufficient role privileges",
            )
        return current_user

    return role_checker
