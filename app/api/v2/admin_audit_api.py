from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session_v2 import get_db
from app.db.models.admin_audit_log import AdminAuditLog
from app.auth.guards import require_role

router = APIRouter(
    prefix="/api/v2/admin-audit",
    tags=["Admin Audit"],
)


@router.get("/")
async def list_audit_logs(db: AsyncSession = Depends(get_db),user=Depends(require_role("admin", "viewer")),
):
    result = await db.execute(
        select(AdminAuditLog)
        .order_by(AdminAuditLog.created_at.desc())
        .limit(200)
    )

    logs = result.scalars().all()

    return [
        {
            "id": str(l.id),
            "action": l.action,
            "entity_type": l.entity_type,
            "entity_id": str(l.entity_id),
            "before": l.before_value,
            "after": l.after_value,
            "meta_data": l.meta_data,
            "created_at": l.created_at,
        }
        for l in logs
    ]
