from datetime import datetime
from uuid import UUID

from pydantic import EmailStr, BaseModel

from model.dao.enums import InviteStatus, Role


class UserDTO(BaseModel):
    id: UUID
    first_name: str
    middle_name: str | None = None
    last_name: str
    email: EmailStr
    business_id: UUID | None = None
    role: Role | None = None


class ResetPasswordDTO(BaseModel):
    password: str


class BusinessDTO(BaseModel):
    id: UUID
    name: str
    industry: str | None = None


class UserInviteDTO(BaseModel):
    id: UUID
    invitee_email: str
    inviter_id: UUID
    business_id: UUID
    status: InviteStatus
    expires_at: datetime
