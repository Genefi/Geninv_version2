from pydantic import BaseModel, EmailStr


class LoginDTO(BaseModel):
    email: EmailStr
    password: str


class InviteRegisterDTO(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    middle_name: str
    last_name: str


class OwnerRegisterDTO(InviteRegisterDTO):
    business_name: str
    industry: str


class InviteUserDTO(BaseModel):
    invitee_email: EmailStr


class ForgotPasswordDTO(BaseModel):
    email: EmailStr


class TokenDataDTO(BaseModel):
    access_token: str
    refresh_token: str
