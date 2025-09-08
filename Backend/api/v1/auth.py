from uuid import UUID
from fastapi import APIRouter, Depends, status

from core.di_container import DependencyContainer
from dependency_injector.wiring import Provide, inject

from model.dto.auth import (
    ForgotPasswordDTO,
    InviteRegisterDTO,
    LoginDTO,
    OwnerRegisterDTO,
    TokenDataDTO,
)
from model.dto.base import BaseResponseDTO
from model.dto.users import ResetPasswordDTO
from service.auth import AuthService
from service.users import UserService


auth_router = APIRouter(prefix="/auth", tags=["Auth"])

UserServiceDependency = Depends(Provide[DependencyContainer.user_service_factory])

AuthServiceDependency = Depends(Provide[DependencyContainer.auth_service_factory])


@auth_router.post("/login", status_code=status.HTTP_200_OK)
@inject
async def login_user(dto: LoginDTO, service: AuthService = AuthServiceDependency) -> BaseResponseDTO:
    user = await service.authenticate_user(dto)

    access_token = service.generate_access_token(user)
    refresh_token = service.generate_refresh_token(user.email)

    return BaseResponseDTO(
        data=TokenDataDTO(access_token=access_token, refresh_token=refresh_token),
        message="Login successful.",
    )


@auth_router.post("/register", status_code=status.HTTP_201_CREATED)
@inject
async def register_user(
    dto: OwnerRegisterDTO, service: UserService = UserServiceDependency
) -> BaseResponseDTO:
    await service.create_user(dto)

    return BaseResponseDTO(
        message="Registration successful. Check your email to verify your account."
    )


@auth_router.post("/verify-account", status_code=status.HTTP_200_OK)
@inject
async def verify_user(code: str, service: UserService = UserServiceDependency) -> BaseResponseDTO:
    await service.verify_user(code)

    return BaseResponseDTO(message="Verification successful. You can proceed to login.")


@auth_router.get("/invite/{invite_id}", status_code=status.HTTP_200_OK)
@inject
async def fetch_invite(invite_id: UUID, service: UserService = UserServiceDependency) -> BaseResponseDTO:
    invite_dto = await service.retrieve_invite(invite_id)

    return BaseResponseDTO(
        data=invite_dto,
        message="Invite retrieved successfully.",
    )


@auth_router.post("/invite/{invite_id}/accept", status_code=status.HTTP_201_CREATED)
@inject
async def accept_invite(
    invite_id: UUID,
    dto: InviteRegisterDTO,
    service: UserService = UserServiceDependency,
) -> BaseResponseDTO:
    await service.create_user_from_invite(invite_id, dto)

    return BaseResponseDTO(
        message="Registration successful. Check your email to verify your account."
    )


@auth_router.post("/invite/{invite_id}/reject", status_code=status.HTTP_202_ACCEPTED)
@inject
async def reject_invite(invite_id: UUID, service: UserService = UserServiceDependency) -> BaseResponseDTO:
    await service.reject_invite(invite_id)

    return BaseResponseDTO(message="Invite rejected.")


@auth_router.post("/forgot-password/", status_code=status.HTTP_200_OK)
@inject
async def initiate_password_reset(
    dto: ForgotPasswordDTO, service: UserService = UserServiceDependency
) -> BaseResponseDTO:
    await service.initiate_password_reset(dto)

    return BaseResponseDTO(
        message="Password reset triggered successfully. Check your email to proceed.",
    )


@auth_router.post(
    "/reset-password/{password_reset_id}", status_code=status.HTTP_202_ACCEPTED
)
@inject
async def reset_password(
    password_reset_id: UUID,
    dto: ResetPasswordDTO,
    service: UserService = UserServiceDependency,
) -> BaseResponseDTO:
    await service.reset_user_password(password_reset_id, dto)

    return BaseResponseDTO(
        message="Password reset successful. You can proceed to login.",
    )
