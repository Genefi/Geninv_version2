from fastapi import APIRouter, Depends, Request, status

from core.di_container import DependencyContainer
from model.dto.auth import InviteUserDTO
from dependency_injector.wiring import Provide, inject

from model.dto.base import BaseResponseDTO
from model.dto.users import UserDTO
from service.users import UserService

users_router = APIRouter(prefix="/users", tags=["Users"])

UserServiceDependency = Depends(Provide[DependencyContainer.user_service_factory])


@users_router.post("/invite", status_code=status.HTTP_202_ACCEPTED)
@inject
async def invite_user(
    request: Request,
    dto: InviteUserDTO,
    service: UserService = UserServiceDependency,
) -> BaseResponseDTO:
    user: UserDTO = request.state.user
    await service.invite_user(dto, user.id)

    return BaseResponseDTO(message="Invite sent successfully.")
