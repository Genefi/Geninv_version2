from base64 import b64decode, b64encode
import binascii
from datetime import timedelta
from logging import Logger
from uuid import UUID
from core.database import SQLDatabase
from core.environment import settings
from core.exceptions import (
    BadRequestException,
    ConflictException,
    ForbiddenException,
    NotFoundException,
)
from core.logger import app_logger
from model.dao.enums import InviteStatus, Role
from model.dao.users import (
    BusinessDAO,
    UserDAO,
    UserInviteDAO,
    UserPasswordResetDAO,
)
from model.dto.auth import (
    ForgotPasswordDTO,
    InviteUserDTO,
    OwnerRegisterDTO,
    InviteRegisterDTO,
)
from model.dto.users import (
    BusinessDTO,
    ResetPasswordDTO,
    UserDTO,
)
from core.utils import CryptographyHelper, naive_utc_now


class UserService:
    def __init__(self, pg_database: SQLDatabase, logger: Logger = app_logger):
        self._db = pg_database
        self._logger = logger

    async def create_user(
        self,
        dto: OwnerRegisterDTO | InviteRegisterDTO,
        business_id: UUID = None,
    ):
        password = CryptographyHelper.hash_password(dto.password)

        user = UserDAO(
            first_name=dto.first_name,
            middle_name=dto.middle_name,
            last_name=dto.last_name,
            email=dto.email,
            password=password,
        )

        await user.save(self._db)

        if business_id is None:
            business = await self.create_business(dto, user.id)
            business_id = business.id

            user.role = Role.BUSINESS_ADMIN

        user.business_id = business_id
        await user.save(self._db)
        
        await self.send_verification_email(user.to_dto())

    async def create_business(
        self, dto: OwnerRegisterDTO, owner_id: UUID
    ) -> BusinessDTO:
        business = BusinessDAO(
            name=dto.business_name,
            owner_id=owner_id,
            industry=dto.industry,
        )

        await business.save(self._db)

        return business.to_dto()

    async def send_verification_email(self, user: UserDTO):
        # TODO (@therealosy): Add logic to send verification email on user registration. For now, log it

        verification_text = f"{user.id}::{user.business_id}"

        verification_code = b64encode(verification_text.encode()).decode()

        self._logger.debug(f"Sending verification email with code {verification_code}")

    async def verify_user(self, verification_code: str):
        _verification_code_exception = BadRequestException("Invalid verification code")

        try:
            verification_code = b64decode(verification_code).decode()
        except binascii.Error as exc:
            self._logger.error(f"Error decoding verification code: {exc}")
            raise _verification_code_exception

        user_id, _, business_id = verification_code.partition("::")

        try:
            user_id = UUID(user_id)
            business_id = UUID(business_id)
        except ValueError:
            raise _verification_code_exception

        user = (
            await UserDAO.filter(
                id=user_id,
                business_id=business_id,
                db_resource=self._db,
            )
        ).first()

        if user is None:
            raise _verification_code_exception

        user.email_verified = True
        await user.save(db_resource=self._db)

    async def invite_user(self, dto: InviteUserDTO, inviter_id: UUID):
        inviter = await UserDAO.get(inviter_id, db_resource=self._db)

        if not inviter:
            raise NotFoundException("Inviter not found")
        
        if inviter.role not in [Role.BUSINESS_ADMIN, Role.GENINV_ADMIN]:
            raise ForbiddenException("You are unable to invite a user.")

        now = naive_utc_now()
        invitee_email = dto.invitee_email

        # Check if user already exists
        invited_user = (
            await UserDAO.filter(email=invitee_email, db_resource=self._db)
        ).first()

        if invited_user is not None:
            raise ConflictException("User already signed up.")

        # Check if any pending invites already exist
        pending_invite = (
            await UserInviteDAO.filter(
                inviter_id=inviter_id,
                invitee_email=invitee_email,
                status_in=[InviteStatus.SENT],
                expires_after=now,
                db_resource=self._db,
            )
        ).first()

        if pending_invite:
            raise ConflictException(
                f"An unaccepted invite already exists for `{invitee_email}`"
            )

        # Check if any accepted invites already exist
        accepted_invite = (
            await UserInviteDAO.filter(
                inviter_id=inviter_id,
                invitee_email=invitee_email,
                status_in=[InviteStatus.ACCEPTED],
                expires_after=now,
                db_resource=self._db,
            )
        ).first()

        if accepted_invite:
            raise ConflictException(
                f"`{invitee_email}` has already accepted your invite."
            )

        expires_at = now + timedelta(seconds=settings.INVITE_EXPIRES_AFTER_SECS)

        invite = UserInviteDAO(
            invitee_email=invitee_email,
            inviter_id=inviter.id,
            business_id=inviter.business_id,
            expires_at=expires_at,
        )
        await invite.save(self._db)

        await self.send_invite(invite)

    async def send_invite(self, invite: UserInviteDAO):
        # TODO (@therealosy): Add logic to send invite email. For now just log it
        self._logger.debug(f"Sending invite `{invite.id}` to `{invite.invitee_email}`")

        invite.status = InviteStatus.SENT
        await invite.save(self._db)

    async def retrieve_invite(self, invite_id: UUID):
        invite = await UserInviteDAO.get(invite_id, db_resource=self._db)

        if invite is None:
            raise BadRequestException("Invalid invite.")
        elif invite.expires_at < naive_utc_now():
            raise BadRequestException("Invite has expired")
        elif invite.status in [InviteStatus.ACCEPTED, InviteStatus.REJECTED]:
            raise BadRequestException(f"Invite has already been {invite.status}")

        return invite

    async def create_user_from_invite(
        self, invite_id: UUID, registration: InviteRegisterDTO
    ):
        invite = await self.retrieve_invite(invite_id)

        if invite.invitee_email != registration.email:
            raise BadRequestException(f"Invalid invite for email {registration.email}")

        await self.create_user(registration, invite.business_id)

        invite.status = InviteStatus.ACCEPTED
        await invite.save(self._db)

    async def reject_invite(self, invite_id: UUID):
        invite = await self.retrieve_invite(invite_id)

        invite.status = InviteStatus.REJECTED
        await invite.save(self._db)

    async def initiate_password_reset(self, dto: ForgotPasswordDTO):
        email = dto.email
        user = (await UserDAO.filter(email=email, db_resource=self._db)).first()

        if not user:
            raise NotFoundException("User not found")

        now = naive_utc_now()

        password_reset = (
            await UserPasswordResetDAO.filter(
                user_id=user.id,
                is_used=False,
                expires_after=now,
                db_resource=self._db,
            )
        ).first()

        if password_reset is None:
            expires_at = now + timedelta(seconds=settings.INVITE_EXPIRES_AFTER_SECS)

            password_reset = UserPasswordResetDAO(
                user_id=user.id, expires_at=expires_at
            )
            await password_reset.save(self._db)

        await self.send_password_reset(password_reset, user.to_dto())

    async def send_password_reset(
        self, password_reset: UserPasswordResetDAO, user: UserDTO
    ):
        # TODO (@therealosy): Implement logic to send password reset email. For now, log it.
        self._logger.debug(f"Sending password reset email `{password_reset.id}` to `{user.email}`")

    async def retrieve_password_reset(self, password_reset_id: UUID):
        password_reset = await UserPasswordResetDAO.get(
            password_reset_id, db_resource=self._db
        )

        if password_reset is None:
            raise BadRequestException("Invalid password reset link.")
        elif password_reset.expires_at < naive_utc_now():
            raise BadRequestException("Password reset link has expired.")
        elif password_reset.is_used:
            raise BadRequestException("Password reset link has already been used.")

        return password_reset

    async def reset_user_password(self, password_reset_id: UUID, dto: ResetPasswordDTO):
        password_reset = await self.retrieve_password_reset(password_reset_id)

        user = await UserDAO.get(password_reset.user_id, db_resource=self._db)

        user.password = CryptographyHelper.hash_password(dto.password)
        password_reset.is_used = True

        await user.save(db_resource=self._db)
        await password_reset.save(db_resource=self._db)
