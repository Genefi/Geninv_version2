from enum import StrEnum


class InviteStatus(StrEnum):
    CREATED = "created"
    SENT = "sent"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Role(StrEnum):
    GENINV_ADMIN = "geninv_admin"
    BUSINESS_ADMIN = "business_admin"
    BUSINESS_USER = "business_user"