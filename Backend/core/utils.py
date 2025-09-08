from datetime import UTC, datetime

from passlib.context import CryptContext


def naive_utc_now():
    return datetime.now(UTC).replace(tzinfo=None)


class CryptographyHelper:
    """Collection of helper methods for handling password cryptography."""

    _hash_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    @classmethod
    def hash_password(cls, password: str):
        return cls._hash_context.hash(password)

    @classmethod
    def verify_password(cls, plain_password, hashed_password):
        return cls._hash_context.verify(plain_password, hashed_password)
