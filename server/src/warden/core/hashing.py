import hashlib

from fastapi.concurrency import run_in_threadpool
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

password_hash = PasswordHash((Argon2Hasher(),))


async def hash_password(password: str) -> str:
    return await run_in_threadpool(password_hash.hash, password)


async def verify_password(password: str, hashed: str) -> bool:
    return await run_in_threadpool(password_hash.verify, password, hashed)


async def verify_and_update_password(password: str, hashed: str) -> tuple[bool, str | None]:
    """Verify a password and return a new hash if the stored one uses outdated
    parameters. The caller should save the new hash when it isn't None."""
    return await run_in_threadpool(password_hash.verify_and_update, password, hashed)


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a random token (64 chars)."""
    return hashlib.sha256(token.encode()).hexdigest()
