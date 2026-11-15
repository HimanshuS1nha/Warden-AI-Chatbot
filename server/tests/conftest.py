"""Test environment setup.

This runs before any `warden` module reads its settings, so the app (and
Alembic) point at the test database and sign tokens with a known secret.

The test database is TEST_DATABASE_URL, or DATABASE_URL with "_test" added
to the database name. It is never the development database.
"""

import os

from sqlalchemy.engine import make_url

os.environ["JWT_SECRET_KEY"] = "test-secret-key-that-is-at-least-32-characters"

if "TEST_DATABASE_URL" not in os.environ:
    from warden.core.config import Settings

    dev_url = make_url(Settings().database_url)  # pyright: ignore[reportCallIssue]
    os.environ["TEST_DATABASE_URL"] = dev_url.set(
        database=f"{dev_url.database}_test"
    ).render_as_string(hide_password=False)

os.environ["DATABASE_URL"] = os.environ["TEST_DATABASE_URL"]
