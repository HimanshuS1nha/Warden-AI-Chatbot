# Cookie names and a path, not secrets.
ACCESS_TOKEN_COOKIE = "access_token"  # noqa: S105
REFRESH_TOKEN_COOKIE = "refresh_token"  # noqa: S105
# The refresh token is only sent to auth routes (refresh, logout), not to
# every API call.
REFRESH_TOKEN_COOKIE_PATH = "/api/auth"  # noqa: S105
