ACCESS_TOKEN_COOKIE = "access_token"
REFRESH_TOKEN_COOKIE = "refresh_token"
# The refresh token is only sent to auth routes (refresh, logout), not to
# every API call.
REFRESH_TOKEN_COOKIE_PATH = "/api/auth"
