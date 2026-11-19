"""Tools the agent can call, built per request for the signed-in user.

Every factory takes the CurrentUser from the session and returns tools with
no identity arguments: the model can choose which tool to call, never whose
data it reads.
"""
