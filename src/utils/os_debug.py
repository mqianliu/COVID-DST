
# Python program to explain os.environ object

# importing os module
import os
import pprint

# Get the list of user's
# environment variables
env_var = os.environ

# Print the list of user's
# environment variables
# print("User's Environment variable:")
pprint.pprint(dict(env_var), width=1)

# Get the value of
# 'HOME' environment variable
# home = os.environ['HOME']

# Print the value of
# 'HOME' environment variable
# print("HOME:", home)

# Set "HOME" environment variable
# os.environ['HOME'] = "D:/ANAHOME"
# pprint.pprint(dict(env_var), width=1)
# home = os.environ['HOME']
# print("HOME:", home)

