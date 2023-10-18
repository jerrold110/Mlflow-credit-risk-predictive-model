import os

x = os.getcwd()
print(x)
print(os.path.normpath(x + os.sep + os.pardir))