import os

def createNoExistsFolders(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)