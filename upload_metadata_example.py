from sre_constants import BRANCH
from dagshub.upload import Repo 
from dagshub.streaming import install_hooks, DagsHubFilesystem
#, repo_url="https://github.com/keitazoumana/DataStreamingClient_Model_Training.git"
from yaml import safe_load 
import os
import logging

logging.basicConfig(level=logging.DEBUG)

# Install hooks
install_hooks()

# Get The Credentials 
credentials = safe_load(open("config.yaml"))["credentials"]

# DVC Credentials
dvc_credentials = credentials["dvc_config"]
DVC_REMOTE_URL = dvc_credentials["DAGSHUB_REMOTE_URL"]
USERNAME = dvc_credentials["USERNAME"]
PASSWORD = dvc_credentials["PASSWORD"]

# Repo information 
repo_info = credentials["repo_info"]
OWNER_NAME = repo_info["OWNER_NAME"]
REPO_NAME = repo_info["REPO_NAME"] 

# My project repo
my_repo = Repo(OWNER_NAME, REPO_NAME, username=USERNAME, password=PASSWORD)
fs = DagsHubFilesystem()

# Helper function to upload all the files from a folder
def upload_files(folder_name, commit_message):

    dvc_folder = my_repo.directory(folder_name)

    for file in os.listdir(folder_name):
        dvc_folder.add(folder_name+'/'+file)
    
    # Run the final commit
    dvc_folder.commit(commit_message, versioning="dvc")

"""
Define folders to upload
"""
meta_data = credentials["metadata_path"]

# Model folder
model_folder = meta_data["MODEL_PATH"]

# Data folders
data = meta_data["DATA_PATH"]

"""
Upload the files to the Repository
"""
upload_files(model_folder, commit_message="Upload of the model to DVC")
upload_files(data, commit_message="Upload of the data to DVC")