import requests
import os

def download_file(url,local_filename):
    if not os.path.exists(os.path.dirname(local_filename)):
        os.makedirs(os.path.dirname(local_filename))
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian

    return local_filename