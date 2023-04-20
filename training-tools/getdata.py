import requests
import threading
import os
import select
import sys
from tqdm import tqdm

FILE_NAME = 'links.txt'
BASE_URL = 'https://storage.lczero.org/files/training_data/test80/'
DOWNLOAD_FOLDER = '~/Downloads/'
NUM_DOWNLOADS = 10


def download_file(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)

        with open(os.path.join(DOWNLOAD_FOLDER, filename+'.tmp'), 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress_bar.update(len(chunk))

        progress_bar.close()
        print(f'Downloaded: {filename}')
        os.system(f"mv {str(os.path.join(DOWNLOAD_FOLDER, filename+'.tmp'))} {str(os.path.join(DOWNLOAD_FOLDER+'lctd/', filename))}")

    except requests.exceptions.RequestException as e:
        print(f'Error downloading {filename}: {e}')


def download_batch(links):
    threads = []

    for link in links:
        url = BASE_URL + link
        t = threading.Thread(target=download_file, args=(url, link))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()


def remove_links_from_file(file, links):
    with open(file, 'r') as f:
        lines = f.readlines()

    with open(file, 'w') as f:
        for line in lines:
            if line.strip() not in links:
                f.write(line)


def get_user_input(timeout):
    print(f'Press ENTER to continue downloading or type "stop" to stop (waiting for {timeout} seconds): ')
    i, _, _ = select.select([sys.stdin], [], [], timeout)
    if i:
        user_input = sys.stdin.readline().strip()
    else:
        user_input = ""
    return user_input

def main():
    if not os.path.exists(DOWNLOAD_FOLDER):
        os.makedirs(DOWNLOAD_FOLDER)

    while True:
        with open(FILE_NAME, 'r') as f:
            links = [line.strip() for line in f.readlines() if line.strip()][:NUM_DOWNLOADS]

        if not links:
            print('No more links to download.')
            break

        download_batch(links)

        with open(FILE_NAME, 'r') as f:
            lines = f.readlines()

        with open(FILE_NAME, 'w') as f:
            for line in lines:
                if line.strip() not in links:
                    f.write(line)

        user_input = get_user_input(timeout=60)
        if user_input.lower() == 'stop':
            break

if __name__ == '__main__':
    main()
