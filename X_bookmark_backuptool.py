import subprocess
import sys
import os
import shutil
import hashlib
import time
import re
import requests
import msvcrt  # For detecting key press on Windows
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Automatically install required packages
print('X bookmark backuptool by qus20000\n\n')
print("message: preparing required packages...")
try:
    import requests
except ImportError:
    install('requests')
    import requests

try:
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    install('selenium')
    install('webdriver-manager')
    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager

try:
    from PIL import Image
except ImportError:
    install('Pillow')
    from PIL import Image

# Suppress TensorFlow Lite messages
import os
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Restore stderr
sys.stderr = stderr

# Create a unique folder for images
folder_base_name = "images"
folder_index = 0
new_folder_path = folder_base_name

while os.path.exists(new_folder_path):
    new_folder_path = f"{folder_base_name}{folder_index}"
    folder_index += 1

os.mkdir(new_folder_path)

# 서비스 객체 생성
chrome_driver_path = ChromeDriverManager().install()
if not chrome_driver_path.endswith(".exe"):
    chrome_driver_path += ".exe"
service = Service(executable_path=chrome_driver_path) 


options = webdriver.ChromeOptions()
service = Service(excutable_path=ChromeDriverManager().install()) 
driver = webdriver.Chrome(service=service)
driver.implicitly_wait(3)

# 트위터 접속, login 페이지
driver.get("https://x.com/login")

# 로그인 대기 시간. 로그인을 수동으로 진행하고, 로그인에 성공하면 아무 키나 누르세요.

print('message: Please log in manually, and press "Enter" key to continue...')

# Wait for any key press
msvcrt.getch()

# 북마크 페이지로 이동
driver.get("https://x.com/i/bookmarks")
time.sleep(2)
img_urls = []
uploader_names = []
upload_times = []
pass_count = 0
error_count = 0

# Initialize before_location
before_location = driver.execute_script("return window.pageYOffset")

# Redirect print statements to both terminal and log file
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")

    def write(self, message):
        if message != '\r':  # Avoid adding newline for progress updates
            self.terminal.write(message)
            self.log.write(message)
        else:
            self.terminal.write(message)
            self.log.write(message)

    def flush(self):
        pass

log_file_path = os.path.join(new_folder_path, "log.txt")
sys.stdout = Logger(log_file_path)
print("message: scrolling down for loading all URLs, please don't touch anything until the process is totally completed")
# Scroll down and collect image URLs
while True:
    for _ in range(20):
        driver.execute_script("window.scrollBy(0, 1000)")
        time.sleep(0.1)
    
    after_location = driver.execute_script("return window.pageYOffset")
    if before_location == after_location:
        print("message: All URLs loaded, Start image URLs collecting")
        break
    else:
        before_location = after_location

# Scroll up and collect image URLs, uploader names, and timestamps
image_data = []
while True:
    try:
        while True:
            new_urls_found = False
            image_elements = driver.find_elements(By.XPATH, "//img[contains(@src, 'media')]")
            for element in image_elements:
                try:
                    url = element.get_attribute("src")
                    new_url = re.sub(r"name=.+", "name=orig", url)
                    
                    parent_element = element.find_element(By.XPATH, "./ancestor::article")
                    uploader_element = parent_element.find_element(By.XPATH, ".//span[contains(text(), '@')]")
                    uploader_name = uploader_element.text

                    time_element = parent_element.find_element(By.XPATH, ".//time")
                    upload_time = time_element.get_attribute("datetime")
                    if upload_time.endswith(".000Z"):
                        upload_time = upload_time[:-5]

                    if new_url not in [data['url'] for data in image_data]:
                        image_data.append({
                            'url': new_url,
                            'uploader_name': uploader_name,
                            'upload_time': upload_time
                        })
                        new_urls_found = True
                except StaleElementReferenceException:
                    continue
                except Exception as e:
                    print(f"message: Error finding uploader name or upload time: {e}")
                    error_count += 1
            
            if not new_urls_found:
                break
            
            driver.execute_script("window.scrollBy(0, -1200)")
            
            after_location = driver.execute_script("return window.pageYOffset")
            if after_location == 0:
                break
        driver.execute_script("window.scrollBy(0, -1200)")
    except StaleElementReferenceException:
        print("message: StaleElementReferenceException encountered. Retrying...")
        continue
    
    after_location = driver.execute_script("return window.pageYOffset")
    if after_location == 0:
        print("message: ALL URLs collected.")
        break

# Remove duplicate URLs
image_data = [dict(t) for t in {tuple(d.items()) for d in image_data}]

# Print collected image URLs
# print("Collected image URLs:")
# for data in image_data:
#     print(data['url'])

# Print the total number of collected URLs
print(f"message: Total number of collected URLs: {len(image_data)}")
print("message: Start downloading images...")

# Function to print progress in place
def print_progress(message):
    sys.stdout.write('\r' + message)
    sys.stdout.flush()

# Create result text file
result_file_path = os.path.join(new_folder_path, "result.txt")
with open(result_file_path, "w", encoding="utf-8") as result_file:
    # Download images
    for index, data in enumerate(image_data, start=1):
        url = data['url']
        uploader_name = data['uploader_name']
        upload_time = data['upload_time']
        
        safe_uploader_name = re.sub(r'[\\/*?:"<>|]', "", uploader_name)
        safe_upload_time = re.sub(r'[\\/*?:"<>|]', "", upload_time)
        file_extension = os.path.splitext(url)[1]
        if not file_extension:
            file_extension = ".png"  # Default to .png if no extension found
        base_file_name = f"{safe_uploader_name}_{safe_upload_time}"
        file_name = f"{new_folder_path}\\{base_file_name}{file_extension}"
        counter = 1
        while os.path.exists(file_name):
            file_name = f"{new_folder_path}\\{base_file_name}_{counter}{file_extension}"
            counter += 1

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(file_name, "wb") as f:
                f.write(response.content)
            print_progress(f"Download complete/TotalURLs/fail : ({index - pass_count}/{len(image_data)}/{pass_count})")
            result_file.write(f"{base_file_name}{file_extension} , URL : {url}\n")
        except requests.exceptions.RequestException as e:
            print(f"message: Failed to download {url}: {e}")
            pass_count += 1

# Calculate the number of successfully downloaded images
successful_downloads = len(image_data) - pass_count
# Print the total number of successfully downloaded images
print(f"\nmessage: Total number of successfully downloaded images: {successful_downloads}")

def move_duplicate_images(directory_path):
    duplicate_folder = os.path.join(directory_path, "duplicates")
    os.makedirs(duplicate_folder, exist_ok=True)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    hash_dict = {}
    duplicate_count = 0

    for dirpath, dirnames, filenames in os.walk(directory_path):
        if dirpath != directory_path and not dirpath.startswith(new_folder_path):
            continue
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)

            if filename.lower().endswith(tuple(image_extensions)):
                try:
                    with open(file_path, 'rb') as f:
                        image_hash = hashlib.md5(f.read()).hexdigest()

                    if image_hash in hash_dict:
                        print(f"message: Duplicate files detail: {file_path}")
                        shutil.move(file_path, os.path.join(duplicate_folder, filename))
                        duplicate_count += 1
                    else:
                        hash_dict[image_hash] = file_path
                except Exception as e:
                    print(f"message: error occured: {file_path}, {e}")
                    error_count += 1
                    continue

    try:
        deleted_files = len(os.listdir(duplicate_folder))
        shutil.rmtree(duplicate_folder)
    except Exception as e:
        print(f"message: Error deleting duplicate folder: {e}")
        error_count += 1
        deleted_files = 0

    return duplicate_count, deleted_files

directory_path = new_folder_path
duplicate_count, deleted_files = move_duplicate_images(directory_path)

# Print the results
print(f"\nmessage: Number of duplicate files: {duplicate_count}")
print(f"message: Number of deleted duplicate files: {deleted_files}")
print(f"message: Number of failed downloads: {pass_count}")
print(f"message: Number of errors: {error_count}")

# Keep the Chrome window open
print("message: Process completed. The Chrome window automatically closed.")

# Close the log file
sys.stdout.log.close()
sys.stdout = sys.stdout.terminal