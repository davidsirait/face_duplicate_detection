# Python Script to download all the image from csv sequentially

import pandas as pd
import numpy as np

import requests
from pathlib import Path
from tqdm import tqdm
import time
import sys
sys.path.append("./src")
from utils.file_utils import sanitize_dataframe
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_session():
    """
    Create requests session with retry logic and proper headers
    """
    session = requests.Session()
    
    # Retry strategy
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Default headers for all requests
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    })
    
    return session

def download_with_retry(session, url, filepath, max_retries=3):
    """
    Download image with retry logic and error handling
    """
    for attempt in range(max_retries):
        try:
            # Add referer from the URL domain
            from urllib.parse import urlparse
            parsed = urlparse(url)
            referer = f"{parsed.scheme}://{parsed.netloc}/"
            
            response = session.get(
                url,
                headers={'Referer': referer},
                timeout=15,
                stream=True,
                allow_redirects=True
            )
            
            # Check status
            if response.status_code == 200:
                # Verify it's actually an image
                content_type = response.headers.get('Content-Type', '')
                if 'image' not in content_type and 'octet-stream' not in content_type:
                    return False, f"Not an image (Content-Type: {content_type})"
                
                # Save file
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify file is not empty
                if filepath.stat().st_size == 0:
                    filepath.unlink()
                    return False, "Downloaded file is empty"
                
                return True, "Success"
            
            elif response.status_code == 403:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return False, "403 Forbidden (blocked)"
            
            elif response.status_code == 404:
                return False, "404 Not Found (broken URL)"
            
            else:
                return False, f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return False, "Timeout"
        
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return False, "Connection Error"
        
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    return False, "Max retries exceeded"

def download_images_robust(csv_file, output_folder='downloaded_images', delay=0.5):
    """
    Robust image downloader with comprehensive error handling
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True) 
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # sanitize the df
    df = sanitize_dataframe(df)
    
    # Create session
    session = create_session()
    
    # Statistics
    stats = {
        'successful': 0,
        'failed_403': 0,
        'failed_404': 0,
        'failed_timeout': 0,
        'failed_other': 0,
        'skipped': 0
    }
    
    # Track failures for report
    failures = []
    
    print(f"\nDownloading {len(df)} images...")
    print(f"Output folder: {output_path.absolute()}\n")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        url = row['url']
        title = row['title'] 
        
        # Get extension
        ext = Path(url).suffix
        if not ext or len(ext) > 5:
            ext = '.jpg'
        
        filename = f"{title}{ext}"
        filepath = output_path / filename
        
        # Skip if exists
        if filepath.exists():
            stats['skipped'] += 1
            continue
        
        # Download
        success, message = download_with_retry(session, url, filepath)
        
        if success:
            stats['successful'] += 1
        else:
            # Track failure reason
            if '403' in message:
                stats['failed_403'] += 1
            elif '404' in message:
                stats['failed_404'] += 1
            elif 'Timeout' in message or 'timeout' in message.lower():
                stats['failed_timeout'] += 1
            else:
                stats['failed_other'] += 1
            
            print(f"Failed download for {url} : {message}")
            
            failures.append({
                'title': title,
                'url': url,
                'error': message
            })
        
        # Be polite to servers
        time.sleep(delay)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Download Complete!")
    print(f"{'='*60}")
    print(f"✓ Successful:    {stats['successful']}")
    print(f"⊘ Skipped:       {stats['skipped']} (already exist)")
    print(f"✗ Failed (403):  {stats['failed_403']}")
    print(f"✗ Failed (404):  {stats['failed_404']}")
    print(f"✗ Failed (timeout): {stats['failed_timeout']}")
    print(f"✗ Failed (other): {stats['failed_other']}")
    print(f"{'='*60}")
    
    # Save failure report
    if failures:
        failures_df = pd.DataFrame(failures)
        failures_path = output_path / 'failed_downloads.csv'
        failures_df.to_csv(failures_path, index=False)
        print(f"\nFailed downloads logged to: {failures_path}")
    
    return stats, failures

if __name__ == "__main__":
    csv_path = "./facescrub_metadata.csv"
    download_path = "./data/images"
    stats, failures = download_images_robust(csv_path, download_path, delay=0.5)