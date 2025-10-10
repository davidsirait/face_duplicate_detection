# Python Script to download all the image from csv asynchronously

import asyncio
import aiohttp
import pandas as pd
from pathlib import Path
from tqdm.asyncio import tqdm
import sys
sys.path.append("./src")
from utils.file_utils import sanitize_dataframe
from urllib.parse import urlparse

async def download_image_async(session, url, filepath, semaphore, retry_count=1):
    """
    Download single image asynchronously with retry logic
    """
    async with semaphore:  # Limit concurrent connections
        for attempt in range(retry_count):
            try:
                # Set referer from URL domain
                parsed = urlparse(url)
                referer = f"{parsed.scheme}://{parsed.netloc}/"
                
                headers = {
                    'Referer': referer,
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                }
                
                async with session.get(
                    url, 
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                    allow_redirects=True
                ) as response:
                    
                    if response.status == 200:
                        # Check content type
                        content_type = response.headers.get('Content-Type', '')
                        if 'image' not in content_type and 'octet-stream' not in content_type:
                            return False, f"Not an image ({content_type})", url
                        
                        # Read and save
                        content = await response.read()
                        
                        if len(content) == 0:
                            return False, "Empty file", url
                        
                        # Save to file
                        async with asyncio.Lock():
                            with open(filepath, 'wb') as f:
                                f.write(content)
                        
                        return True, "Success", url
                    
                    elif response.status == 403:
                        if attempt < retry_count - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return False, "403 Forbidden", url
                    
                    elif response.status == 404:
                        return False, "404 Not Found", url
                    
                    else:
                        return False, f"HTTP {response.status}", url
            
            except asyncio.TimeoutError:
                if attempt < retry_count - 1:
                    await asyncio.sleep(1)
                    continue
                return False, "Timeout", url
            
            except aiohttp.ClientError as e:
                if attempt < retry_count - 1:
                    await asyncio.sleep(2)
                    continue
                return False, f"Connection error: {str(e)}", url
            
            except Exception as e:
                return False, f"Error: {str(e)}", url
        
        return False, "Max retries exceeded", url

async def download_images_async(csv_file, output_folder='downloaded_images', 
                                max_concurrent=10, delay_between_batches=0):
    """
    Download images asynchronously with controlled concurrency
    
    Args:
        csv_file: Path to CSV file
        output_folder: Output directory
        max_concurrent: Maximum concurrent downloads (default: 10)
        delay_between_batches: Delay in seconds between batches (default: 0)
    """
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True) 
    
    # Read CSV
    df = pd.read_csv(csv_file)

    # sanitize the dataframe
    df = sanitize_dataframe(df)

    print(f"\nStarting async download of {len(df)} images...")
    print(f"Max concurrent downloads: {max_concurrent}")
    print(f"Output folder: {output_path.absolute()}\n")
    
    # Statistics
    stats = {
        'successful': 0,
        'failed_403': 0,
        'failed_404': 0,
        'failed_timeout': 0,
        'failed_other': 0,
        'skipped': 0
    }
    failures = []
    
    # Semaphore to limit concurrent connections
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Configure session with proper headers and connection pooling
    connector = aiohttp.TCPConnector(
        limit=max_concurrent,
        limit_per_host=5,
        ttl_dns_cache=300
    )
    
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    
    async with aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
        }
    ) as session:
        
        # Create download tasks
        tasks = []
        
        for _, row in df.iterrows():
            url = row['url']
            title = row['title']
            
            ext = Path(url).suffix or '.jpg'
            filename = f"{title}{ext}"
            filepath = output_path / filename
            
            # Skip if exists
            if filepath.exists():
                stats['skipped'] += 1
                continue
            
            # Create task
            task = download_image_async(session, url, filepath, semaphore)
            tasks.append((task, title, url))
        
        # Execute all tasks with progress bar
        results = []
        for task, title, url in tqdm(tasks, desc="Downloading"):
            result = await task
            results.append((result, title))
            
            # Optional delay between batches
            if delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)
        
        # Process results
        for (success, message, url), title in results:
            if success:
                stats['successful'] += 1
            else:
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
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Async Download Complete!")
    print(f"{'='*60}")
    print(f"✓ Successful:        {stats['successful']}")
    print(f"⊘ Skipped:           {stats['skipped']}")
    print(f"✗ Failed (403):      {stats['failed_403']}")
    print(f"✗ Failed (404):      {stats['failed_404']}")
    print(f"✗ Failed (timeout):  {stats['failed_timeout']}")
    print(f"✗ Failed (other):    {stats['failed_other']}")
    print(f"{'='*60}")
    
    # Save failure report
    if failures:
        failures_df = pd.DataFrame(failures)
        failures_path = output_path / 'failed_downloads.csv'
        failures_df.to_csv(failures_path, index=False)
        print(f"\nFailed downloads logged to: {failures_path}")
    
    return stats, failures

# Usage
async def main(csv_filepath, download_path):
    stats, failures = await download_images_async(
        csv_filepath,
        download_path,
        max_concurrent=30,
        delay_between_batches=0
    )

# Run it
if __name__ == "__main__":
    csv_filepath = "./facescrub_metadata.csv"
    download_path = "./data/images"
    asyncio.run(main(csv_filepath, download_path))