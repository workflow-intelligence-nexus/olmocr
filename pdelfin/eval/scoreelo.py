# TODO Takes in a list of tinyhost urls as arguments
# ex https://jakep-tinyhost.s3.amazonaws.com/review_page-a1617c2734b2.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=NEsAN69b98Z%2BqDR23zmQKu%2B5IHM%3D&Expires=1737496145

# Extracts out the presignedGetUrl from the source code,
# const presignedGetUrl = "https://jakep-tinyhost.s3.amazonaws.com//etSe2zObhx1hpcO7TcS7.json?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=bl0wav%2BDqXL5%2FCo12Mmu2Sm0gGQ%3D&Expires=1737496145";
# And gets the contents of this page

# Next, get's all the votes, figures out what they match to

# Given all the votes calculates the ELO score

import requests
import re
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode
import logging

logging.basicConfig(level=logging.DEBUG)

def fetch_presigned_content(urls):
    """
    Extracts the `presignedGetUrl` from the source code of the given URLs and fetches the content of the URL.

    Args:
        urls (list): List of tinyhost URLs.

    Returns:
        dict: A dictionary mapping the original URL to the content of the `presignedGetUrl`.
    """
    results = {}

    for url in urls:
        try:
            # Fetch the source code of the page
            response = requests.get(url)
            response.raise_for_status()
            source_code = response.text

            # Extract the presignedGetUrl using a regular expression
            match = re.search(r'const presignedGetUrl = \"(.*?)\";', source_code)
            if not match:
                print(f"No presignedGetUrl found in {url}")
                results[url] = None
                continue

            presigned_url = match.group(1)

            # Fetch the content of the presigned URL
            print(presigned_url)
            # Step 1: Split the URL into components
            url_parts = urlsplit(presigned_url)

            # Step 2: Parse query parameters
            query_params = parse_qs(url_parts.query)

            print(query_params)
            # Step 3: Re-encode the query parameters properly
            encoded_query = urlencode(query_params, doseq=True)

            # Step 4: Rebuild the URL with the cleaned query string
            cleaned_url = urlunsplit((url_parts.scheme, url_parts.netloc, url_parts.path, encoded_query, url_parts.fragment))

            print("Cleaned URL:", cleaned_url)

            presigned_response = requests.get(presigned_url, headers={"Host": "jakep-tinyhost.s3.amazonaws.com",
                                                                       "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"})
            presigned_response.raise_for_status()

            # Store the content in the results dictionary
            results[url] = presigned_response.text
        except requests.RequestException as e:
            print(f"Error fetching data from {url} or its presigned URL: {e}")
            results[url] = None

    return results

# Example usage
urls = [
    "https://jakep-tinyhost.s3.amazonaws.com/review_page-59c2f52d9bf3.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=UPIEQMLEXWG%2BpAkvm7YJrrEIgnI%3D&Expires=1737499054"
]

content_map = fetch_presigned_content(urls)

for original_url, content in content_map.items():
    print(f"Content fetched from presigned URL in {original_url}:")
    print(content)
