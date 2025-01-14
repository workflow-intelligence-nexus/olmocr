import requests
import re
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode
import json
from collections import defaultdict

def fetch_review_page_html(url):
    """
    Fetch the HTML from the Tinyhost URL.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def extract_presigned_url(html):
    """
    Given the HTML of the page, extract the `presignedGetUrl`.
    Returns None if not found.
    """
    match = re.search(r'const presignedGetUrl = \"(.*?)\";', html)
    if not match:
        return None
    return match.group(1)

def fetch_presigned_datastore(presigned_url):
    """
    Fetch the JSON datastore from the presigned URL.
    Returns a dict. If any error or no content, returns {}.
    """
    try:
        # Clean up the presigned URL (sometimes the signature may need re-encoding)
        url_parts = urlsplit(presigned_url)
        query_params = parse_qs(url_parts.query)
        encoded_query = urlencode(query_params, doseq=True)
        cleaned_url = urlunsplit((
            url_parts.scheme,
            url_parts.netloc,
            url_parts.path,
            encoded_query,
            url_parts.fragment
        ))

        resp = requests.get(cleaned_url, headers={
            "Host": url_parts.netloc,
            "User-Agent": "Mozilla/5.0"
        })
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error fetching datastore from {presigned_url}: {e}")
        return {}
    
def sanitize_key(key):
    return re.sub(r'[^a-zA-Z0-9-_]', '_', key)


def parse_entry_metadata(html):
    """
    Parse each .entry block from the HTML to figure out:
      - data-entry-id
      - left-metadata
      - right-metadata
      - classes that might indicate 'gold', 'eval', etc.
    Returns a dict:
      {
        entry_id_1: {
          'left_metadata': str,
          'right_metadata': str,
          'is_left_gold': bool,
          'is_right_gold': bool,
          'is_left_eval': bool,
          'is_right_eval': bool,
          ...
        },
        ...
      }
    
    Note: This uses a regex that looks for something like:
        <div class="entry gold eval" data-entry-id="..." 
             data-left-metadata="..." data-right-metadata="...">
    Adjust if your HTML changes structure or you need more nuance.
    """
    # This regex attempts to capture:
    #   1) The entire `class="entry ..."` portion (to detect "gold", "eval", etc.)
    #   2) data-entry-id="some_string"
    #   3) data-left-metadata="some_string"
    #   4) data-right-metadata="some_string"
    # 
    # We use a DOTALL and multiline approach, and some non-greedy capturing.
    pattern = re.compile(
        r'<div\s+class="entry([^"]*)"\s+data-entry-id="([^"]+)"\s+data-left-metadata="([^"]+)"\s+data-right-metadata="([^"]+)"',
        re.DOTALL | re.MULTILINE
    )
    
    entries = {}
    for m in pattern.finditer(html):
        class_str = m.group(1).strip()  # e.g. " gold eval"
        entry_id = m.group(2).strip()
        left_md = m.group(3).strip()
        right_md = m.group(4).strip()

        # Transform the HTML's data-entry-id to match the JS datastore keys:
        entry_id = sanitize_key(entry_id)


        # Basic flags
        is_left_gold = " gold" in class_str or class_str.startswith("gold")
        is_right_gold = " gold" in class_str or class_str.startswith("gold")
        is_left_eval = " eval" in class_str or class_str.startswith("eval")
        is_right_eval = " eval" in class_str or class_str.startswith("eval")

        # The above checks are naive because the template often has: 
        #    class="entry gold eval"
        # for the entire entry, meaning it might not tell you *which side* is gold/eval.
        # 
        # If your template uses "left_class=gold" or "right_class=eval" etc. 
        # you might need to refine your approach:
        # For instance, we see the code uses separate classes on the text blocks themselves:
        #    <div class="text-block {{ entry.left_class }}" data-choice="left">
        # So you may want to parse those sub-blocks if you need the actual side. 
        #
        # For demonstration, we'll just store the raw class string plus the metadata:
        # 
        # If the real logic is "gold" is always left, "eval" is always right, you can adapt here.

        entries[entry_id] = {
            "class_str": class_str,
            "left_metadata": left_md,
            "right_metadata": right_md,
        }
    return entries

def build_comparison_report(entries_dict, datastore):
    """
    Build a comparison report showing how often each type of method
    beats each other type of method, based on user votes in the datastore.
    
    We assume:
      - If user vote is 'left', then left_metadata's method "wins".
      - If user vote is 'right', then right_metadata's method "wins".
      - If user vote is 'both_good', 'both_bad', or 'invalid_pdf', we do not count it as a direct matchup.
    
    Returns a nested dict or a summary of (methodA, methodB) => { "A_wins": x, "B_wins": y }.
    """
    print(entries_dict)
    
    print("")

    print(datastore)

    print("")

    # comparisons[ (methodA, methodB) ] = [A_wins, B_wins]
    comparisons = defaultdict(lambda: [0, 0])  

    for entry_id, vote in datastore.items():
        if entry_id not in entries_dict:
            # No matching <div> found for this key in the HTML
            continue
        
        left_method = entries_dict[entry_id]["left_metadata"]
        right_method = entries_dict[entry_id]["right_metadata"]
        
        if vote == "left":
            # left_method beats right_method
            if left_method != right_method:
                # Use a sorted tuple so the pair is always in a consistent order
                pair = tuple(sorted([left_method, right_method]))
                if pair[0] == left_method:
                    comparisons[pair][0] += 1  # left_method is pair[0]
                else:
                    comparisons[pair][1] += 1  # left_method is pair[1]
        
        elif vote == "right":
            # right_method beats left_method
            if left_method != right_method:
                pair = tuple(sorted([left_method, right_method]))
                if pair[0] == right_method:
                    comparisons[pair][0] += 1
                else:
                    comparisons[pair][1] += 1
        
        else:
            # "both_good", "both_bad", "invalid_pdf", etc. 
            # Not counted as a direct head-to-head winner.
            pass


    # Build a more readable summary
    # comparisons[(A, B)] = [A_wins, B_wins], where A < B lexicographically in that tuple
    results = []
    for (A, B), (A_wins, B_wins) in comparisons.items():
        total = A_wins + B_wins
        A_rate = (A_wins / total) if total else 0
        B_rate = (B_wins / total) if total else 0
        results.append({
            "methodA": A,
            "methodB": B,
            "A_wins": A_wins,
            "B_wins": B_wins,
            "A_win_rate": f"{100 * A_rate:.1f}%",
            "B_win_rate": f"{100 * B_rate:.1f}%",
        })
    return results

def make_report(urls):
    """
    Main function that:
      - Fetches each HTML page
      - Extracts presignedGetUrl
      - Fetches the JSON datastore
      - Parses .entry blocks for metadata
      - Produces an overall "win rate" report for each method vs method
    """
    # Aggregate all entries from all URLs into a single dict
    # so each entry_id is unique across all pages (they usually are).
    master_entries_dict = {}
    master_datastore = {}

    for url in urls:
        try:
            html = fetch_review_page_html(url)
        except Exception as e:
            print(f"Error fetching HTML from {url}: {e}")
            continue
        
        # Extract the presignedGetUrl
        presigned_url = extract_presigned_url(html)
        if not presigned_url:
            print(f"Warning: Could not find presignedGetUrl in {url}")
            continue
        
        # Fetch the datastore
        datastore = fetch_presigned_datastore(presigned_url)
        
        # Parse the HTML for entry metadata
        entries_dict = parse_entry_metadata(html)
        
        # Merge into master
        for k, v in entries_dict.items():
            master_entries_dict[k] = v
        for k, v in datastore.items():
            master_datastore[k] = v

    # Now build the comparison report
    report = build_comparison_report(master_entries_dict, master_datastore)

    print("=== Comparison Report (Win Rates) ===")
    if not report:
        print("No head-to-head comparisons found (did not find left/right votes).")
        return
    
    # Print out each matchup
    for row in report:
        A = row["methodA"]
        B = row["methodB"]
        A_wins = row["A_wins"]
        B_wins = row["B_wins"]
        A_win_rate = row["A_win_rate"]
        B_win_rate = row["B_win_rate"]
        
        print(
            f"{A} vs {B}: "
            f"{A} wins={A_wins} ({A_win_rate}), "
            f"{B} wins={B_wins} ({B_win_rate})"
        )

if __name__ == "__main__":
    # Example usage
    urls = [
        "https://jakep-tinyhost.s3.amazonaws.com/review_page-681aae527593.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=BR1nqCUKQLBlh3HIsHjeyRVQumI%3D&Expires=1737500018",
        # Add more URLs here...
    ]
    make_report(urls)
