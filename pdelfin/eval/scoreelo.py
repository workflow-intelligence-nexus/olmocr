import requests
import re
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode
import csv
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
          'class_str': str,
        },
        ...
      }
    
    Note: This uses a regex that looks for something like:
        <div class="entry gold eval" data-entry-id="..." 
             data-left-metadata="..." data-right-metadata="...">
    """
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
      - If user vote is 'both_good', 'both_bad', or 'invalid_pdf', 
        we do not count it as a direct matchup.
    
    Returns a structure:
      comparisons[(A, B)] = [A_wins, B_wins],
        where A < B lexicographically in that tuple.
    """
    comparisons = defaultdict(lambda: [0, 0])  

    for entry_id, vote in datastore.items():
        if entry_id not in entries_dict:
            # No matching <div> found for this key in the HTML
            continue
        
        left_method = entries_dict[entry_id]["left_metadata"]
        right_method = entries_dict[entry_id]["right_metadata"]
        
        if left_method == right_method:
            # Same "method" on both sides => skip
            continue
        
        if vote == "left":
            # left_method beats right_method
            pair = tuple(sorted([left_method, right_method]))
            if pair[0] == left_method:
                comparisons[pair][0] += 1
            else:
                comparisons[pair][1] += 1
        
        elif vote == "right":
            # right_method beats left_method
            pair = tuple(sorted([left_method, right_method]))
            if pair[0] == right_method:
                comparisons[pair][0] += 1
            else:
                comparisons[pair][1] += 1
        
        else:
            # "both_good", "both_bad", "invalid_pdf", etc. -> not counted
            pass

    return comparisons

def elo_update(ratingA, ratingB, scoreA, scoreB, k=32):
    """
    Perform a single ELO update for a match between A and B.
      - ratingA, ratingB are current ELO ratings of A and B.
      - scoreA, scoreB in {0 or 1} (1 if the player is the winner, 0 if loser).
      - Returns (new_ratingA, new_ratingB).
    """
    # Expected scores for each player
    expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400))
    expectedB = 1 / (1 + 10 ** ((ratingA - ratingB) / 400))

    new_ratingA = ratingA + k * (scoreA - expectedA)
    new_ratingB = ratingB + k * (scoreB - expectedB)
    return new_ratingA, new_ratingB

def compute_elo_arena(comparisons, k=32, initial_rating=1500):
    """
    Given the aggregated comparisons dict:
      comparisons[(A, B)] = [A_wins, B_wins]

    1) Collect all unique methods.
    2) Initialize them to initial_rating (1500).
    3) For each pair (A, B), apply A_wins times the scenario 
       "A beats B" -> ELO update
       B_wins times the scenario "B beats A" -> ELO update

    Because we don't have a strict order of matches, we just 
    apply them in some consistent but arbitrary order.

    Returns a dict { method_name: final_elo_rating }
    """
    # 1) Collect all unique methods
    methods = set()
    for (A,B) in comparisons.keys():
        methods.add(A)
        methods.add(B)

    # 2) Initialize ratings
    ratings = {m: float(initial_rating) for m in methods}

    # 3) Walk through each pair
    for (A, B), (A_wins, B_wins) in comparisons.items():
        for _ in range(A_wins):
            # A beats B
            oldA = ratings[A]
            oldB = ratings[B]
            newA, newB = elo_update(oldA, oldB, 1, 0, k=k)
            ratings[A] = newA
            ratings[B] = newB
        
        for _ in range(B_wins):
            # B beats A
            oldA = ratings[A]
            oldB = ratings[B]
            newA, newB = elo_update(oldA, oldB, 0, 1, k=k)
            ratings[A] = newA
            ratings[B] = newB

    return ratings

def make_report(urls):
    """
    Main function that:
      - Fetches each HTML page
      - Extracts presignedGetUrl
      - Fetches the JSON datastore
      - Parses .entry blocks for metadata
      - Produces an overall "win rate" report for each method vs method
      - Produces an ELO arena result for each method
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
    comparisons = build_comparison_report(master_entries_dict, master_datastore)
    
    print("=== Pairwise Win/Loss Report ===")
    if not comparisons:
        print("No head-to-head comparisons found (did not find left/right votes).")
        return
    
    # Print out each matchup
    for (A, B), (A_wins, B_wins) in comparisons.items():
        total = A_wins + B_wins
        A_rate = A_wins / total * 100 if total else 0
        B_rate = B_wins / total * 100 if total else 0
        print(
            f"{A} vs {B}: "
            f"{A} wins={A_wins} ({A_rate:.1f}%), "
            f"{B} wins={B_wins} ({B_rate:.1f}%)"
        )

    # -- ADDED: Write the same data to scoreelo.csv
    with open("scoreelo.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["MethodA", "MethodB", "A_wins", "B_wins", "A_rate(%)", "B_rate(%)"])
        for (A, B), (A_wins, B_wins) in comparisons.items():
            total = A_wins + B_wins
            A_rate = A_wins / total * 100 if total else 0
            B_rate = B_wins / total * 100 if total else 0
            writer.writerow([A, B, A_wins, B_wins, f"{A_rate:.1f}", f"{B_rate:.1f}"])


    # ==== ELO Arena ====
    elo_ratings = compute_elo_arena(comparisons, k=32, initial_rating=1500)
    
    # Sort methods by final rating descending
    sorted_ratings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== ELO Arena Results ===")
    for method, rating in sorted_ratings:
        print(f"{method}: {rating:.2f}")

if __name__ == "__main__":
    # Example usage
    urls = [
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_0-e09ebadf34a7.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=JEQpJxSaMIHuc9DFHyfHuxx0dEU%3D&Expires=1737654586",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_1-c2d267f97a73.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=KMiOTQiFEvgxU94ZrlJRFAgSQZA%3D&Expires=1737654587",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_10-b806c811fb67.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=NaoHNU2ZmEGrgMsxg2JHK%2Fv5zd0%3D&Expires=1737654587",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_11-19c1936b4372.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=BjkVydyKjzzH3uZiZ1GkWAk6cbk%3D&Expires=1737654588",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_12-cd41808a7974.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=jsk8TzJTKJwHi1Ru4%2Bw%2BiHZG638%3D&Expires=1737654589",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_13-8b055079b5eb.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=SE7kkobEBip44O8JY5axoMTV2Bs%3D&Expires=1737654590",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_14-1126e0da563c.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=jLSEWpDUzpmS8P9mNXbBoDYDOwU%3D&Expires=1737654590",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_15-05704e3d000d.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=TaCbyv2%2FDGCnCOgTzUvfEXdO%2Fmo%3D&Expires=1737654591",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_16-e57f795a89da.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=stqm1etAfDIpAQGNvZwe9c%2BYUbA%3D&Expires=1737654592",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_17-041a6d042764.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=rOTroBcSqCh3oM65bOJHEfaeal8%3D&Expires=1737654592",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_18-7a29697cee63.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=abmYM9KtzjicmdacRykPWXCdQr0%3D&Expires=1737654593",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_19-d32f14c067f8.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=iVg3nxrZXVpYybkLJIgOEJ3v37E%3D&Expires=1737654594",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_2-43c553548e69.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=IK27gl7b6NY05YNnnsimMVJc99I%3D&Expires=1737654595",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_3-fb42a458ecd5.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=d1qevJe8ZQONnu7zezYSJe3cbBw%3D&Expires=1737654595",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_4-76a50eed331a.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=qwZu2q1H4Y%2Bf3Kw7DNSYcTxwI7A%3D&Expires=1737654596",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_5-150b4d3583de.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=c%2FeqjnDSIRirgQviFWRLWVowKmA%3D&Expires=1737654597",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_6-6ca285526fd3.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=tkWDDuRinY77BLQCqumtlMiFJU8%3D&Expires=1737654598",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_7-01d711ee8bf7.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=eQtFo6CHJYHGu85wK0YG5khlE5U%3D&Expires=1737654598",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_8-0f36b852f274.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=weI3WB8vhjBYjk6t85DmyLdP97k%3D&Expires=1737654599",
        "https://jakep-tinyhost.s3.amazonaws.com/review_page_9-115e33463fd2.html?AWSAccessKeyId=AKIASHLPW4FEVZOPGK46&Signature=b4CpkHprCUtZoL0u%2FFYzsu%2BB1yU%3D&Expires=1737654600",
    ]
    make_report(urls)
