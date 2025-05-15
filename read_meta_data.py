"""
Helper module to fetch arXiv abstracts or metadata using a unified interface.
Includes detailed debug prints to trace API queries and responses.
"""
import re
import requests
import xml.etree.ElementTree as ET

# ArXiv API endpoint and XML namespaces for query mode
ATOM_URL = 'http://export.arxiv.org/api/query'
NS = {
    'atom': 'http://www.w3.org/2005/Atom',
    'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
}


def fetch_arxiv_batch(id_regex=None,
                      start_date=None,  # YYYY-MM-DD
                      end_date=None, # YYYY-MM-DD
                      category=None,
                      formats="metadata",
                      batch_size: int = 200,
                      output_dir: str = None,
                      verbose: bool = False):
    """
    Fetch arXiv items by filtering IDs and retrieving requested formats.

    id_regex    : str (regex) to match IDs; e.g. '^2309\.00543v1$'
    start_date  : 'YYYY-MM-DD' publication start
    end_date    : 'YYYY-MM-DD' publication end
    category    : arXiv category filter
    formats     : single or list of 'metadata','abstract','pdf','html','source'
    batch_size  : API pagination size
    output_dir  : save files here if provided
    verbose     : print debug

    Returns dict[id -> {format: data_or_path, ...}]
    """
    # Normalize formats to list
    if isinstance(formats, str):
        formats = [formats]
    # Compile regex or exact ID
    pat = re.compile(id_regex) if id_regex else None

    # ===== Stage 1: Filter IDs =====
    id_meta = {}
    # fallback API by date/category
    if not (start_date and end_date and category):
        raise RuntimeError("Must supply start_date, end_date, and category, or use id-only regex.")
    # get total results
    lb = start_date.replace('-', '') + '0000'
    ub = end_date.replace('-', '') + '2359'
    total = 100 #_get_total_results(category, start_date, end_date)
    fetched = 0
    while fetched < total:
        chunk = min(batch_size, total - fetched)
        params = {
            'search_query': f"cat:{category}+AND+submittedDate:[{lb}+TO+{ub}]",
            'start': fetched,
            'max_results': chunk,
            'sortBy': 'submittedDate',
            'sortOrder': 'ascending'
        }
        if verbose:
            print(f"[DEBUG] Metadata API params: {params}")
        r = requests.get(ATOM_URL, params=params)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        entries = root.findall('atom:entry', NS)
        for e in entries:
            aid = e.find('atom:id', NS).text.rsplit('/', 1)[-1]
            if pat and not pat.match(aid):
                continue
            pub = e.find('atom:published', NS).text[:10]
            cats = [c.attrib['term'] for c in e.findall('atom:category', NS)]
            if not (start_date <= pub <= end_date and category in cats):
                continue
            id_meta[aid] = {
                'id': aid,
                'published': pub,
                'categories': cats,
                'title': e.find('atom:title', NS).text.strip(),
                'abstract': e.find('atom:summary', NS).text.strip()
            }
        fetched += len(entries)
        if not entries:
            break

    # Early return for metadata-only
    return id_meta


res = fetch_arxiv_batch(start_date='2011-01-01', end_date='2024-05-01', category='math.CO')




import arxivloader
import pandas as pd

submittedDate = "[2000010100000+TO+202412310000]"
query = "search_query=submittedDate:{sd}".format(sd=submittedDate)

df = arxivloader.load(query, sortBy="submittedDate", sortOrder="ascending",
                      columns=["id", "title", "summary", "authors", "primary_category", "categories"],
                      page_size=10000, num=50000, timeout=90000)

# df.to_csv(r"C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\data\arxiv_metadata.csv", index=False)

# tmp = pd.read_csv(r"C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\data\arxiv_metadata.csv")
