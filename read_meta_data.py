"""
Helper module to fetch arXiv abstracts or metadata using a unified interface.
Includes detailed debug prints to trace API queries and responses.
"""
import re
import requests
import xml.etree.ElementTree as ET

# ArXiv API endpoint and XML namespaces for query mode
ATOM_URL = 'http://export.arxiv.org/api/query'
NS = {'atom': 'http://www.w3.org/2005/Atom',
    'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'}
ARXIV_NS = {'arxiv': 'http://arxiv.org/schemas/atom'}


def fetch_arxiv_batch(id_regex=None,
                      start_date=None,  # YYYY-MM-DD
                      end_date=None,  # YYYY-MM-DD
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
    total = 100  # _get_total_results(category, start_date, end_date)
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

# ===== File: extract_arxiv_data.py =====
# !/usr/bin/env python3
"""
Helper module to fetch arXiv abstracts or metadata using a unified interface.
Includes detailed debug prints to trace API queries and responses.
"""
import re
import requests
import time
import xml.etree.ElementTree as ET
import urllib.request as urlrequest
import urllib.parse as urlparse
from urllib.error import HTTPError


def list_arxiv_ids(set_name=None, from_date=None, until_date=None,
                   max_retries=5):
    """List and cache ArXiv IDs via OAI-PMH into JSON under data_dir."""
    parts = [
        (set_name or "all").replace(":", "_"),
        from_date or "min",
        until_date or "max"
    ]

    print(f">>> Fetching IDs: set={set_name}, from={from_date}, until={until_date}")
    base = "http://export.arxiv.org/oai2"
    params = {"verb": "ListIdentifiers", "metadataPrefix": "oai_dc"}
    if set_name:
        params["set"] = set_name
    if from_date:
        params["from"] = from_date
    if until_date:
        params["until"] = until_date

    ns = {"oai": "http://www.openarchives.org/OAI/2.0/"}
    ids = []
    res_token = None
    batch = 0

    while True:
        batch += 1
        # retry on 503
        query = {"verb": "ListIdentifiers", "resumptionToken": res_token} if res_token else params
        # retry on HTTP 503
        for attempt in range(1, max_retries + 1):
            url = base + "?" + urlparse.urlencode(query)
            try:
                with urlrequest.urlopen(url) as resp:
                    xml = resp.read()
            except HTTPError as e:
                if e.code == 503:
                    wait = int(e.headers.get("Retry-After", "10"))
                    print(f">>> 503, sleeping {wait}s (attempt {attempt}/{max_retries})")
                    time.sleep(wait)
                    continue
                else:
                    raise
            break
        else:
            raise RuntimeError("Too many 503 errors from arXiv OAI-PMH")

        print(f">>> Batch {batch}: fetched {len(ids)} IDs so far")
        root = ET.fromstring(xml)
        headers = root.findall(".//oai:header", ns)
        if not headers:
            break


def filter_ids_by_regex(id_regex,
                        from_date=None,
                        until_date=None,
                        verbose=False):
    """
    Returns two lists:
      - ids_no_version: base IDs matching the regex (e.g. '2309.00543')
      - ids_with_version: versioned IDs to fetch (e.g. '2309.00543v1', or multiple if regex allows)
    """
    # compile the user pattern
    pat = re.compile(id_regex)

    # load all IDs (cached or fresh)
    all_ids = list_arxiv_ids(
        set_name=None,
        from_date=from_date,
        until_date=until_date,
    )
    if verbose:
        print(f">>> filter_ids_by_regex: loaded {len(all_ids)} IDs")

    # detect if regex mentions a version suffix "v<number>"
    ver_match = re.search(r'v(\d+)', id_regex)
    if ver_match:
        ver = ver_match.group(0)  # e.g. "v1"
        # strip the version part out of the regex to match the base IDs
        base_pattern = id_regex.replace(ver, '').rstrip('$')
        pat_base = re.compile(base_pattern)
        # 1) find base IDs
        ids_no_version = [aid for aid in all_ids if pat_base.match(aid)]
        # 2) re-apply the full regex to generate versioned IDs
        ids_with_version = [
            f"{aid}{ver}"
            for aid in ids_no_version
            if pat.match(f"{aid}{ver}")
        ]
    else:
        # no version in regex → assume v1
        ids_no_version = [aid for aid in all_ids if pat.match(aid)]
        ids_with_version = [f"{aid}v1" for aid in ids_no_version]

    if verbose:
        print(f">>> filter_ids_by_regex: {len(ids_no_version)} bases, "
              f"{len(ids_with_version)} versioned IDs")
    return ids_no_version, ids_with_version


# Helper to query total results for a given category and date range
def _get_total_results(category, start_date, end_date, verbose=False):
    """
    Return the total number of matching papers via arXiv API.
    """
    lb = start_date.replace('-', '') + '0000'
    ub = end_date.replace('-', '') + '2359'
    q = f"cat:{category}+AND+submittedDate:[{lb}+TO+{ub}]"
    if verbose:
        print(f"[DEBUG] _get_total_results query: {q}")
    resp = requests.get(ATOM_URL, params={'search_query': q, 'start': 0, 'max_results': 0})
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    el = root.find('opensearch:totalResults', NS)
    return int(el.text) if el is not None else 0


def fetch_arxiv_batch(id_regex=None,
                      start_date=None,
                      end_date=None,
                      category=None,
                      formats="metadata",
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
    formats = list(set(formats))  # get unique

    # Compile regex or exact ID
    pat = re.compile(id_regex) if id_regex else None

    # ===== Stage 1: Filter IDs =====
    # 1a) Filter base and versioned IDs via helper
    if verbose:
        print(f"[DEBUG] Filtering IDs with regex {id_regex!r} and date range {start_date} to {end_date}")
    ids_no_ver, ids_with_ver = filter_ids_by_regex(
        id_regex=id_regex,
        from_date=start_date,
        until_date=end_date,
        verbose=verbose
    )
    if verbose:
        print(f"[DEBUG] {len(ids_no_ver)} base IDs, {len(ids_with_ver)} versioned IDs to process")
    # 1b) Now fetch metadata for each versioned ID and apply date/category filters
    id_meta = {}
    for aid in ids_with_ver:
        meta = None
        # First try snapshot (bare ID match)
        base_id = aid.split('v')[0]
        # Fallback: use id_list param to fetch exact version
        if meta is None:
            if verbose:
                print(f"[DEBUG] Fetching metadata via id_list for {aid}")
            resp = requests.get(
                ATOM_URL,
                params={'id_list': aid}
            )
            if verbose:
                print(f"[DEBUG] GET {resp.url} -> {resp.status_code}")
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            e = root.find('atom:entry', NS)
            if e is not None:
                meta = {
                    'id': aid,
                    'published': e.find('atom:published', NS).text,
                    'updated': e.find('atom:updated', NS).text,
                    'title': e.find('atom:title', NS).text.strip(),
                    'abstract': e.find('atom:summary', NS).text.strip(),
                    # all authors as a list of strings
                    'authors': [a.find('atom:name', NS).text
                                for a in e.findall('atom:author', NS)],
                    # all categories (primary + secondary)
                    'categories': [c.attrib['term']
                                   for c in e.findall('atom:category', NS)],
                    # arXiv‐specific metadata (may be None)
                    'primary_category': e.find('arxiv:primary_category', ARXIV_NS)
                    .attrib.get('term', None),
                    'comment': (e.find('arxiv:comment', ARXIV_NS).text
                                if e.find('arxiv:comment', ARXIV_NS) is not None
                                else None),
                    'journal_ref': (e.find('arxiv:journal_ref', ARXIV_NS).text
                                    if e.find('arxiv:journal_ref', ARXIV_NS) is not None
                                    else None),
                    'doi': (e.find('arxiv:doi', ARXIV_NS).text
                            if e.find('arxiv:doi', ARXIV_NS) is not None
                            else None),
                    # all <link> elements (e.g. HTML, PDF) as {'rel':…, 'href':…}
                    'links': [{'rel': l.attrib.get('rel'),
                               'href': l.attrib.get('href')}
                              for l in e.findall('atom:link', NS)]
                }

            elif verbose:
                print(f"[DEBUG] No metadata entry for {aid}")
        # apply additional filters if metadata found
        if meta:
            pub = meta.get('published', '')[:10]
            cats = meta.get('categories', [])
            if ((not start_date or start_date <= pub <= end_date) and
                    (not category or category in cats)):
                id_meta[aid] = meta
                if verbose:
                    print(f"[DEBUG] Accepted {aid}: pub={pub}, cats={cats}")
    id_meta = {}
    for aid in ids_with_ver:
        # load either from local snapshot or API query
        meta = None
        # fetch single-entry via search_query
        #            resp = requests.get(ATOM_URL, params={'search_query': f'id:{aid}', 'start': 0, 'max_results': 1})
        resp = requests.get(ATOM_URL, params={'id_list': aid})
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        e = root.find('atom:entry', NS)
        if e is None:
            if verbose: print(f"[DEBUG] No metadata entry for {aid}")
        else:
            meta = {
                'id': aid,
                'published': e.find('atom:published', NS).text,
                'updated': e.find('atom:updated', NS).text,
                'title': e.find('atom:title', NS).text.strip(),
                'abstract': e.find('atom:summary', NS).text.strip(),
                # all authors as a list of strings
                'authors': [a.find('atom:name', NS).text
                            for a in e.findall('atom:author', NS)],
                # all categories (primary + secondary)
                'categories': [c.attrib['term']
                               for c in e.findall('atom:category', NS)],
                # arXiv‐specific metadata (may be None)
                'primary_category': e.find('arxiv:primary_category', ARXIV_NS)
                .attrib.get('term', None),
                'comment': (e.find('arxiv:comment', ARXIV_NS).text
                            if e.find('arxiv:comment', ARXIV_NS) is not None
                            else None),
                'journal_ref': (e.find('arxiv:journal_ref', ARXIV_NS).text
                                if e.find('arxiv:journal_ref', ARXIV_NS) is not None
                                else None),
                'doi': (e.find('arxiv:doi', ARXIV_NS).text
                        if e.find('arxiv:doi', ARXIV_NS) is not None
                        else None),
                # all <link> elements (e.g. HTML, PDF) as {'rel':…, 'href':…}
                'links': [{'rel': l.attrib.get('rel'),
                           'href': l.attrib.get('href')}
                          for l in e.findall('atom:link', NS)]
            }
        # apply additional filters
        if meta:
            pub = meta.get('published', '')[:10]
            cats = meta.get('categories', [])
            if ((not start_date or start_date <= pub <= end_date) and
                    (not category or category in cats)):
                id_meta[aid] = meta
                if verbose: print(f"[DEBUG] Accepted {aid}: pub={pub}, cats={cats}")
    # Early return for metadata-only
    if formats == ['metadata']:
        return id_meta

        # ===== Stage 2: Retrieve requested formats =====
    results = {}
    for aid, meta in id_meta.items():
        if verbose:
            print(f"[DEBUG] Processing {aid}")
        entry = {}
        if 'metadata' in formats:
            entry['metadata'] = meta
        if 'abstract' in formats:
            entry['abstract'] = meta.get('abstract', '')
        results[aid] = entry
    return results

if __name__ == "__main__":

    # interactive example: single paper, extract all formats
    example_id = '2309.00543v1'
    print(f"=== Interactive example: ID={example_id} ===")
    id_regex = f'^{example_id}$'
    start_date = None
    end_date = None
    category = None
    formats = 'metadata'
    verbose = True

    # Single unified call
    data = fetch_arxiv_batch(
        id_regex=id_regex,
        start_date=start_date,
        end_date=end_date,
        category=category,
        formats=formats,
        verbose=verbose
    )

    # Print snippets for each paper
    for aid, entry in data.items():
        print(f"== = {aid} == = ")
        if 'metadata' in formats:
            md = entry.get('metadata', {})
        print("Title:", md.get('title'))
        print("Authors:", ', '.join(md.get('authors', [])))
        abs_txt = md.get('abstract', '')
        print("Abstract (first 3 lines):")
        for ln in abs_txt.splitlines()[:3]: print("  ", ln)
        if 'abstract' in formats:
            abs_txt = entry.get('abstract', '')
        print("Abstract (first 3 lines):")
        for ln in abs_txt.splitlines()[:3]: print("  ", ln)

if __name__ == "__main__":
    import sys, argparse

    # interactive example: single paper, extract all formats
    example_id = '2309.00543v1'
    print(f"=== Interactive example: ID={example_id} ===")
    id_regex = f'^{example_id}$'
    start_date = None
    end_date = None
    category = None
    formats = ['metadata', 'pdf', 'html', 'source']
    output_dir = None
    verbose = True

    # Single unified call
    data = fetch_arxiv_batch(
        id_regex=id_regex,
        start_date=start_date,
        end_date=end_date,
        category=category,
        formats=formats,
        output_dir=output_dir,
        verbose=verbose
    )

    # Print snippets for each paper
    for aid, entry in data.items():
        print(f"=== {aid} ===")
        if 'metadata' in formats:
            md = entry.get('metadata', {})
            print("Title:", md.get('title'))
            print("Authors:", ', '.join(md.get('authors', [])))
            abs_txt = md.get('abstract', '')
            print("Abstract (first 3 lines):")
            for ln in abs_txt.splitlines()[:3]: print("  ", ln)
        if 'abstract' in formats:
            abs_txt = entry.get('abstract', '')
            print("Abstract (first 3 lines):")
            for ln in abs_txt.splitlines()[:3]: print("  ", ln)
    sys.exit(0)

import arxivloader
import pandas as pd

submittedDate = "[2000010100000+TO+202412310000]"
query = "search_query=submittedDate:{sd}".format(sd=submittedDate)

df = arxivloader.load(query, sortBy="submittedDate", sortOrder="ascending",
                      columns=["id", "title", "summary", "authors", "primary_category", "categories"],
                      page_size=10000, num=50000, timeout=90000)

# df.to_csv(r"C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\data\arxiv_metadata.csv", index=False)

# tmp = pd.read_csv(r"C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\data\arxiv_metadata.csv")
