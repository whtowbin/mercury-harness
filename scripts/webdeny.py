"""Web deny patterns every benchmark driver adds to its own list (DESIGN-LOG §131.5).

Mercury 3 found benchmark answers through routes the per-benchmark lists did not name: GitHub's
API (`api.github.com/repos/django/django/pulls/<task>/files`, the upstream fix), code search
(Sourcegraph, grep.app, GitHub search) for a problem's own function names, and web search
engines fetched as pages. The harness arms run with search off, so a search engine reached
through fetch_url is the same lookup by another door. Matching is `agent/tools/web.py`
`denied_by`: lowercased substrings, also checked against the code a run executes, so no pattern
here may be something a solution could plausibly contain.
"""

# CDNs that serve any GitHub repository's files at any ref: Mercury 3 read django's main branch,
# fixes and tests included, from cdn.jsdelivr.net/gh/django/django@main (§131.13). npm packages on
# jsDelivr (/npm/) stay reachable.
GITHUB_MIRRORS = [
    "jsdelivr.net/gh/",
    "githack.com",
    "statically.io/gh/",
    "gitcdn.",
    "ghproxy.",
]

SEARCH_DENY = [
    "github.com/search",  # also api.github.com/search/...
    "sourcegraph.com",
    "grep.app",
    "searchcode.com",
    "google.com/search",
    "bing.com/search",
    "duckduckgo.com",
]
