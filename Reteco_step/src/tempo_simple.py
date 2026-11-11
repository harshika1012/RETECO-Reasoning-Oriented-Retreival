"""Cheap, LLM-free temporal signals: year-level regex parsing of queries and documents.
Granularity is the year. Intervals are inclusive (start_year, end_year)."""
import math
import re

MIN_Y, MAX_Y = 1500, 2030
Y = r"(1[5-9]\d\d|20[0-3]\d)"
YEAR_RE = re.compile(rf"(?<![\d.,$]){Y}(?![\d])")
RANGE_RE = re.compile(rf"(?:\b(?:from|between)\s+)?(?<!\d){Y}\s*(?:to|and|until|through|thru|[-\u2013\u2014])\s*{Y}(?!\d)", re.I)
SINCE_RE = re.compile(rf"\b(?:since|starting in|beginning in|from)\s+(?:early |late |mid-?)?{Y}\b", re.I)
BEFORE_RE = re.compile(rf"\b(?:before|prior to|until|up to)\s+{Y}\b", re.I)
AFTER_RE = re.compile(rf"\b(?:after|following)\s+{Y}\b", re.I)
DECADE_RE = re.compile(r"\b(1[5-9]|20)(\d)0'?s\b")


def _clip(a, b):
    return (max(MIN_Y, min(a, b)), min(MAX_Y, max(a, b)))


def parse_query_intervals(text):
    """Returns a list of (start, end) year intervals the query refers to (possibly empty)."""
    ivs, used = [], []
    for m in RANGE_RE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if abs(a - b) <= 200:
            ivs.append(_clip(a, b)); used.append(m.span())
    for m in DECADE_RE.finditer(text):
        s = int(m.group(1) + m.group(2) + "0")
        ivs.append(_clip(s, s + 9)); used.append(m.span())
    for rx, f in ((SINCE_RE, lambda y: (y, MAX_Y)), (BEFORE_RE, lambda y: (MIN_Y, y - 1)), (AFTER_RE, lambda y: (y + 1, MAX_Y))):
        for m in rx.finditer(text):
            if any(a <= m.start(1) < b for a, b in used):
                continue
            ivs.append(f(int(m.group(1)))); used.append(m.span())
    for m in YEAR_RE.finditer(text):
        if any(a <= m.start(1) < b for a, b in used):
            continue
        y = int(m.group(1)); ivs.append((y, y))
    return list(dict.fromkeys(ivs))


def parse_doc_years(text):
    """Set of years a document mentions (ranges/decades expanded, capped)."""
    years = set()
    for m in RANGE_RE.finditer(text):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        if b - a <= 60:
            years.update(range(a, b + 1))
    for m in DECADE_RE.finditer(text):
        s = int(m.group(1) + m.group(2) + "0"); years.update(range(s, s + 10))
    years.update(int(m.group(1)) for m in YEAR_RE.finditer(text))
    return years


def interval_distance(years, iv):
    """Min distance in years between any doc year and the interval; None if the doc has no years."""
    if not years:
        return None
    s, e = iv
    return min(0 if s <= y <= e else min(abs(y - s), abs(y - e)) for y in years)


def temporal_score(years, intervals, mode="graded", sigma=2.0, agg="mean"):
    """mode 'binary': 1 if the doc has a year inside the interval else 0.
    mode 'graded': exp(-d/sigma), d = years outside the interval (0 inside). Docs with no years score 0."""
    if not intervals or not years:
        return 0.0
    vals = []
    for iv in intervals:
        d = interval_distance(years, iv)
        vals.append(1.0 if d == 0 else 0.0) if mode == "binary" else vals.append(math.exp(-d / sigma))
    return max(vals) if agg == "max" else sum(vals) / len(vals)