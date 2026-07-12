"""
Fixed Schema Invoice Extraction API
------------------------------------
POST /extract  {"invoice_text": "..."}  ->  always returns 6 keys:
invoice_no, date, vendor, amount, tax, currency (null if not found)
"""

import re
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser as dateparser

app = FastAPI(title="Invoice Extraction API")

# --- Rule 4: CORS must be enabled so a Cloudflare Worker (a different origin)
# is allowed to call this API from the browser/edge. "*" = allow any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class InvoiceRequest(BaseModel):
    invoice_text: str


def clean_number(raw: str) -> Optional[float]:
    """Turn '1,40,000.00' or '395.82' into a plain float."""
    if not raw:
        return None
    raw = raw.replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def extract_invoice_no(text: str) -> Optional[str]:
    patterns = [
        r"invoice\s*(?:no\.?|#|number)?\s*[:#]\s*([A-Za-z0-9\/\-]+)",
        r"ref(?:erence)?\.?\s*[:#]\s*([A-Za-z0-9\/\-]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_date(text: str) -> Optional[str]:
    patterns = [
        r"(?:date|issued)\s*[:#]?\s*([A-Za-z0-9,\-\/ ]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip().split("\n")[0]
            try:
                dt = dateparser.parse(candidate, fuzzy=True)
                return dt.date().isoformat()
            except (ValueError, OverflowError):
                continue
    return None


GENERIC_HEADER_WORDS = {
    "invoice", "tax", "commercial", "bill", "receipt", "statement", "proforma",
    "original", "duplicate", "copy",
}

# words/patterns that mean "this line is metadata, not a company name"
NON_VENDOR_LINE_PATTERNS = [
    r"\binvoice\s*(?:no\.?|#|number)\b",
    r"\bref(?:erence)?\b",
    r"\bdate\b",
    r"\bissued\b",
    r"\bclient\b",
    r"\bcustomer\b",
    r"\bbill\s*to\b",
    r"\bship\s*to\b",
    r"\b(?:original|duplicate|triplicate)\s*(?:for|copy)\b",
    r"^\s*$",
]


def extract_vendor(text: str) -> Optional[str]:
    patterns = [
        r"(?:seller|vendor|supplier|from|billed by|sold by|issued by|merchant|company(?:\s*name)?|business(?:\s*name)?|provider)\s*[:#]\s*([A-Za-z0-9&.,\'\-\s]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            line = m.group(1).strip().split("\n")[0]
            if line.strip(" ."):
                return line.strip(" .,")

    # Fallback: scan the first few lines for something that looks like a
    # company name (not a generic header, not another metadata field).
    for raw_line in text.strip().split("\n")[:4]:
        line = raw_line.strip()
        if not line:
            continue
        if any(re.search(p, line, re.IGNORECASE) for p in NON_VENDOR_LINE_PATTERNS):
            continue
        # cut off trailing " — Tax Invoice" / " - Invoice" style suffixes
        candidate = re.split(r"[—\-]\s*(?:tax\s*)?invoice", line, flags=re.IGNORECASE)[0].strip()
        words = set(w.lower().strip(".,") for w in candidate.split())
        if candidate and not words.issubset(GENERIC_HEADER_WORDS):
            return candidate
    return None


NUMBER_RE = r"[\d,]+\.?\d*"


def _numbers_excluding_percent(line: str):
    """Return all numeric amounts on a line, skipping any number that is
    actually a percentage (i.e. immediately followed by a % sign)."""
    results = []
    for m in re.finditer(NUMBER_RE, line):
        after = line[m.end():m.end() + 2].strip()
        if after.startswith("%"):
            continue
        results.append(m.group(0))
    return results


def _line_containing(text: str, keyword_pattern: str) -> Optional[str]:
    """Return the first line that contains the keyword AND at least one
    digit (so header lines like 'NovaSoft — Tax Invoice' don't get picked
    over the real 'IGST (18%): Rs. 25,200.00' line)."""
    for line in text.split("\n"):
        if re.search(keyword_pattern, line, re.IGNORECASE) and re.search(r"\d", line):
            return line
    return None


def extract_amount(text: str) -> Optional[float]:
    """Subtotal = amount BEFORE tax (rule 3)."""
    # Try, in priority order, the labels most likely to mean "pre-tax amount".
    # More specific/unambiguous labels come first.
    keyword_patterns = [
        r"\bsub[\s\-]?total\b",
        r"\bnet\s*amount\b",
        r"\btaxable\s*(?:value|amount)\b",
        r"\bamount\s*\(?\s*(?:excl(?:uding)?\.?|before)\s*tax\)?",
        r"\bbase\s*amount\b",
        r"\bnet\s*total\b",
        r"\bprice\s*\(?\s*excl(?:uding)?\.?\s*tax\)?",
    ]
    for kp in keyword_patterns:
        line = _line_containing(text, kp)
        if line:
            nums = _numbers_excluding_percent(line)
            if nums:
                return clean_number(nums[-1])

    # Last-resort fallback: a bare "Amount:" label, as long as it's not
    # part of "Tax Amount" / "Total Amount" (those mean something else).
    for line in text.split("\n"):
        if re.search(r"\bamount\b", line, re.IGNORECASE) and re.search(r"\d", line):
            if re.search(r"\b(?:tax|total|grand|due|paid)\s*amount\b", line, re.IGNORECASE):
                continue
            nums = _numbers_excluding_percent(line)
            if nums:
                return clean_number(nums[-1])
    return None


def extract_tax(text: str) -> Optional[float]:
    # 1) An explicit "Total Tax" / "Total GST" / "GST Total" line is the
    #    most reliable source if the invoice provides one.
    total_line = _line_containing(text, r"\btotal\s*(?:gst|tax)\b|\b(?:gst|tax)\s*total\b")
    if total_line:
        nums = _numbers_excluding_percent(total_line)
        if nums:
            return clean_number(nums[-1])

    # 2) Indian-style invoices often split tax into CGST + SGST (or add
    #    Cess). These are separate line items that together make up the
    #    real tax amount, so we sum every distinct component we find.
    component_patterns = {
        "cgst": r"\bcgst\b",
        "sgst": r"\bsgst\b",
        "igst": r"\bigst\b",
        "cess": r"\bcess\b",
    }
    total = 0.0
    found_components = []
    for line in text.split("\n"):
        if not re.search(r"\d", line):
            continue
        for key, pat in component_patterns.items():
            if key in found_components:
                continue
            if re.search(pat, line, re.IGNORECASE):
                nums = _numbers_excluding_percent(line)
                if nums:
                    total += clean_number(nums[-1])
                    found_components.append(key)
                break
    if len(found_components) >= 2:
        return round(total, 2)
    if len(found_components) == 1:
        return round(total, 2)

    # 3) Generic single "GST:" / "VAT:" style line.
    specific_line = _line_containing(text, r"\b(?:gst|vat)\b")
    if specific_line:
        nums = _numbers_excluding_percent(specific_line)
        if nums:
            return clean_number(nums[-1])

    # 4) Fall back to the bare word "tax" (skip "Tax ID" style lines).
    for line in text.split("\n"):
        if re.search(r"\btax\b", line, re.IGNORECASE) and re.search(r"\d", line):
            if re.search(r"\btax\s*(?:id|no\.?|number)\b", line, re.IGNORECASE):
                continue
            nums = _numbers_excluding_percent(line)
            if nums:
                return clean_number(nums[-1])
    return None


def extract_currency(text: str) -> Optional[str]:
    m = re.search(r"\b(INR|USD|EUR|GBP|AED|SGD)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    if re.search(r"₹|rs\.?", text, re.IGNORECASE):
        return "INR"
    if "$" in text:
        return "USD"
    return None


@app.post("/extract")
def extract(req: InvoiceRequest):
    text = req.invoice_text
    return {
        "invoice_no": extract_invoice_no(text),
        "date": extract_date(text),
        "vendor": extract_vendor(text),
        "amount": extract_amount(text),
        "tax": extract_tax(text),
        "currency": extract_currency(text),
    }


@app.get("/")
def health():
    return {"status": "ok"}
