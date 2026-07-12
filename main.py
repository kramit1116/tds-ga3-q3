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
}


def extract_vendor(text: str) -> Optional[str]:
    patterns = [
        r"(?:seller|vendor|from|billed by|company)\s*[:#]\s*([A-Za-z0-9&.,\'\-\s]+)",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            line = m.group(1).strip().split("\n")[0]
            if line.strip(" ."):
                return line.strip(" .,")

    # Fallback: guess from the first line, e.g. "NovaSoft Solutions — Tax Invoice"
    first_line = text.strip().split("\n")[0].strip()
    # cut off trailing " — Tax Invoice" / " - Invoice" style suffixes
    first_line = re.split(r"[—\-]\s*(?:tax\s*)?invoice", first_line, flags=re.IGNORECASE)[0].strip()
    words = set(w.lower().strip(".,") for w in first_line.split())
    if first_line and not words.issubset(GENERIC_HEADER_WORDS):
        return first_line
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
    line = _line_containing(text, r"subtotal")
    if not line:
        return None
    nums = _numbers_excluding_percent(line)
    if nums:
        return clean_number(nums[-1])
    return None


def extract_tax(text: str) -> Optional[float]:
    # Try specific tax-type keywords first (most reliable), then fall back
    # to the generic word "tax" -- but skip lines like "Tax ID: 123456789"
    # which aren't a tax amount at all.
    specific_line = _line_containing(text, r"\b(?:gst|cgst|sgst|igst|vat|cess)\b")
    if specific_line:
        nums = _numbers_excluding_percent(specific_line)
        if nums:
            return clean_number(nums[-1])

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
