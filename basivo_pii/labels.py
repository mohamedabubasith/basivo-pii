"""The 18 labels (docs/pii-model-v1-overview.pdf section 3.4), the same list as pii/labels.py."""

LABELS = [
    "PERSON_NAME",
    "DATE",
    "AGE",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "BANK_ACCOUNT",
    "CARD_NUMBER",
    "NATIONAL_ID",
    "TAX_ID",
    "PASSPORT",
    "DRIVING_LICENCE",
    "USERNAME",
    "URL",
    "IP_ADDRESS",
    "CUSTOMER_ID",
    "PASSWORD",
    "API_KEY",
]

# Labels that identify a person on their own. A miss here is a leak.
DIRECT = {
    "PERSON_NAME",
    "EMAIL",
    "PHONE",
    "ADDRESS",
    "BANK_ACCOUNT",
    "CARD_NUMBER",
    "NATIONAL_ID",
    "TAX_ID",
    "PASSPORT",
    "DRIVING_LICENCE",
    "USERNAME",
    "CUSTOMER_ID",
}
