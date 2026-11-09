"""Field-type vocabulary of the ledger simulator. The network has a small field-type channel; at inference every real column
is UNKNOWN (the released model does not use dataset-specific tags), so the vocabulary only matters for the synthetic stream."""
from __future__ import annotations

TYPES = [
    "UNKNOWN",
    # loan / contract
    "LOAN_AMOUNT", "TERM", "INTEREST_RATE", "INSTALMENT", "LOAN_PURPOSE", "PRODUCT_TYPE", "LENDER_GRADE", "ASSET_VALUE", "LTV",
    # applicant
    "INCOME", "DTI", "AGE", "EMPLOYMENT_LENGTH", "EMPLOYMENT_STATUS", "HOME_OWNERSHIP", "EDUCATION", "MARITAL_STATUS", "SEX", "DEPENDENTS", "REGION", "OCCUPATION", "VERIFICATION",
    # bureau / credit history
    "BUREAU_SCORE", "CREDIT_LIMIT", "BALANCE", "UTILISATION", "OPEN_ACCOUNTS", "TOTAL_ACCOUNTS", "ACCOUNT_AGE", "INQUIRIES", "DELINQUENCY_COUNT", "MONTHS_SINCE_DELINQUENCY",
    "PUBLIC_RECORDS", "DEROGATORY", "OVERDUE_AMOUNT", "MORTGAGE_ACCOUNTS",
    # behavioural ledger (ordered lags share one type; the position in the slot order carries the lag)
    "DELINQUENCY_STATUS", "BILL_AMOUNT", "PAYMENT_AMOUNT", "PAYMENT_RATIO", "SPEND_COUNT",
    # time
    "APPLICATION_DATE", "FLAG",
]
ID = {t: i for i, t in enumerate(TYPES)}
N_TYPES = len(TYPES)

# simulator column prefix -> type
SIMULATOR_TAGS = {"pay_": "DELINQUENCY_STATUS", "bill_": "BILL_AMOUNT", "payamt_": "PAYMENT_AMOUNT", "limit": "CREDIT_LIMIT", "utilisation": "UTILISATION", "pay_ratio": "PAYMENT_RATIO",
                  "bur_": "BUREAU_SCORE", "app_": "UNKNOWN", "demo_": "UNKNOWN", "age": "AGE", "application_date": "APPLICATION_DATE"}

def simulator_tag_ids(names) -> list[int]:
    """Prefix match of the simulator's column names to field types (UNKNOWN otherwise)."""
    out = []
    for n in names:
        t = "UNKNOWN"
        for prefix, typ in SIMULATOR_TAGS.items():
            if n.startswith(prefix): t = typ; break
        out.append(ID[t])
    return out
