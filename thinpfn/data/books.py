"""The ten public credit books of the benchmark, rebuilt from their sources. The repository ships no data: `prepare` downloads
each raw file (or reads one you placed in data/raw), builds the numeric feature view exactly as in the benchmark, joins the
shipped split manifest (benchmark/splits/<book>.csv.gz: raw row key -> train / validation / calibration / test) and writes
data/books/<book>.parquet. The test split is sealed: `load(name, "test")` needs sealed_ok=True.

    python -m thinpfn.data.books prepare            # all books (Lending Club is a 1.7 GB download)
    python -m thinpfn.data.books prepare us_bureau  # one book

Licences are those stated at the sources (SOURCES below); check them before any use beyond research. Prosper carries no
licence and is used for evaluation only. uci_full and tw_cards are two views of the same UCI table (tw_cards keeps 4,000
training rows); lending_club and us_p2p both derive from LendingClub (different periods and fields)."""
from __future__ import annotations

import hashlib, sys, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"; BOOKS = ROOT / "data" / "books"; SPLITS = ROOT / "benchmark" / "splits"
SPLIT_NAMES = ("train", "validation", "calibration", "test")
BOOK_NAMES = ["uci_full", "tw_cards", "us_bureau", "in_vehicle", "us_homeequity", "us_p2p", "lending_club", "ppdai", "prosper", "tvs"]
CHARTER = BOOK_NAMES[:6]   # the six books of the original study design; the other four were added for breadth

SOURCES = {
    "uci": dict(files={"default_of_credit_card_clients.csv": "https://raw.githubusercontent.com/maximer-v/qwenderwriter/main/data/default_of_credit_card_clients.csv"},
                origin="UCI 350 Default of Credit Card Clients (Taiwan, 2005), CSV export of the UCI xls as published with Qwenderwriter", licence="CC BY 4.0 (UCI)",
                label="default payment next month", rows=30000),
    "us_bureau": dict(files={"GiveMeSomeCredit.arff": "https://openml.org/data/v1/download/22125240/GiveMeSomeCredit.arff"}, origin="Give Me Some Credit (US consumer bureau, 2011), OpenML 46929 (TabArena curation)",
                      licence="Kaggle competition rules (2011); OpenML copy flagged Public", label="FinancialDistressNextTwoYears = serious delinquency (90+ days) within two years", rows=150000),
    "in_vehicle": dict(files={"LT-Vehicle-Loan-Default-Prediction.arff": "https://openml.org/data/v1/download/22121040/LT-Vehicle-Loan-Default-Prediction.arff"},
                       origin="L&T Vehicle Loan Default (India, 2018), OpenML 46430", licence="Analytics Vidhya hackathon data; OpenML copy flagged Public", label="loan_default = default on the first EMI", rows=233154),
    "us_homeequity": dict(files={"HMEQ_Data.arff": "https://openml.org/data/v1/download/22102162/HMEQ_Data.arff"}, origin="HMEQ home equity loans (US), OpenML 43337",
                          licence="CC0 as declared by the uploader; SAS sample data", label="BAD = defaulted or seriously delinquent", rows=5960),
    "us_p2p": dict(files={"Lending-Club-Loan-Data.arff": "https://openml.org/data/v1/download/22102554/Lending-Club-Loan-Data.arff"}, origin="LendingClub 2007 to 2015 subset, OpenML 43729",
                   licence="CC0 (OpenML)", label="not.fully.paid", rows=9578),
    "lending_club": dict(files={"accepted_2007_to_2018Q4.csv": "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"},
                         origin="LendingClub accepted loans 2007 to 2018Q4 (Hugging Face re-host)", licence="CC0-1.0 as declared by the re-host; LendingClub published no licence",
                         label="Charged Off or Default vs Fully Paid, matured loans only; out-of-time splits by issue date", rows=2260701),
    "ppdai": dict(files={"ppdai.csv": "https://figshare.com/ndownloader/files/26092985"}, origin="PPDai (China P2P, 2016 competition release), figshare 13573856", licence="CC BY 4.0 (figshare)",
                  label="label = overdue / default", rows=55596),
    "prosper": dict(files={"prosperLoanData.csv": "https://s3.amazonaws.com/udacity-hosted-downloads/ud651/prosperLoanData.csv"}, origin="Prosper listings 2005 to 2014 (Udacity course copy)",
                    licence="unknown: EVALUATION ONLY", label="Chargedoff or Defaulted vs Completed, matured loans only; out-of-time splits by origination date", rows=113937),
    "tvs": dict(files={"TVS_Loan_Default.arff": "https://openml.org/data/v1/download/22102568/TVS_Loan_Default.arff"}, origin="TVS Credit cross-sell personal loan default (India), OpenML 43743",
                licence="CC0 as declared by the uploader; hackathon terms unknown", label="V32 = defaulter on the cross-sold loan", rows=119528),
}
# SHA-256 of the source files the benchmark was built from (verified after download; a mismatch means the source changed)
SHA256 = {'GiveMeSomeCredit.arff': '92c897782a049feef5de124c1e5b82a61ed9373aa7e78565f03aea3aa3be21da',
           'HMEQ_Data.arff': '760f6a54019189ed83e8909b4ccf008736b700ff2925589d8475eca4c6965e58',
           'LT-Vehicle-Loan-Default-Prediction.arff': '9e195cc96c9dc397effbaa3a292224c73cc13dfff7233c7d534a9221c1161fc3',
           'Lending-Club-Loan-Data.arff': 'a48a56b1aedb9869f7a1541c8b58d7ab3df7599c5674faa17dffbb088c0a2821',
           'TVS_Loan_Default.arff': '273a82edf8fe58caea82ac0cb7fe48b3a246bd80a5c7e136aaef2cd558412d25',
           'accepted_2007_to_2018Q4.csv': '3eae03c28fd9d2e8a076ebeb73507e8d4d0f44d90500decdb0936e0933d1f36a',
           'default_of_credit_card_clients.csv': 'bf15b02eb5a2c34f01f6d071083ecb7b8f4e803a4280a7ef497903d84c791c48',
           'ppdai.csv': 'cc951f408392023d40758d57d58ab55fe74bfe6b98f77136928e5a9ac0b3cb63',
           'prosperLoanData.csv': '71c1c1931f65bc4d2e3a8f3603d865e074e327b2271590f3a06829197b9cce0f'}
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
UCI_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
LC_NUM = ["loan_amnt", "int_rate", "installment", "annual_inc", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
          "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies"]
LC_CAT = ["term", "grade", "sub_grade", "emp_length", "home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type"]
LC_BAD = {"Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"}; LC_GOOD = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}


# ----------------------------------------------------------------------------------------------------------------------- utilities
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22): h.update(chunk)
    return h.hexdigest()


def fetch(source: str):
    """Download the source files into data/raw (atomically: a partial download never takes the final name) and verify their
    SHA-256 against the benchmark's. A file you placed in data/raw yourself is verified too; a mismatch is a warning, not an
    error, so a re-issued source can still be used (the split join will then fail if rows moved)."""
    RAW.mkdir(parents=True, exist_ok=True)
    for fname, url in SOURCES[source]["files"].items():
        p = RAW / fname
        if not p.exists():
            print(f"downloading {fname} from {url}", flush=True); tmp = p.with_suffix(p.suffix + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": "thinpfn-data-prepare"})
            with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 22): f.write(chunk)
            tmp.replace(p)
        got = sha256_of(p); want = SHA256.get(fname)
        if want and got != want: print(f"warning: {fname} sha256 {got[:12]} differs from the benchmark's {want[:12]}; the source may have changed", flush=True)
    return RAW


def read_arff(path: Path) -> pd.DataFrame:
    """Minimal ARFF reader: attribute names from the header, body via pandas (string attributes allowed)."""
    names, body_at = [], None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            l = line.strip()
            if l.lower().startswith("@attribute"): names.append(l.split(None, 2)[1].strip("'\""))
            elif l.lower().startswith("@data"): body_at = i + 1; break
    f = pd.read_csv(path, skiprows=body_at, header=None, names=names, na_values=["?"], quotechar="'", low_memory=False)
    for c in f.columns:
        if f[c].dtype == object: f[c] = f[c].str.strip("'\"")
    return f


def codes(series: pd.Series) -> pd.Series:
    """Deterministic integer codes for a categorical column (sorted unique values), NaN kept."""
    cats = sorted(v for v in series.dropna().unique()); m = {v: i for i, v in enumerate(cats)}
    return series.map(m).astype("float32")


def opaque(prefix: str, key) -> str:
    return prefix + "_" + hashlib.sha256(f"{prefix}:{key}".encode()).hexdigest()[:16]


def join_splits(book: str, frame: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    """Keep the rows named in the shipped manifest and attach their split; rows absent from the manifest are not part of the book."""
    man = pd.read_csv(SPLITS / f"{book}.csv.gz"); man["key"] = man["key"].astype(str)
    frame = frame.copy(); frame.insert(0, "key", key.astype(str).to_numpy()); out = man.merge(frame, on="key", how="inner", validate="one_to_one")
    if len(out) != len(man): raise RuntimeError(f"{book}: {len(man) - len(out)} manifest rows missing from the raw file (wrong file or version?)")
    return out.drop(columns=["key"])


def numeric_view(f: pd.DataFrame, drop=()) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    for c in f.columns:
        if c in drop: continue
        col = f[c]; out[c] = codes(col) if (col.dtype == object or str(col.dtype).startswith("category")) else pd.to_numeric(col, errors="coerce").astype("float32")
    return out


# ----------------------------------------------------------------------------------------------------------------------- builders
def build_uci(book: str) -> pd.DataFrame:
    raw = fetch("uci"); f = pd.read_csv(raw / "default_of_credit_card_clients.csv", skiprows=1)
    frame = pd.DataFrame({"y": f["default payment next month"].astype("int8")}); frame[UCI_FEATURES] = f[UCI_FEATURES].astype("float32")
    return join_splits(book, frame, pd.Series(np.arange(len(f))))


def build_us_bureau(book="us_bureau") -> pd.DataFrame:
    raw = fetch("us_bureau"); d = read_arff(raw / "GiveMeSomeCredit.arff")
    y = d["FinancialDistressNextTwoYears"].astype(str).str.lower().isin(["yes", "1", "true"]).astype("int8")
    feats = d.drop(columns=["FinancialDistressNextTwoYears"]).apply(pd.to_numeric, errors="coerce").astype("float32")
    frame = pd.concat([y.rename("y"), feats], axis=1); return join_splits(book, frame, pd.Series(np.arange(len(d))))


def build_in_vehicle(book="in_vehicle") -> pd.DataFrame:
    raw = fetch("in_vehicle"); d = read_arff(raw / "LT-Vehicle-Loan-Default-Prediction.arff")
    dob = pd.to_datetime(d["date.of.birth"], format="%d-%m-%y", errors="coerce"); dis = pd.to_datetime(d["disbursaldate"], format="%d-%m-%y", errors="coerce")
    age = (dis - dob).dt.days / 365.25; age = age.where(age > 0, age + 100)   # two-digit years: births before 1970 wrap
    ids_n = d[["aadhar_flag", "pan_flag", "voterid_flag", "driving_flag", "passport_flag"]].astype(float).sum(1)
    def months(s):
        try: a, b = str(s).strip("'\" ").split(" "); return int(a[:-3]) * 12 + int(b[:-3])
        except Exception: return np.nan
    feats = pd.DataFrame({"disbursed_amount": d["disbursed_amount"], "asset_cost": d["asset_cost"], "ltv": d["ltv"], "age": age, "ids_n": ids_n,
                          "salaried": (d["employment.type"].astype(str).str.strip("' ") == "Salaried").astype(float), "cns_score": d["perform_cns.score"], "pri_accts": d["pri.no.of.accts"],
                          "pri_active": d["pri.active.accts"], "pri_overdue": d["pri.overdue.accts"], "pri_balance": d["pri.current.balance"], "pri_sanctioned": d["pri.sanctioned.amount"],
                          "pri_instal": d["primary.instal.amt"], "sec_overdue": d["sec.overdue.accts"], "new_accts_6m": d["new.accts.in.last.six.months"], "delinq_6m": d["delinquent.accts.in.last.six.months"],
                          "acct_age_m": d["average.acct.age"].map(months), "hist_m": d["credit.history.length"].map(months), "inquiries": d["no.of_inquiries"]}).apply(pd.to_numeric, errors="coerce").astype("float32")
    frame = pd.concat([d["loan_default"].astype("int8").rename("y"), feats], axis=1); return join_splits(book, frame, pd.Series(np.arange(len(d))))


def build_us_homeequity(book="us_homeequity") -> pd.DataFrame:
    raw = fetch("us_homeequity"); d = read_arff(raw / "HMEQ_Data.arff")
    feats = d.drop(columns=["BAD"]).copy(); feats["REASON"] = (feats["REASON"] == "HomeImp").astype(float); feats["JOB"] = feats["JOB"].astype("category").cat.codes.astype(float)
    feats = feats.apply(pd.to_numeric, errors="coerce").astype("float32")
    frame = pd.concat([pd.to_numeric(d["BAD"]).astype("int8").rename("y"), feats], axis=1); return join_splits(book, frame, pd.Series(np.arange(len(d))))


def build_us_p2p(book="us_p2p") -> pd.DataFrame:
    raw = fetch("us_p2p"); d = read_arff(raw / "Lending-Club-Loan-Data.arff")
    feats = d.drop(columns=["not.fully.paid"]).copy(); feats["purpose"] = feats["purpose"].astype("category").cat.codes.astype(float); feats = feats.apply(pd.to_numeric, errors="coerce").astype("float32")
    frame = pd.concat([pd.to_numeric(d["not.fully.paid"]).astype("int8").rename("y"), feats], axis=1); return join_splits(book, frame, pd.Series(np.arange(len(d))))


def build_lending_club(book="lending_club") -> pd.DataFrame:
    raw = fetch("lending_club"); f = pd.read_csv(raw / "accepted_2007_to_2018Q4.csv", usecols=["id", "issue_d", "loan_status"] + LC_NUM + LC_CAT, low_memory=False)
    f = f[f["loan_status"].isin(LC_BAD | LC_GOOD)].copy(); y = f["loan_status"].isin(LC_BAD).astype("int8")
    ym = f["issue_d"].astype(str).str.split("-", expand=True); date = pd.to_datetime(ym[1] + "-" + ym[0].map(MONTHS).astype("Int64").astype(str) + "-01", errors="coerce")
    out = pd.DataFrame({"y": y.to_numpy()}, index=f.index)
    for c in LC_NUM: out[c] = pd.to_numeric(f[c], errors="coerce").astype("float32")
    out["term_months"] = f["term"].astype(str).str.extract(r"(\d+)")[0].astype("float32")
    out["emp_length_years"] = f["emp_length"].astype(str).str.replace("< 1", "0").str.replace("10+", "10", regex=False).str.extract(r"(\d+)")[0].astype("float32")
    for c in ["grade", "sub_grade", "home_ownership", "verification_status", "purpose", "addr_state", "initial_list_status", "application_type"]: out[c] = codes(f[c])
    out = out[date.notna()]; out.insert(0, "date", date[date.notna()].to_numpy())
    return join_splits(book, out, f.loc[out.index, "id"].astype(str))


def build_ppdai(book="ppdai") -> pd.DataFrame:
    raw = fetch("ppdai"); f = pd.read_csv(raw / "ppdai.csv")
    frame = pd.concat([f["label"].astype("int8").rename("y"), numeric_view(f.drop(columns=["label"]))], axis=1); return join_splits(book, frame, pd.Series(np.arange(len(f))))


def build_prosper(book="prosper") -> pd.DataFrame:
    raw = fetch("prosper"); f = pd.read_csv(raw / "prosperLoanData.csv", low_memory=False).drop_duplicates(subset=["ListingKey"])
    f = f[f["LoanStatus"].isin(["Chargedoff", "Defaulted", "Completed"])].copy(); y = f["LoanStatus"].isin(["Chargedoff", "Defaulted"]).astype("int8")
    keep = ["Term", "BorrowerRate", "ProsperRating (Alpha)", "ProsperScore", "ListingCategory (numeric)", "BorrowerState", "Occupation", "EmploymentStatus", "EmploymentStatusDuration", "IsBorrowerHomeowner",
            "CreditScoreRangeLower", "CreditScoreRangeUpper", "CurrentCreditLines", "OpenCreditLines", "TotalCreditLinespast7years", "OpenRevolvingAccounts", "OpenRevolvingMonthlyPayment", "InquiriesLast6Months",
            "TotalInquiries", "CurrentDelinquencies", "AmountDelinquent", "DelinquenciesLast7Years", "PublicRecordsLast10Years", "PublicRecordsLast12Months", "RevolvingCreditBalance", "BankcardUtilization",
            "AvailableBankcardCredit", "TotalTrades", "TradesNeverDelinquent (percentage)", "TradesOpenedLast6Months", "DebtToIncomeRatio", "IncomeRange", "IncomeVerifiable", "StatedMonthlyIncome", "LoanOriginalAmount", "MonthlyLoanPayment"]
    feats = f[keep].copy(); feats["IsBorrowerHomeowner"] = feats["IsBorrowerHomeowner"].astype(str); feats["IncomeVerifiable"] = feats["IncomeVerifiable"].astype(str)
    frame = pd.concat([y.rename("y"), numeric_view(feats)], axis=1); frame.insert(0, "date", pd.to_datetime(f["LoanOriginationDate"], errors="coerce").to_numpy())
    frame.columns = [c.replace(" (Alpha)", "_alpha").replace(" (numeric)", "").replace(" (percentage)", "_pct") for c in frame.columns]
    return join_splits(book, frame, f["ListingKey"].astype(str))


def build_tvs(book="tvs") -> pd.DataFrame:
    raw = fetch("tvs"); f = read_arff(raw / "TVS_Loan_Default.arff")
    feats = f.drop(columns=[c for c in ("V32", "V1", "V16") if c in f.columns])   # V1 customer id, V16 date-of-birth string
    frame = pd.concat([pd.to_numeric(f["V32"], errors="coerce").astype("int8").rename("y"), numeric_view(feats)], axis=1); return join_splits(book, frame, pd.Series(np.arange(len(f))))


BUILDERS = {"uci_full": lambda: build_uci("uci_full"), "tw_cards": lambda: build_uci("tw_cards"), "us_bureau": build_us_bureau, "in_vehicle": build_in_vehicle, "us_homeequity": build_us_homeequity,
            "us_p2p": build_us_p2p, "lending_club": build_lending_club, "ppdai": build_ppdai, "prosper": build_prosper, "tvs": build_tvs}


# ----------------------------------------------------------------------------------------------------------------------- public API
def prepare(names=None):
    BOOKS.mkdir(parents=True, exist_ok=True)
    for name in (names or BOOK_NAMES):
        df = BUILDERS[name](); df.to_parquet(BOOKS / f"{name}.parquet", index=False)
        s = df.groupby("split")["y"].agg(["size", "mean"]); print(name, {k: (int(v["size"]), round(float(v["mean"]), 3)) for k, v in s.iterrows()}, flush=True)


def features_of(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("split", "y", "date")]


def load(name: str, split: str, sealed_ok: bool = False):
    """(X float32 [n, d], y int64 [n]) of one split of a prepared book. The test split needs sealed_ok=True."""
    if split == "test" and not sealed_ok: raise PermissionError("the test split is sealed; pass sealed_ok=True only from a final evaluation")
    p = BOOKS / f"{name}.parquet"
    if not p.exists(): raise FileNotFoundError(f"{p} not found: run `python -m thinpfn.data.books prepare {name}`")
    df = pd.read_parquet(p); df = df[df.split == split]; feats = features_of(df)
    return df[feats].to_numpy(np.float32), df["y"].to_numpy(np.int64)


def subsample_context(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A context of n rows drawn from a split, stratified so the prevalence is kept and both classes are present (the benchmark's sampler)."""
    rng = np.random.default_rng(seed)
    if n >= len(y):
        i = rng.choice(len(y), min(n, len(y)), replace=False); return X[i], y[i]
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_pos = int(np.clip(round(n * len(pos) / len(y)), 1, n - 1))
    i = np.concatenate([rng.choice(pos, n_pos, replace=False), rng.choice(neg, n - n_pos, replace=False)]); rng.shuffle(i)
    return X[i], y[i]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prepare": prepare(sys.argv[2:] or None)
    else: print(__doc__)
