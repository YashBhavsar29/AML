import pandas as pd
import re
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

PATTERN_FILE = "patterns.txt"
TRANSACTION_FILE = "new_transactions.csv"

OUTPUT_FILE = "pattern_match_results.csv"


# ============================================================
# COLUMN MAPPING
# ============================================================
#
# patterns.txt:
#
# timestamp,from_account,from_bank,to_account,to_bank,
# amount,currency,received_amount,received_currency,
# payment_mode,is_laundering
#
# new_transactions.csv:
#
# timestamp,from_account,to_account,amount_paid,payment_mode,
# is_laundering,from_district,to_district,
# from_indian_bank,to_indian_bank
#
# We deliberately IGNORE:
#   bank
#   district
#   currency
#
# ============================================================


# ============================================================
# READ NEW TRANSACTIONS
# ============================================================

print("Reading transaction dataset...")

df = pd.read_csv(TRANSACTION_FILE)

required_columns = [
    "timestamp",
    "from_account",
    "to_account",
    "amount_paid",
    "payment_mode",
    "is_laundering"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns in {TRANSACTION_FILE}: {missing}"
    )


df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

df["amount_paid"] = pd.to_numeric(
    df["amount_paid"],
    errors="coerce"
)

df["from_account"] = df["from_account"].astype(str).str.strip()
df["to_account"] = df["to_account"].astype(str).str.strip()
df["payment_mode"] = df["payment_mode"].astype(str).str.strip()

df = df.dropna(
    subset=[
        "timestamp",
        "from_account",
        "to_account",
        "amount_paid"
    ]
).copy()

# Only laundering transactions are relevant
df_laundering = df[
    df["is_laundering"].astype(str).str.lower().isin(
        ["1", "true", "yes"]
    )
].copy()

# Sort chronologically
df_laundering = df_laundering.sort_values("timestamp").reset_index(drop=True)

print(f"Total transactions       : {len(df)}")
print(f"Laundering transactions  : {len(df_laundering)}")


# ============================================================
# PARSE patterns.txt
# ============================================================

print("\nReading patterns.txt...")

with open(PATTERN_FILE, "r", encoding="utf-8") as f:
    text = f.read()


# ------------------------------------------------------------
# Find every BEGIN ... END block
# ------------------------------------------------------------

block_pattern = re.compile(
    r"BEGIN LAUNDERING ATTEMPT\s*-\s*(.*?)\s*\n"
    r"(.*?)"
    r"END LAUNDERING ATTEMPT\s*-\s*.*?",
    re.IGNORECASE | re.DOTALL
)

blocks = block_pattern.findall(text)

print(f"Patterns found: {len(blocks)}")


# ============================================================
# PARSE A TRANSACTION LINE
# ============================================================

def parse_transaction(line):
    """
    Parse one transaction from patterns.txt.

    Format:

    timestamp,
    from_bank,
    from_account,
    to_bank,
    to_account,
    amount,
    currency,
    received_amount,
    received_currency,
    payment_mode,
    is_laundering
    """

    parts = [x.strip() for x in line.split(",")]

    if len(parts) < 11:
        return None

    return {
        "timestamp": pd.to_datetime(parts[0], errors="coerce"),
        "from_bank": parts[1],
        "from_account": parts[2],
        "to_bank": parts[3],
        "to_account": parts[4],
        "amount": float(parts[5]),
        "currency": parts[6],
        "received_amount": float(parts[7]),
        "received_currency": parts[8],
        "payment_mode": parts[9],
        "is_laundering": parts[10]
    }


# ============================================================
# BUILD PATTERN OBJECTS
# ============================================================

patterns = []

for pattern_number, (pattern_name, block) in enumerate(blocks, start=1):

    transactions = []

    for line in block.strip().splitlines():

        line = line.strip()

        if not line:
            continue

        tx = parse_transaction(line)

        if tx is not None:
            transactions.append(tx)

    if not transactions:
        continue

    patterns.append({
        "pattern_id": pattern_number,
        "pattern_name": pattern_name.strip(),
        "transactions": transactions
    })


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_amount(x):
    """
    Exact numeric comparison.

    100.0 == 100.00
    """

    return round(float(x), 2)


def same_transaction(pattern_tx, dataset_row):
    """
    Exact transaction comparison.

    We intentionally DO NOT compare:
        bank
        district
        currency

    because those fields were modified in the generated dataset.
    """

    return (
        pattern_tx["from_account"]
        == dataset_row["from_account"]

        and

        pattern_tx["to_account"]
        == dataset_row["to_account"]

        and

        pattern_tx["timestamp"]
        == dataset_row["timestamp"]

        and

        normalize_amount(pattern_tx["amount"])
        ==
        normalize_amount(dataset_row["amount_paid"])

        and

        pattern_tx["payment_mode"].lower()
        ==
        str(dataset_row["payment_mode"]).lower()
    )


# ============================================================
# EXACT TRANSACTION MATCH
# ============================================================

def exact_transaction_exists(pattern_tx):
    """
    Check whether one exact transaction exists.
    """

    matches = df_laundering[
        (df_laundering["from_account"] == pattern_tx["from_account"])
        &
        (df_laundering["to_account"] == pattern_tx["to_account"])
        &
        (df_laundering["timestamp"] == pattern_tx["timestamp"])
        &
        (
            df_laundering["amount_paid"].round(2)
            ==
            round(pattern_tx["amount"], 2)
        )
        &
        (
            df_laundering["payment_mode"].str.lower()
            ==
            pattern_tx["payment_mode"].lower()
        )
    ]

    return len(matches) > 0


# ============================================================
# EXACT SEQUENCE MATCH
# ============================================================

def match_sequential_pattern(pattern_transactions):
    """
    Used for RANDOM and STANK.

    Every transaction must exist exactly,
    and the timestamps must occur in the same order.
    """

    # Sort pattern chronologically
    pattern_transactions = sorted(
        pattern_transactions,
        key=lambda x: x["timestamp"]
    )

    previous_time = None

    matched_rows = []

    for tx in pattern_transactions:

        matches = df_laundering[
            (df_laundering["from_account"] == tx["from_account"])
            &
            (df_laundering["to_account"] == tx["to_account"])
            &
            (df_laundering["timestamp"] == tx["timestamp"])
            &
            (
                df_laundering["amount_paid"].round(2)
                ==
                round(tx["amount"], 2)
            )
            &
            (
                df_laundering["payment_mode"].str.lower()
                ==
                tx["payment_mode"].lower()
            )
        ]

        if len(matches) == 0:
            return False, []

        row = matches.iloc[0]

        if previous_time is not None:
            if row["timestamp"] < previous_time:
                return False, []

        previous_time = row["timestamp"]

        matched_rows.append(row)

    return True, matched_rows


# ============================================================
# FAN-IN
# ============================================================

def match_fan_in(pattern_transactions):
    """
    FAN-IN:

        A ──┐
        B ──┼──> X
        C ──┘

    Every source -> destination transaction must exist exactly.
    """

    # Determine destination
    destinations = [
        tx["to_account"]
        for tx in pattern_transactions
    ]

    destination_counts = defaultdict(int)

    for d in destinations:
        destination_counts[d] += 1

    # Most common destination is the gathering node
    central = max(
        destination_counts,
        key=destination_counts.get
    )

    matched_rows = []

    for tx in pattern_transactions:

        if tx["to_account"] != central:
            continue

        matches = df_laundering[
            (df_laundering["from_account"] == tx["from_account"])
            &
            (df_laundering["to_account"] == tx["to_account"])
            &
            (df_laundering["timestamp"] == tx["timestamp"])
            &
            (
                df_laundering["amount_paid"].round(2)
                ==
                round(tx["amount"], 2)
            )
            &
            (
                df_laundering["payment_mode"].str.lower()
                ==
                tx["payment_mode"].lower()
            )
        ]

        if len(matches) == 0:
            return False, []

        matched_rows.append(matches.iloc[0])

    return True, matched_rows


# ============================================================
# FAN-OUT
# ============================================================

def match_fan_out(pattern_transactions):
    """
    FAN-OUT:

             ┌──> A
        X ───┼──> B
             └──> C

    Every X -> destination transaction must exist exactly.
    """

    sources = [
        tx["from_account"]
        for tx in pattern_transactions
    ]

    source_counts = defaultdict(int)

    for s in sources:
        source_counts[s] += 1

    central = max(
        source_counts,
        key=source_counts.get
    )

    matched_rows = []

    for tx in pattern_transactions:

        if tx["from_account"] != central:
            continue

        matches = df_laundering[
            (df_laundering["from_account"] == tx["from_account"])
            &
            (df_laundering["to_account"] == tx["to_account"])
            &
            (df_laundering["timestamp"] == tx["timestamp"])
            &
            (
                df_laundering["amount_paid"].round(2)
                ==
                round(tx["amount"], 2)
            )
            &
            (
                df_laundering["payment_mode"].str.lower()
                ==
                tx["payment_mode"].lower()
            )
        ]

        if len(matches) == 0:
            return False, []

        matched_rows.append(matches.iloc[0])

    return True, matched_rows


# ============================================================
# GATHER-SCATTER
# ============================================================

def match_gather_scatter(pattern_transactions):
    """
    GATHER-SCATTER:

    A ──┐
    B ──┼──> X ──> C
    C ──┘     ├──> D
              └──> E

    We require every transaction in the pattern
    to exist exactly.
    """

    matched_rows = []

    for tx in pattern_transactions:

        if not exact_transaction_exists(tx):
            return False, []

        matches = df_laundering[
            (df_laundering["from_account"] == tx["from_account"])
            &
            (df_laundering["to_account"] == tx["to_account"])
            &
            (df_laundering["timestamp"] == tx["timestamp"])
            &
            (
                df_laundering["amount_paid"].round(2)
                ==
                round(tx["amount"], 2)
            )
            &
            (
                df_laundering["payment_mode"].str.lower()
                ==
                tx["payment_mode"].lower()
            )
        ]

        matched_rows.append(matches.iloc[0])

    return True, matched_rows


# ============================================================
# SCATTER-GATHER
# ============================================================

def match_scatter_gather(pattern_transactions):
    """
    SCATTER-GATHER:

          ┌──> A ──┐
    X ────┼──> B ──┼──> Y
          ├──> C ──┤
          └──> D ──┘

    Every transaction must exist exactly.
    """

    matched_rows = []

    for tx in pattern_transactions:

        if not exact_transaction_exists(tx):
            return False, []

        matches = df_laundering[
            (df_laundering["from_account"] == tx["from_account"])
            &
            (df_laundering["to_account"] == tx["to_account"])
            &
            (df_laundering["timestamp"] == tx["timestamp"])
            &
            (
                df_laundering["amount_paid"].round(2)
                ==
                round(tx["amount"], 2)
            )
            &
            (
                df_laundering["payment_mode"].str.lower()
                ==
                tx["payment_mode"].lower()
            )
        ]

        matched_rows.append(matches.iloc[0])

    return True, matched_rows


# ============================================================
# GENERIC EXACT PATTERN MATCH
# ============================================================

def match_generic(pattern_transactions):

    """
    For unknown pattern types.

    Requires every transaction in the pattern
    to exist exactly.
    """

    matched_rows = []

    for tx in pattern_transactions:

        if not exact_transaction_exists(tx):
            return False, []

        matches = df_laundering[
            (df_laundering["from_account"] == tx["from_account"])
            &
            (df_laundering["to_account"] == tx["to_account"])
            &
            (df_laundering["timestamp"] == tx["timestamp"])
            &
            (
                df_laundering["amount_paid"].round(2)
                ==
                round(tx["amount"], 2)
            )
            &
            (
                df_laundering["payment_mode"].str.lower()
                ==
                tx["payment_mode"].lower()
            )
        ]

        matched_rows.append(matches.iloc[0])

    return True, matched_rows


# ============================================================
# DETERMINE PATTERN TYPE
# ============================================================

def get_pattern_type(name):

    name = name.upper()

    if "FAN-IN" in name:
        return "FAN-IN"

    if "FAN-OUT" in name:
        return "FAN-OUT"

    if "SCATTER-GATHER" in name:
        return "SCATTER-GATHER"

    if "GATHER-SCATTER" in name:
        return "GATHER-SCATTER"

    if "RANDOM" in name:
        return "RANDOM"

    if "STANK" in name:
        return "STANK"

    return "UNKNOWN"


# ============================================================
# RUN MATCHING
# ============================================================

results = []

print("\n" + "=" * 70)
print("PATTERN MATCHING")
print("=" * 70)

for pattern in patterns:

    pattern_id = pattern["pattern_id"]
    pattern_name = pattern["pattern_name"]
    transactions = pattern["transactions"]

    pattern_type = get_pattern_type(pattern_name)

    print(
        f"\nPattern {pattern_id}: "
        f"{pattern_name}"
    )

    if pattern_type == "FAN-IN":

        found, matched = match_fan_in(transactions)

    elif pattern_type == "FAN-OUT":

        found, matched = match_fan_out(transactions)

    elif pattern_type == "SCATTER-GATHER":

        found, matched = match_scatter_gather(transactions)

    elif pattern_type == "GATHER-SCATTER":

        found, matched = match_gather_scatter(transactions)

    elif pattern_type in ["RANDOM", "STANK"]:

        found, matched = match_sequential_pattern(transactions)

    else:

        found, matched = match_generic(transactions)

    print(f"Type       : {pattern_type}")
    print(f"Transactions: {len(transactions)}")
    print(f"Exact Match: {found}")

    results.append({
        "pattern_id": pattern_id,
        "pattern_name": pattern_name,
        "pattern_type": pattern_type,
        "transaction_count": len(transactions),
        "exact_match": found
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))

print(f"\nSaved to: {OUTPUT_FILE}")