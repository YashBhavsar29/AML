def extract_to_csv():
    import pandas as pd

    # ============================================================
    # FILES
    # ============================================================

    PATTERN_FILE = "patterns.txt"
    TRANSACTION_FILE = "new_transactions.csv"
    OUTPUT_FILE = "matched_pattern_transactions.csv"


    # ============================================================
    # 1. READ patterns.txt
    # ============================================================

    with open(PATTERN_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()


    # Extract unique (FROM ACCOUNT, TO ACCOUNT) pairs
    pattern_pairs = set()

    for line in lines:

        line = line.strip()

        # Ignore pattern headers / footers
        if not line or line.startswith("BEGIN") or line.startswith("END"):
            continue

        parts = [x.strip() for x in line.split(",")]

        if len(parts) >= 11:

            # CORRECT MAPPING:
            # 0 = timestamp
            # 1 = from_bank
            # 2 = from_account
            # 3 = to_bank
            # 4 = to_account
            # 5 = amount
            # ...

            from_account = parts[2]
            to_account = parts[4]

            pattern_pairs.add(
                (from_account, to_account)
            )


    print(f"Unique account pairs in patterns.txt: {len(pattern_pairs)}")


    # ============================================================
    # 2. READ new_transactions.csv
    # ============================================================

    df = pd.read_csv(TRANSACTION_FILE)

    df["from_account"] = (
        df["from_account"]
        .astype(str)
        .str.strip()
    )

    df["to_account"] = (
        df["to_account"]
        .astype(str)
        .str.strip()
    )


    # ============================================================
    # 3. FIND ALL MATCHING ACCOUNT PAIRS
    # ============================================================

    transaction_pairs = list(
        zip(
            df["from_account"],
            df["to_account"]
        )
    )

    matched = df[
        pd.Series(transaction_pairs, index=df.index)
        .isin(pattern_pairs)
    ].copy()


    # ============================================================
    # 4. SAVE COMPLETE TRANSACTIONS
    # ============================================================

    matched.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ============================================================
    # 5. SUMMARY
    # ============================================================

    unique_matched_pairs = (
        matched[
            ["from_account", "to_account"]
        ]
        .drop_duplicates()
    )

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    print(
        f"Total transactions in new_transactions.csv : {len(df):,}"
    )

    print(
        f"Unique account pairs in patterns.txt        : "
        f"{len(pattern_pairs):,}"
    )

    print(
        f"Unique account pairs found in dataset       : "
        f"{len(unique_matched_pairs):,}"
    )

    print(
        f"Total matching transactions                 : "
        f"{len(matched):,}"
    )

    print(f"\nSaved to: {OUTPUT_FILE}")

    print(matched.head(20))


import pandas as pd
import re

# ============================================================
# FILES
# ============================================================

PATTERN_FILE = "patterns.txt"
TRANSACTION_FILE = "new_transactions.csv"
OUTPUT_FILE = "new_patterns.txt"


# ============================================================
# 1. READ NEW TRANSACTIONS
# ============================================================

df = pd.read_csv(TRANSACTION_FILE)

# Normalize account IDs
df["from_account"] = df["from_account"].astype(str).str.strip()
df["to_account"] = df["to_account"].astype(str).str.strip()

# Normalize timestamp
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# Normalize amount
df["amount_paid"] = pd.to_numeric(
    df["amount_paid"],
    errors="coerce"
)

print(f"Loaded {len(df):,} transactions")


# ============================================================
# 2. CREATE LOOKUP BY ACCOUNT PAIR
# ============================================================
#
# Key:
#     (from_account, to_account)
#
# Value:
#     ALL transactions having that account pair
#
# ============================================================

pair_lookup = {}

for pair, group in df.groupby(
    ["from_account", "to_account"],
    sort=False
):
    pair_lookup[pair] = group


print(
    f"Unique account pairs in new_transactions.csv: "
    f"{len(pair_lookup):,}"
)


# ============================================================
# 3. READ patterns.txt
# ============================================================

with open(PATTERN_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()


# ============================================================
# 4. PARSE PATTERN BLOCKS
# ============================================================

blocks = []

current_header = None
current_transactions = []

for line in lines:

    line = line.rstrip("\n")

    if line.startswith("BEGIN LAUNDERING ATTEMPT"):

        current_header = line
        current_transactions = []

    elif line.startswith("END LAUNDERING ATTEMPT"):

        if current_header is not None:

            blocks.append({
                "begin": current_header,
                "transactions": current_transactions,
                "end": line
            })

        current_header = None
        current_transactions = []

    elif current_header is not None:

        if line.strip():
            current_transactions.append(line)


print(f"Patterns found: {len(blocks)}")


# ============================================================
# 5. FUNCTION TO PARSE PATTERN TRANSACTION
# ============================================================

def parse_pattern_transaction(line):

    parts = [x.strip() for x in line.split(",")]

    if len(parts) < 11:
        return None

    # CORRECT FORMAT:
    #
    # 0 = timestamp
    # 1 = FROM BANK
    # 2 = FROM ACCOUNT
    # 3 = TO BANK
    # 4 = TO ACCOUNT
    # 5 = amount
    # 6 = currency
    # 7 = received amount
    # 8 = received currency
    # 9 = payment mode
    # 10 = laundering flag

    return {
        "timestamp": parts[0],
        "from_bank": parts[1],
        "from_account": parts[2],
        "to_bank": parts[3],
        "to_account": parts[4],
        "amount": parts[5],
        "currency": parts[6],
        "received_amount": parts[7],
        "received_currency": parts[8],
        "payment_mode": parts[9],
        "is_laundering": parts[10]
    }


# ============================================================
# 6. FORMAT TRANSACTION FROM new_transactions.csv
# ============================================================

def format_new_transaction(row):
    """
    Convert a transaction from new_transactions.csv
    into the new_patterns.txt format.

    We preserve the pattern-file style but use the
    bank/district information from new_transactions.csv.
    """

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = row["timestamp"]

    if pd.notna(timestamp):
        timestamp = timestamp.strftime("%Y/%m/%d %H:%M")
    else:
        timestamp = ""

    # --------------------------------------------------------
    # Banks
    # --------------------------------------------------------

    from_bank = str(
        row["from_indian_bank"]
    ).strip()

    to_bank = str(
        row["to_indian_bank"]
    ).strip()

    # --------------------------------------------------------
    # Accounts
    # --------------------------------------------------------

    from_account = str(
        row["from_account"]
    ).strip()

    to_account = str(
        row["to_account"]
    ).strip()

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    amount = row["amount_paid"]

    if pd.notna(amount):
        amount = f"{float(amount):.2f}"
    else:
        amount = ""

    # --------------------------------------------------------
    # Payment mode
    # --------------------------------------------------------

    payment_mode = str(
        row["payment_mode"]
    ).strip()

    # --------------------------------------------------------
    # Laundering flag
    # --------------------------------------------------------

    laundering = str(
        row["is_laundering"]
    ).strip()

    # --------------------------------------------------------
    # Districts
    #
    # These are NEW fields that do not exist in the
    # original patterns.txt.
    # --------------------------------------------------------

    from_district = str(
        row["from_district"]
    ).strip()

    to_district = str(
        row["to_district"]
    ).strip()

    # --------------------------------------------------------
    # Output line
    #
    # Format:
    #
    # timestamp,
    # from_bank,
    # from_account,
    # to_bank,
    # to_account,
    # amount,
    # payment_mode,
    # is_laundering,
    # from_district,
    # to_district
    #
    # --------------------------------------------------------

    return (
        f"{timestamp},"
        f"{from_bank},"
        f"{from_account},"
        f"{to_bank},"
        f"{to_account},"
        f"{amount},"
        f"{payment_mode},"
        f"{laundering},"
        f"{from_district},"
        f"{to_district}"
    )


# ============================================================
# 7. GENERATE new_patterns.txt
# ============================================================

output_lines = []

total_pattern_transactions = 0
matched_pattern_transactions = 0
unmatched_pattern_transactions = 0


for block_number, block in enumerate(blocks, start=1):

    # Preserve BEGIN line
    output_lines.append(block["begin"])

    for line in block["transactions"]:

        parsed = parse_pattern_transaction(line)

        if parsed is None:
            continue

        total_pattern_transactions += 1

        pair = (
            parsed["from_account"],
            parsed["to_account"]
        )

        # ----------------------------------------------------
        # Find matching account pair
        # ----------------------------------------------------

        matches = pair_lookup.get(pair)

        if matches is None or len(matches) == 0:

            unmatched_pattern_transactions += 1

            print(
                f"WARNING: No match found for "
                f"{parsed['from_account']} -> "
                f"{parsed['to_account']}"
            )

            continue

        # ----------------------------------------------------
        # Add ALL matching transactions
        # ----------------------------------------------------

        for _, row in matches.iterrows():

            output_lines.append(
                format_new_transaction(row)
            )

            matched_pattern_transactions += 1

    # Preserve END line
    output_lines.append(block["end"])

    # Blank line between patterns
    output_lines.append("")


# ============================================================
# 8. WRITE OUTPUT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(output_lines)
    )


# ============================================================
# 9. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NEW PATTERN FILE GENERATED")
print("=" * 70)

print(
    f"Patterns                         : {len(blocks):,}"
)

print(
    f"Pattern transactions              : "
    f"{total_pattern_transactions:,}"
)

print(
    f"Matched transactions written      : "
    f"{matched_pattern_transactions:,}"
)

print(
    f"Unmatched pattern transactions    : "
    f"{unmatched_pattern_transactions:,}"
)

print(
    f"\nOutput file: {OUTPUT_FILE}"
)


# ============================================================
# 10. SHOW FIRST PART OF FILE
# ============================================================

print("\n" + "=" * 70)
print("PREVIEW")
print("=" * 70)

with open(
    OUTPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for i, line in enumerate(f):

        print(line.rstrip())

        if i >= 20:
            break