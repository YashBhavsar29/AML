#!/usr/bin/env python3
"""
Generate IBM AMLSim input parameter files for the SIH26184 prototype.

This script does NOT replace IBM AMLSim's transaction_graph_generator.py.
It creates the input files that IBM's generator expects, then you run the
official graph generator separately.

Default scenario:
    Telangana districts as the AMLSim "country" field
    Common Indian public/private sector banks
    Random but heavy-tailed transaction degree distribution
    AML typology and normal-model configuration
    AMLSim conf.json

Example:
    python generate_sih_dataset.py --accounts 50000 --simulation-name SIH_TELANGANA

Then:
    python scripts/transaction_graph_generator.py \
        paramFiles/SIH_TELANGANA/conf.json
"""

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path


TELANGANA_DISTRICTS = [
    "Cyberabad Commissionerate",
    "Hyderabad Commissionerate",
    "Karimnagar Commissionerate",
    "Khammam Commissionerate",
    "Nizamabad Commissionerate",
    "Rachakonda Commissionerate",
    "Ramagundam Commissionerate",
    "Siddipet Commissionerate",
    "Warangal Commissionerate",
    "Adilabad",
    "Bhadradri Kothagudem",
    "Jagityal",
    "Jayashankar Bhupalpalli",
    "Jogulamba Gadwal",
    "Kamareddy",
    "Kumaram Bheem Asifabad",
    "Mahabubabad",
    "Mahabubnagar",
    "Medak",
    "Nagarkurnool",
    "Nalgonda",
    "Nirmal",
    "Rajanna Siricilla",
    "Sangareddy",
    "Suryapet",
    "Vikarabad",
    "Wanaparthy",
    "Railway Police Secunderabad",
    "Mulug",
    "Narayanpet",
]

# Population-based weights supplied for this SIH scenario.
# Commissionerates use the population of the corresponding area in the
# supplied population table. Area and density are deliberately unused.
DISTRICT_WEIGHTS = {
    "Cyberabad Commissionerate": 2446265,
    "Hyderabad Commissionerate": 3943323,
    "Karimnagar Commissionerate": 1005711,
    "Khammam Commissionerate": 1401639,
    "Nizamabad Commissionerate": 1571022,
    "Rachakonda Commissionerate": 2446265,
    "Ramagundam Commissionerate": 795332,
    "Siddipet Commissionerate": 1012065,
    "Warangal Commissionerate": 1789395,
    "Adilabad": 708972,
    "Bhadradri Kothagudem": 1069261,
    "Jagityal": 985417,
    "Jayashankar Bhupalpalli": 416763,
    "Jogulamba Gadwal": 609990,
    "Kamareddy": 972625,
    "Kumaram Bheem Asifabad": 515812,
    "Mahabubabad": 774549,
    "Mahabubnagar": 919903,
    "Medak": 767428,
    "Nagarkurnool": 893308,
    "Nalgonda": 1618416,
    "Nirmal": 709418,
    "Rajanna Siricilla": 552037,
    "Sangareddy": 1527628,
    "Suryapet": 1099560,
    "Vikarabad": 927140,
    "Wanaparthy": 577758,
    "Railway Police Secunderabad": 3943323,
    "Mulug": 257744,
    "Narayanpet": 566874,
}

BANKS = [
    ("STATE BANK OF INDIA", 1215),
    ("UNION BANK OF INDIA", 683),
    ("HDFC BANK", 465),
    ("ICICI BANK", 398),
    ("CANARA BANK", 396),
    ("AXIS BANK", 199),
    ("BANK OF BARODA", 178),
    ("INDIAN BANK", 167),
    ("BANDHAN BANK", 159),
    ("PUNJAB NATIONAL BANK", 147),
    ("KOTAK MAHINDRA BANK", 123),
    ("INDIAN OVERSEAS BANK", 116),
    ("INDUSIND BANK", 107),
    ("CENTRAL BANK OF INDIA", 101),
    ("AU SMALL FIN.BANK", 92),
    ("BANK OF INDIA", 89),
    ("KARUR VYSYA BANK", 65),
    ("IDBI BANK", 55),
    ("IDFC FIRST BANK", 53),
    ("CITY UNION BANK", 47),
    ("SOUTH INDIAN BANK", 44),
    ("YES BANK", 43),
    ("FEDERAL BANK", 38),
    ("DCB BANK", 37),
    ("DBS BANK INDIA (E-LVB)", 35),
    ("CSB BANK LIMITED", 32),
    ("RBL BANK", 32),
    ("KARNATAKA BANK", 30),
    ("EQUITAS SMALL FIN. BANK", 28),
    ("INDIA POST PAYMENTS BANK", 23),
    ("PUNJAB AND SIND BANK", 17),
    ("KBS LOCAL AREA BANK", 14),
    ("TAMILNAD MERCANTILE BANK", 12),
]
# Approximate activity weights. Hyderabad/Rangareddy/Medchal receive more
# synthetic accounts, while every district remains represented.


def weighted_choice(rng, items, weights):
    return rng.choices(items, weights=weights, k=1)[0]


def allocate_counts(total, names, weights):
    """Allocate exactly total records according to relative weights."""
    raw = [total * w / float(sum(weights)) for w in weights]
    counts = [int(math.floor(x)) for x in raw]
    remainder = total - sum(counts)

    order = sorted(
        range(len(names)),
        key=lambda i: raw[i] - counts[i],
        reverse=True,
    )
    for i in order[:remainder]:
        counts[i] += 1

    return dict(zip(names, counts))


def generate_degree_sequence(n, rng, max_degree=50):
    """
    Create a heavy-tailed in/out degree sequence.

    The mixture intentionally contains:
      - many low-degree retail/customer accounts
      - a substantial medium-degree group
      - a small high-degree hub population

    It is a synthetic heuristic, not a measurement of an actual bank network.
    AMLSim requires total in-degree == total out-degree.
    """
    in_degrees = []
    out_degrees = []

    for _ in range(n):
        bucket = rng.random()

        if bucket < 0.72:
            # Most ordinary accounts.
            in_d = rng.randint(0, 5)
            out_d = rng.randint(0, 5)
        elif bucket < 0.95:
            # More connected accounts / businesses.
            in_d = rng.randint(4, 15)
            out_d = rng.randint(4, 15)
        else:
            # Small hub population.
            in_d = min(max_degree, max(10, int(rng.lognormvariate(2.6, 0.65))))
            out_d = min(max_degree, max(10, int(rng.lognormvariate(2.6, 0.65))))

        in_degrees.append(in_d)
        out_degrees.append(out_d)

    # Balance the two degree totals while preserving the overall shape.
    diff = sum(in_degrees) - sum(out_degrees)

    if diff > 0:
        # Need more outgoing stubs.
        candidates = [i for i, d in enumerate(out_degrees) if d < max_degree]
        while diff > 0:
            i = candidates[rng.randrange(len(candidates))]
            if out_degrees[i] < max_degree:
                out_degrees[i] += 1
                diff -= 1
    elif diff < 0:
        # Need more incoming stubs.
        diff = -diff
        candidates = [i for i, d in enumerate(in_degrees) if d < max_degree]
        while diff > 0:
            i = candidates[rng.randrange(len(candidates))]
            if in_degrees[i] < max_degree:
                in_degrees[i] += 1
                diff -= 1

    assert sum(in_degrees) == sum(out_degrees)

    return in_degrees, out_degrees


def write_accounts(path, n, rng, min_balance, max_balance):
    districts = TELANGANA_DISTRICTS
    district_weights = [DISTRICT_WEIGHTS.get(d, 1) for d in districts]

    district_counts = allocate_counts(n, districts, district_weights)
    bank_names = [b[0] for b in BANKS]
    bank_weights = [b[1] for b in BANKS]

    # Aggregated AMLSim account parameter file:
    # count,min_balance,max_balance,country,business_type,bank_id
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count",
            "min_balance",
            "max_balance",
            "country",
            "business_type",
            "bank_id",
        ])

        # Keep one row per district/bank combination. AMLSim expands these
        # rows into individual accounts.
        for district in districts:
            count = district_counts[district]
            if count == 0:
                continue

            # Split the district's population across banks.
            bank_counts = allocate_counts(
                count,
                bank_names,
                bank_weights,
            )

            for bank_id in bank_names:
                bank_count = bank_counts[bank_id]
                if bank_count == 0:
                    continue

                writer.writerow([
                    bank_count,
                    min_balance,
                    max_balance,
                    district,       # Deliberately stored in AMLSim's "country".
                    "I",            # Individual/customer-style account.
                    bank_id,
                ])


def write_degree(path, n, rng, max_degree):
    in_degrees, out_degrees = generate_degree_sequence(
        n, rng, max_degree=max_degree
    )

    pairs = Counter(zip(in_degrees, out_degrees))

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Count", "In-degree", "Out-degree"])

        for (in_d, out_d), count in sorted(pairs.items()):
            writer.writerow([count, in_d, out_d])

    return in_degrees, out_degrees


def write_transaction_types(path):
    # AMLSim accepts arbitrary transaction type labels.
    # TRANSFER is retained as the core graph transaction type.
    rows = [
        ("UPI_TRANSFER", 65),
        ("NEFT_TRANSFER", 8),
        ("IMPS_TRANSFER", 8),
        ("POS_PAYMENT", 7),
        ("ATM_CASH_WITHDRAWAL", 12),
    ]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Type", "Frequency"])
        writer.writerows(rows)


def write_alert_patterns(path, accounts, rng):
    # Counts are intentionally modest relative to the population.
    # They can be increased later after validating graph density/runtime.
    suspicious = max(20, int(accounts * 0.002))

    rows = [
        [max(5, suspicious // 4), "fan_in", 2, 5, 12, 1000.0, 50000.0, 2, 24, "", True],
        [max(5, suspicious // 4), "fan_out", 2, 5, 12, 1000.0, 50000.0, 2, 24, "", True],
        [max(3, suspicious // 6), "cycle", 2, 4, 10, 2000.0, 75000.0, 3, 30, "", True],
        [max(3, suspicious // 6), "scatter_gather", 2, 6, 15, 5000.0, 100000.0, 3, 36, "", True],
        [max(2, suspicious // 8), "gather_scatter", 2, 6, 15, 5000.0, 100000.0, 3, 36, "", True],
    ]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count",
            "type",
            "schedule_id",
            "min_accounts",
            "max_accounts",
            "min_amount",
            "max_amount",
            "min_period",
            "max_period",
            "bank_id",
            "is_sar",
        ])
        writer.writerows(rows)


def write_normal_models(path, accounts):
    # Enough models to make the synthetic population heterogeneous.
    base = max(50, int(accounts * 0.01))

    rows = [
        [base * 2, "single", 2, 1, 1, 5, 30, ""],
        [base, "fan_out", 2, 5, 20, 5, 30, ""],
        [base, "fan_in", 2, 5, 20, 5, 30, ""],
        [base, "forward", 2, 3, 3, 5, 30, ""],
        [base, "mutual", 2, 2, 2, 5, 30, ""],
        [base, "periodical", 2, 2, 2, 10, 60, ""],
    ]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "count",
            "type",
            "schedule_id",
            "min_accounts",
            "max_accounts",
            "min_period",
            "max_period",
            "bank_id",
        ])
        writer.writerows(rows)


def write_schema(source_schema, destination):
    if source_schema.exists():
        shutil.copy2(str(source_schema), str(destination))
        return

    # Fallback schema for an AMLSim checkout where paramFiles/1K/schema.json
    # is absent. Normally the existing IBM schema should be copied.
    schema = {
        "account": [],
        "transaction": [],
        "alert_tx": [],
        "alert_member": [],
        "party_individual": [],
        "party_organization": [],
        "account_mapping": [],
        "resolved_entities": [],
    }
    destination.write_text(json.dumps(schema, indent=2))


def write_conf(path, scenario_dir, simulation_name, total_steps, seed,
               output_dir, transaction_interval):
    conf = {
        "general": {
            "random_seed": seed,
            "simulation_name": simulation_name,
            "total_steps": total_steps,
            "base_date": "2024-01-01",
        },
        "default": {
            "min_amount": 100,
            "max_amount": 10000,
            "min_balance": 50000,
            "max_balance": 500000,
            "start_step": -1,
            "end_step": -1,
            "start_range": -1,
            "end_range": -1,
            "transaction_model": 1,
            "margin_ratio": 0.1,
            "bank_id": "",
            "cash_in": {
                "normal_interval": 100,
                "fraud_interval": 50,
                "normal_min_amount": 500,
                "normal_max_amount": 5000,
                "fraud_min_amount": 5000,
                "fraud_max_amount": 50000,
            },
            "cash_out": {
                "normal_interval": 10,
                "fraud_interval": 100,
                "normal_min_amount": 200,
                "normal_max_amount": 10000,
                "fraud_min_amount": 5000,
                "fraud_max_amount": 50000,
            },
        },
        "input": {
            "directory": str(scenario_dir),
            "schema": "schema.json",
            "accounts": "accounts.csv",
            "alert_patterns": "alertPatterns.csv",
            "normal_models": "normalModels.csv",
            "degree": "degree.csv",
            "transaction_type": "transactionType.csv",
            "is_aggregated_accounts": True,
        },
        "temporal": {
            "directory": "tmp",
            "transactions": "transactions.csv",
            "accounts": "accounts.csv",
            "alert_members": "alert_members.csv",
            "normal_models": "normal_models.csv",
        },
        "output": {
            "directory": str(output_dir),
            "accounts": "accounts.csv",
            "transactions": "transactions.csv",
            "cash_transactions": "cash_tx.csv",
            "alert_members": "alert_accounts.csv",
            "alert_transactions": "alert_transactions.csv",
            "sar_accounts": "sar_accounts.csv",
            "party_individuals": "individuals-bulkload.csv",
            "party_organizations": "organizations-bulkload.csv",
            "account_mapping": "accountMapping.csv",
            "resolved_entities": "resolvedentities.csv",
            "transaction_log": "tx_log.csv",
            "counter_log": "tx_count.csv",
            "diameter_log": "diameter.csv",
        },
        "graph_generator": {
            "degree_threshold": 10,
            "high_risk_countries": "",
            "high_risk_business": "",
        },
        "simulator": {
            "compute_diameter": False,
            "transaction_limit": 0,
            "transaction_interval": transaction_interval,
            "numBranches": 1000,
        },
        "visualizer": {
            "degree": "deg.png",
            "wcc": "wcc.png",
            "alert": "alert.png",
            "count": "count.png",
            "clustering": "cc.png",
            "diameter": "diameter.png",
        },
    }

    with path.open("w") as f:
        json.dump(conf, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SIH26184 AMLSim input parameter files."
    )

    parser.add_argument("--accounts", type=int, default=50000,
                        help="Number of synthetic accounts.")
    parser.add_argument("--simulation-name", default="SIH_TELANGANA",
                        help="AMLSim simulation name.")
    parser.add_argument("--seed", type=int, default=99,
                        help="Random seed.")
    parser.add_argument("--min-balance", type=float, default=50000,
                        help="Minimum initial balance.")
    parser.add_argument("--max-balance", type=float, default=500000,
                        help="Maximum initial balance.")
    parser.add_argument("--max-degree", type=int, default=50,
                        help="Maximum synthetic in/out degree.")
    parser.add_argument("--steps", type=int, default=720,
                        help="AMLSim simulation steps.")
    parser.add_argument("--transaction-interval", type=int, default=7,
                        help="AMLSim transaction interval.")
    parser.add_argument("--param-root", default="paramFiles",
                        help="Root directory for AMLSim parameter scenarios.")
    parser.add_argument("--output-root", default="outputs/new/synthetic",
                        help="Root directory for final generated outputs.")
    parser.add_argument("--tmp-root", default="tmp/new/synthetic",
                        help="Root directory for temporal graph-generator files.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing scenario directory.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.accounts < 100:
        raise SystemExit("Use at least 100 accounts for a meaningful graph.")

    root = Path.cwd()

    scenario_dir = root / args.param_root / args.simulation_name
    output_root = root / args.output_root
    tmp_root = root / args.tmp_root

    if scenario_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                "Scenario already exists: {}\n"
                "Use --overwrite to replace it.".format(scenario_dir)
            )
        shutil.rmtree(str(scenario_dir))

    scenario_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    write_accounts(
        scenario_dir / "accounts.csv",
        args.accounts,
        rng,
        args.min_balance,
        args.max_balance,
    )

    in_degrees, out_degrees = write_degree(
        scenario_dir / "degree.csv",
        args.accounts,
        rng,
        args.max_degree,
    )

    write_transaction_types(scenario_dir / "transactionType.csv")
    write_alert_patterns(scenario_dir / "alertPatterns.csv", args.accounts, rng)
    write_normal_models(scenario_dir / "normalModels.csv", args.accounts)

    source_schema = root / "paramFiles" / "1K" / "schema.json"
    write_schema(source_schema, scenario_dir / "schema.json")

    # IMPORTANT:
    # IBM's graph generator uses temporal.directory + simulation_name for
    # its generated temporal files. We therefore keep a clean scenario-level
    # location and tell conf.json where it is.
    #
    # output.directory is similarly configured as a separate synthetic root.
    write_conf(
        scenario_dir / "conf.json",
        scenario_dir,
        args.simulation_name,
        args.steps,
        args.seed,
        output_root,
        args.transaction_interval,
    )

    total_in = sum(in_degrees)
    total_out = sum(out_degrees)

    print()
    print("SIH AMLSim scenario generated successfully.")
    print("Scenario directory :", scenario_dir)
    print("Accounts           :", args.accounts)
    print("Districts          :", len(TELANGANA_DISTRICTS))
    print("Banks              :", len(BANKS))
    print("Total in-degree    :", total_in)
    print("Total out-degree   :", total_out)
    print("Degree balanced    :", total_in == total_out)
    print("Seed               :", args.seed)
    print()
    print("Input files:")
    for name in [
        "accounts.csv",
        "degree.csv",
        "alertPatterns.csv",
        "normalModels.csv",
        "transactionType.csv",
        "schema.json",
        "conf.json",
    ]:
        print("  -", scenario_dir / name)
    print()
    print("Next command:")
    print(
        "python scripts/transaction_graph_generator.py "
        "{}".format(scenario_dir / "conf.json")
    )
    print()


if __name__ == "__main__":
    main()
