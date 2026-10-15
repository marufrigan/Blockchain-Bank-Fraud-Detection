# Blockchain Bank Fraud Detection

Machine-learning fraud scoring for digital banking, with decisions written to an Ethereum smart contract that approves, reviews, flags or rejects each transaction and freezes flagged accounts.

The repository contains two pipelines:

| Folder | What it is |
|---|---|
| repository root (`step1` to `step8`, `contracts/`) | The original prototype: data cleaning, EDA, Random Forest / XGBoost / graph-enhanced LightGBM, and the `FraudDetection` contract driven from Ganache. |
| `paper/` | The leakage-controlled pipeline, the hardened `FraudDetectionV2` contract and the benchmark used in the accompanying paper, *Auditable Fraud Decisioning for Digital Banking: A Leakage-Controlled Learning Pipeline with a Hardened Smart-Contract Ledger*. |

## Dataset

Synthetic banking data, January 2022 to December 2024.

| File | Contents |
|---|---|
| `transactions.csv.gz` | 1,000,000 transactions from 49,903 accounts, 17,143 fraudulent (1.71%) across seven fraud patterns. Compressed to stay under GitHub's file size limit; run `gunzip -k transactions.csv.gz` for the original scripts. The `paper/` pipeline reads the `.gz` directly. |
| `account_profiles.csv` | 50,000 account profiles. |
| `network_edges.csv` | 7,411 edges between 6,783 accounts that share a device, e-mail domain, IP address or phone. |
| `fraud_patterns.csv`, `time_series_stats.csv` | Pattern summaries and hourly statistics. |

### Leakage notes

An audit of the data found three sources of label leakage that matter for any model trained on it:

1. **Label-derived account columns.** `fraud_count`, `fraud_amount`, `fraud_rate` and `is_fraudster` are functions of the target.
2. **Full-history aggregates.** `total_transactions`, `avg_amount` and `max_amount` equal the count, mean and maximum over each account's entire three-year history, so they include future transactions.
3. **Label-defined rings.** `ring_id` is present on an edge if and only if both accounts are fraudsters, so `is_ring_member` is a partial copy of the label.

## Leakage-controlled pipeline (`paper/`)

- Point-in-time behavioural features computed only from each account's earlier transactions (prior count, 24-hour and 7-day counts, amount relative to the prior mean, first use of a country or device, prior foreign share).
- Unsupervised graph features (degree, weighted degree, k-core, component size, shared-attribute types) that never read `ring_id` or `both_fraud`.
- Account-disjoint stratified 5-fold cross-validation, with thresholds selected on held-out accounts.
- A risk policy calibrated for FLAG precision of at least 0.90 and REVIEW+FLAG recall of 0.90.

### Results (5-fold, mean ± SD)

| Model (leakage-free features) | PR-AUC | ROC-AUC | F1 | Recall @ 1% FPR |
|---|---|---|---|---|
| Logistic regression | 0.795 ± 0.004 | 0.990 | 0.718 ± 0.003 | 0.801 ± 0.005 |
| Random Forest | 0.820 ± 0.002 | 0.993 | 0.748 ± 0.003 | 0.847 ± 0.002 |
| XGBoost | 0.866 ± 0.003 | 0.995 | 0.787 ± 0.005 | 0.875 ± 0.005 |
| LightGBM | **0.867 ± 0.003** | 0.995 | 0.787 ± 0.005 | **0.878 ± 0.003** |

Adding the label-derived columns inflates XGBoost PR-AUC from 0.866 to 0.976. Full-history aggregates and ring membership change it by only +0.0003 on this dataset.

## Hardened contract (`paper/chain/FraudDetectionV2.sol`)

| | V1 (`contracts/FraudDetection.sol`) | V2 |
|---|---|---|
| Who can submit decisions | anyone | authorised scoring oracles only |
| Resubmitting a transaction ID | overwrites the record | reverted (write-once) |
| Identifiers on chain | plaintext strings | salted keccak256 references |
| Thresholds | hard-coded | owner-governed, linked to a model version |
| Lifting a freeze | not possible | owner `unfreeze` with a hashed case reason |
| Batching | no | `verifyBatch` |
| Mean gas per decision | 139,051 | 83,025 (53,343 at batch size 100) |

Replaying the decision rule shows that a permanent freeze-on-flag rejects 16.9% of legitimate transactions over the three years, against 0.23% with a 72-hour review hold.

## Reproducing the paper results

Requires Python 3.11 or later and Node.js 18 or later.

```bash
cd paper
pip install -r requirements.txt
python3 ml.py features          # point-in-time + graph features
python3 ml.py models            # LR, RF, XGBoost, LightGBM on the leakage-free set
python3 ml.py audit             # leakage audit configurations C0, C1, C3, C4
python3 policy.py LightGBM_C2_ours   # threshold calibration, freeze replay, on-chain input
cd chain && npm install && npm run bench && cd ..
python3 fig_arch.py && python3 figs_results.py
```

On 2 vCPUs the full model and audit runs take about 15 minutes, and the contract benchmark about 7 minutes. Outputs go to `paper/results/` and `paper/figures/`; the committed versions are the ones reported in the paper.

## Original prototype (repository root)

```bash
gunzip -k transactions.csv.gz
python3 step1_install.py
python3 step2_cleaning.py
python3 step3_eda_charts.py
python3 step4_graph_features.py
python3 step5_ml_models_FIXED.py
# start Ganache on 127.0.0.1:8545, deploy contracts/FraudDetection.sol with Truffle,
# and write the deployed address to output/contract_address.txt
# (step6 expects the Truffle project, its node_modules and build/ in a fraud_blockchain/ subfolder)
node step6_blockchain_interact.js
python3 step7_blockchain_charts.py
python3 step8_final_summary.py
```

Known issues in the original prototype, kept as-is because the paper audits it: `step5_ml_models.py` and `step5_ml_models_FIXED.py` are identical, the `is_ring_member` feature leaks the label, and the on-chain sample is balanced at 50% fraud rather than the real 1.71%.
