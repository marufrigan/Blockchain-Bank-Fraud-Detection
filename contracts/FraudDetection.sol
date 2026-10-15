// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract FraudDetection {

    enum Decision { APPROVE, REVIEW, FLAG, REJECT }

    struct TransactionRecord {
        string transaction_id;
        string account_id;
        uint256 fraud_score;
        Decision decision;
        uint256 timestamp;
    }

    address public owner;
    mapping(string => bool) public frozenAccounts;
    mapping(string => TransactionRecord) public ledger;
    string[] public allTransactionIds;

    uint256 public totalApproved;
    uint256 public totalReview;
    uint256 public totalFlagged;
    uint256 public totalRejected;

    event TransactionProcessed(string transaction_id, string account_id, uint256 fraud_score, Decision decision);
    event AccountFrozen(string account_id, uint256 fraud_score);

    constructor() {
        owner = msg.sender;
    }

    function verifyTransaction(
        string memory transaction_id,
        string memory account_id,
        uint256 fraud_score
    ) public returns (uint8) {
        Decision decision;

        if (frozenAccounts[account_id]) {
            decision = Decision.REJECT;
            totalRejected++;
        } else if (fraud_score >= 85) {
            decision = Decision.FLAG;
            frozenAccounts[account_id] = true;
            totalFlagged++;
            emit AccountFrozen(account_id, fraud_score);
        } else if (fraud_score >= 60) {
            decision = Decision.REVIEW;
            totalReview++;
        } else {
            decision = Decision.APPROVE;
            totalApproved++;
        }

        ledger[transaction_id] = TransactionRecord(
            transaction_id, account_id, fraud_score, decision, block.timestamp
        );
        allTransactionIds.push(transaction_id);
        emit TransactionProcessed(transaction_id, account_id, fraud_score, decision);
        return uint8(decision);
    }

    function getStats() public view returns (uint256, uint256, uint256, uint256) {
        return (totalApproved, totalReview, totalFlagged, totalRejected);
    }

    function isAccountFrozen(string memory account_id) public view returns (bool) {
        return frozenAccounts[account_id];
    }

    function getTotalTransactions() public view returns (uint256) {
        return allTransactionIds.length;
    }
}
