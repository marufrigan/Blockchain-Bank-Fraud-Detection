// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// Hardened fraud-decision ledger: authorised scoring oracles, pseudonymous
/// identifiers, write-once records, governed thresholds, auditable unfreeze,
/// model-version binding and batched submission.
contract FraudDetectionV2 {
    enum Decision { APPROVE, REVIEW, FLAG, REJECT }

    struct Record {
        bytes32 accountRef;   // keccak256(salt || account_id), no plaintext ID on chain
        uint16  score;        // fraud probability x 10000
        Decision decision;
        uint32  modelVersion;
        uint64  timestamp;
    }

    address public owner;
    mapping(address => bool) public isOracle;
    mapping(bytes32 => Record) public ledger;      // txRef -> record
    mapping(bytes32 => bool)   public frozen;      // accountRef -> frozen
    uint16 public reviewThreshold;                 // x 10000
    uint16 public flagThreshold;                   // x 10000
    uint32 public modelVersion;
    uint256[4] public counts;

    event Decided(bytes32 indexed txRef, bytes32 indexed accountRef, uint16 score, Decision decision, uint32 modelVersion);
    event Frozen(bytes32 indexed accountRef, bytes32 txRef);
    event Unfrozen(bytes32 indexed accountRef, address officer, bytes32 reasonHash);
    event ThresholdsUpdated(uint16 review, uint16 flag, uint32 modelVersion);

    modifier onlyOwner()  { require(msg.sender == owner, "not owner"); _; }
    modifier onlyOracle() { require(isOracle[msg.sender], "not oracle"); _; }

    constructor(uint16 _review, uint16 _flag, uint32 _model) {
        require(_review < _flag && _flag <= 10000, "bad thresholds");
        owner = msg.sender;
        reviewThreshold = _review; flagThreshold = _flag; modelVersion = _model;
    }

    function setOracle(address o, bool allowed) external onlyOwner { isOracle[o] = allowed; }

    function setThresholds(uint16 _review, uint16 _flag, uint32 _model) external onlyOwner {
        require(_review < _flag && _flag <= 10000, "bad thresholds");
        reviewThreshold = _review; flagThreshold = _flag; modelVersion = _model;
        emit ThresholdsUpdated(_review, _flag, _model);
    }

    function _decide(bytes32 txRef, bytes32 accountRef, uint16 score) internal returns (Decision d) {
        require(ledger[txRef].timestamp == 0, "duplicate tx");
        require(score <= 10000, "bad score");
        if (frozen[accountRef])            { d = Decision.REJECT; }
        else if (score >= flagThreshold)   { d = Decision.FLAG; frozen[accountRef] = true; emit Frozen(accountRef, txRef); }
        else if (score >= reviewThreshold) { d = Decision.REVIEW; }
        else                               { d = Decision.APPROVE; }
        ledger[txRef] = Record(accountRef, score, d, modelVersion, uint64(block.timestamp));
        counts[uint8(d)] += 1;
        emit Decided(txRef, accountRef, score, d, modelVersion);
    }

    function verifyTransaction(bytes32 txRef, bytes32 accountRef, uint16 score) external onlyOracle returns (Decision) {
        return _decide(txRef, accountRef, score);
    }

    function verifyBatch(bytes32[] calldata txRefs, bytes32[] calldata accountRefs, uint16[] calldata scores) external onlyOracle {
        require(txRefs.length == accountRefs.length && txRefs.length == scores.length, "length");
        for (uint256 i = 0; i < txRefs.length; i++) { _decide(txRefs[i], accountRefs[i], scores[i]); }
    }

    function unfreeze(bytes32 accountRef, bytes32 reasonHash) external onlyOwner {
        require(frozen[accountRef], "not frozen");
        frozen[accountRef] = false;
        emit Unfrozen(accountRef, msg.sender, reasonHash);
    }

    function getStats() external view returns (uint256, uint256, uint256, uint256) {
        return (counts[0], counts[1], counts[2], counts[3]);
    }
}
