const { Web3 } = require('./fraud_blockchain/node_modules/web3');
const fs = require('fs');
const path = require('path');

async function main() {
    console.log("=== BLOCKCHAIN FRAUD DETECTION ===");

    const web3 = new Web3('http://127.0.0.1:8545');
    const accounts = await web3.eth.getAccounts();
    console.log("Connected to Ganache. Account:", accounts[0]);

    const contractJSON = JSON.parse(
        fs.readFileSync('./fraud_blockchain/build/contracts/FraudDetection.json', 'utf8')
    );
    const contractAddress = fs.readFileSync('./output/contract_address.txt', 'utf8').trim();
    console.log("Contract address:", contractAddress);

    const contract = new web3.eth.Contract(contractJSON.abi, contractAddress);

    const csvContent = fs.readFileSync('./output/blockchain_input.csv', 'utf8');
    const lines = csvContent.trim().split('\n');
    const headers = lines[0].split(',');
    const txnIdIdx = headers.indexOf('transaction_id');
    const accIdIdx = headers.indexOf('account_id');
    const scoreIdx = headers.indexOf('xgb_fraud_score');
    const truthIdx = headers.indexOf('y_true');

    const decisionMap = { 0: 'APPROVE', 1: 'REVIEW', 2: 'FLAG', 3: 'REJECT' };
    const results = [];

    console.log(`Processing ${lines.length - 1} transactions on blockchain...`);

    for (let i = 1; i < lines.length; i++) {
        const parts = lines[i].split(',');
        const transaction_id = parts[txnIdIdx];
        const account_id = parts[accIdIdx];
        const fraud_score = parseFloat(parts[scoreIdx]);
        const y_true = parseInt(parts[truthIdx]);
        const score_scaled = Math.round(fraud_score * 100);

        try {
            const receipt = await contract.methods
                .verifyTransaction(transaction_id, account_id, score_scaled)
                .send({ from: accounts[0], gas: 300000 });

            const decision_int = await contract.methods
                .ledger(transaction_id)
                .call()
                .then(r => parseInt(r.decision));

            const decision = decisionMap[decision_int] || 'UNKNOWN';

            results.push({
                transaction_id,
                account_id,
                fraud_score: fraud_score.toFixed(4),
                score_scaled,
                y_true,
                decision,
                tx_hash: receipt.transactionHash,
                block_number: receipt.blockNumber
            });

            if (i % 50 === 0) {
                console.log(`Processed ${i}/${lines.length - 1} transactions...`);
            }
        } catch (err) {
            console.error(`Error on transaction ${transaction_id}: ${err.message}`);
            results.push({
                transaction_id, account_id,
                fraud_score: fraud_score.toFixed(4),
                score_scaled, y_true,
                decision: 'ERROR',
                tx_hash: 'N/A',
                block_number: 'N/A'
            });
        }
    }

    // Get final stats from contract
    const stats = await contract.methods.getStats().call();
    const totalTxns = await contract.methods.getTotalTransactions().call();

    console.log("\n=== BLOCKCHAIN RESULTS ===");
    console.log(`Total Transactions: ${totalTxns}`);
    console.log(`Approved:  ${stats[0]}`);
    console.log(`Review:    ${stats[1]}`);
    console.log(`Flagged:   ${stats[2]}`);
    console.log(`Rejected:  ${stats[3]}`);

    // Save ledger CSV
    const csvHeader = 'transaction_id,account_id,fraud_score,score_scaled,y_true,decision,tx_hash,block_number\n';
    const csvRows = results.map(r =>
        `${r.transaction_id},${r.account_id},${r.fraud_score},${r.score_scaled},${r.y_true},${r.decision},${r.tx_hash},${r.block_number}`
    ).join('\n');
    fs.writeFileSync('./output/blockchain_ledger.csv', csvHeader + csvRows);
    console.log("\nSaved: output/blockchain_ledger.csv");

    // Save stats
    const statsContent = `total_transactions,approved,review,flagged,rejected\n${totalTxns},${stats[0]},${stats[1]},${stats[2]},${stats[3]}`;
    fs.writeFileSync('./output/blockchain_stats.csv', statsContent);
    console.log("Saved: output/blockchain_stats.csv");

    console.log("\n=== BLOCKCHAIN PHASE COMPLETE ===");
}

main().catch(console.error);
