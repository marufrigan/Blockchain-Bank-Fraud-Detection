// Benchmark original (V1) and hardened (V2) fraud-decision contracts on a local Ganache chain.
// Usage: node bench.js <input.csv with transaction_id,account_id,score> <out.json>
const fs = require('fs');
const path = require('path');
const solc = require('solc');
const ganache = require('ganache');
const { ethers } = require('ethers');

const [,, inCsv, outJson, thR, thF] = process.argv;
const TH_R = parseInt(thR || '6000'), TH_F = parseInt(thF || '8500');

function compile(file, name) {
  const src = fs.readFileSync(file, 'utf8');
  const input = { language: 'Solidity', sources: { [name + '.sol']: { content: src } },
    settings: { optimizer: { enabled: true, runs: 200 }, outputSelection: { '*': { '*': ['abi', 'evm.bytecode.object'] } } } };
  const out = JSON.parse(solc.compile(JSON.stringify(input)));
  if (out.errors) out.errors.filter(e => e.severity === 'error').forEach(e => { throw new Error(e.formattedMessage); });
  const c = out.contracts[name + '.sol'][name];
  return { abi: c.abi, bytecode: '0x' + c.evm.bytecode.object };
}

function readCsv(f) {
  const lines = fs.readFileSync(f, 'utf8').trim().split('\n');
  const h = lines[0].split(',');
  return lines.slice(1).map(l => { const p = l.split(','); const o = {}; h.forEach((k, i) => o[k] = p[i]); return o; });
}

const now = () => Number(process.hrtime.bigint()) / 1e6;
const stats = a => { const s = [...a].sort((x, y) => x - y); const m = s.reduce((x, y) => x + y, 0) / s.length;
  const sd = Math.sqrt(s.reduce((x, y) => x + (y - m) ** 2, 0) / s.length);
  return { n: s.length, mean: m, sd, p50: s[Math.floor(s.length * 0.5)], p95: s[Math.floor(s.length * 0.95)], min: s[0], max: s[s.length - 1] }; };

(async () => {
  const server = ganache.server({ logging: { quiet: true }, wallet: { totalAccounts: 3 }, chain: { chainId: 1337 }, miner: { blockGasLimit: 30000000 } });
  await new Promise(r => server.listen(8545, r));
  const provider = new ethers.JsonRpcProvider('http://127.0.0.1:8545', undefined, { cacheTimeout: -1 });
  const ownerS = await provider.getSigner(0), oracleS = await provider.getSigner(1), attackerS = await provider.getSigner(2);

  const rows = readCsv(inCsv);
  const salt = ethers.id('demo-salt');
  const res = { n_input: rows.length };

  // ---------- V1 (original repo contract) ----------
  const v1c = compile(path.join(__dirname, '..', '..', 'contracts', 'FraudDetection.sol'), 'FraudDetection');
  let f = new ethers.ContractFactory(v1c.abi, v1c.bytecode, ownerS);
  let t0 = now(); let v1 = await f.deploy(); const v1dep = await v1.deploymentTransaction().wait(); res.v1_deploy_gas = Number(v1dep.gasUsed);
  const v1g = { 0: [], 1: [], 2: [], 3: [] }, v1lat = [], v1dec = [];
  t0 = now();
  for (const r of rows) {
    const s = Math.round(parseFloat(r.score) * 100);
    const a = now();
    const tx = await v1.connect(ownerS).verifyTransaction(r.transaction_id, r.account_id, s, { gasLimit: 500000 });
    const rc = await tx.wait();
    v1lat.push(now() - a);
    const d = Number((await v1.ledger(r.transaction_id)).decision);
    v1g[d].push(Number(rc.gasUsed)); v1dec.push(d);
  }
  res.v1_total_ms = now() - t0;
  res.v1_gas_by_decision = Object.fromEntries(Object.entries(v1g).filter(([, v]) => v.length).map(([k, v]) => [k, stats(v)]));
  res.v1_gas_all = stats(Object.values(v1g).flat());
  res.v1_latency_ms = stats(v1lat);
  res.v1_tps = rows.length / (res.v1_total_ms / 1000);
  res.v1_counts = [0, 1, 2, 3].map(k => v1dec.filter(d => d === k).length);
  res.v1_decisions = v1dec;

  // V1 security probes
  try { const tx = await v1.connect(attackerS).verifyTransaction('ATTACK-1', rows[0].account_id, 99, { gasLimit: 500000 }); await tx.wait();
        res.v1_unauthorised_freeze = await v1.isAccountFrozen(rows[0].account_id); } catch (e) { res.v1_unauthorised_freeze = 'reverted'; }
  try { const tx = await v1.connect(attackerS).verifyTransaction(rows[1].transaction_id, rows[1].account_id, 0, { gasLimit: 500000 }); await tx.wait();
        res.v1_overwrite_accepted = Number((await v1.ledger(rows[1].transaction_id)).fraud_score) === 0; } catch (e) { res.v1_overwrite_accepted = false; }

  // ---------- V2 (hardened) ----------
  const v2c = compile(path.join(__dirname, 'FraudDetectionV2.sol'), 'FraudDetectionV2');
  f = new ethers.ContractFactory(v2c.abi, v2c.bytecode, ownerS);
  const deployV2 = async () => { const c = await f.deploy(TH_R, TH_F, 1); const rc = await c.deploymentTransaction().wait();
    await (await c.setOracle(await oracleS.getAddress(), true)).wait(); return [c, rc]; };
  let [v2, v2dep] = await deployV2(); res.v2_deploy_gas = Number(v2dep.gasUsed);
  const ref = s => ethers.keccak256(ethers.concat([salt, ethers.toUtf8Bytes(s)]));
  const enc = rows.map(r => ({ tx: ref(r.transaction_id), acc: ref(r.account_id), s: Math.min(10000, Math.round(parseFloat(r.score) * 10000)) }));
  const v2g = { 0: [], 1: [], 2: [], 3: [] }, v2lat = [], v2dec = [];
  t0 = now();
  for (const e of enc) {
    const a = now();
    const tx = await v2.connect(oracleS).verifyTransaction(e.tx, e.acc, e.s, { gasLimit: 500000 });
    const rc = await tx.wait();
    v2lat.push(now() - a);
    const d = Number((await v2.ledger(e.tx)).decision);
    v2g[d].push(Number(rc.gasUsed)); v2dec.push(d);
  }
  res.v2_total_ms = now() - t0;
  res.v2_gas_by_decision = Object.fromEntries(Object.entries(v2g).filter(([, v]) => v.length).map(([k, v]) => [k, stats(v)]));
  res.v2_gas_all = stats(Object.values(v2g).flat());
  res.v2_latency_ms = stats(v2lat);
  res.v2_tps = rows.length / (res.v2_total_ms / 1000);
  res.v2_counts = [0, 1, 2, 3].map(k => v2dec.filter(d => d === k).length);
  res.v2_decisions = v2dec;

  // V2 security probes
  const probe = async fn => { try { await (await fn()).wait(); return 'accepted'; } catch (e) { return 'reverted'; } };
  res.v2_unauthorised_call = await probe(() => v2.connect(attackerS).verifyTransaction(ethers.id('ATTACK-1'), enc[0].acc, 9999, { gasLimit: 500000 }));
  res.v2_duplicate_tx = await probe(() => v2.connect(oracleS).verifyTransaction(enc[1].tx, enc[1].acc, 0, { gasLimit: 500000 }));
  res.v2_oracle_unfreeze = await probe(() => v2.connect(oracleS).unfreeze(enc[0].acc, ethers.id('x'), { gasLimit: 500000 }));
  const frozenAcc = enc.find((e, i) => v2dec[i] === 2);
  if (frozenAcc) { const rc = await (await v2.connect(ownerS).unfreeze(frozenAcc.acc, ethers.id('case-001'))).wait(); res.v2_unfreeze_gas = Number(rc.gasUsed); }

  // ---------- V2 batching ----------
  res.batch = [];
  for (const B of [1, 10, 25, 50, 100]) {
    const [c] = await deployV2();
    const n = Math.floor(Math.min(enc.length, 500) / B) * B;
    if (n === 0) continue;
    let gas = 0; const a = now();
    for (let i = 0; i + B <= n; i += B) {
      const sl = enc.slice(i, i + B);
      const rc = await (await c.connect(oracleS).verifyBatch(sl.map(x => x.tx), sl.map(x => x.acc), sl.map(x => x.s), { gasLimit: 29000000 })).wait();
      gas += Number(rc.gasUsed);
    }
    const ms = now() - a;
    res.batch.push({ B, n, gas_per_tx: gas / n, tps: n / (ms / 1000) });
  }

  fs.writeFileSync(outJson, JSON.stringify(res, null, 1));
  console.log(JSON.stringify({ ...res, v1_decisions: undefined, v2_decisions: undefined }, null, 1));
  await server.close();
})().catch(e => { console.error(e); process.exit(1); });
