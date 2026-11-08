// Sidecar backend and PostInvocation compaction hook for the Context Usage panel.
//
// Serves the panel and a /api/report endpoint that runs context_report.py (the
// same parser as the /context skill) on the open conversation's database. The
// model's context window and display name come from the app's live model
// catalog (GetAvailableModels), like the agy CLI.

import { execFile } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import https from 'node:https';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPT = join(HERE, 'context_report.py');
const GEMINI_DIR = process.env.ANTIGRAVITY_GEMINI_DIR || join(homedir(), '.gemini');
const APP_DIR = join(GEMINI_DIR, 'antigravity');
const CONVERSATIONS = join(APP_DIR, 'conversations');
const DATA_DIR = process.env.ANTIGRAVITY_EXECUTABLE_DATA_DIR || join(HERE, '.data');
const PLUGIN_STATE_DIR = join(APP_DIR, 'desktop-context');
const CONFIG_PATH = join(PLUGIN_STATE_DIR, 'config.json');
const ALERT_STATE_PATH = join(PLUGIN_STATE_DIR, 'alert-state.json');
const CATALOG_PATH = join(DATA_DIR, 'model-catalog.json');
const CATALOG_TTL_MS = 10 * 60 * 1000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const VALID_AGENT_ALERTS = new Set(['off', 'critical', 'tiered']);
const TIER_RANK = { normal: 0, warn85: 1, crit95: 2, exceeded100: 3 };

// --- protobuf wire reader (for fast zero-spawn hook checks) -------------------

function readVarint(buf, pos) {
  let val = 0n;
  let shift = 0n;
  while (pos < buf.length) {
    const b = buf[pos++];
    val |= BigInt(b & 0x7f) << shift;
    if ((b & 0x80) === 0) return [val, pos];
    shift += 7n;
    if (shift > 70n) throw new Error('Varint overflow');
  }
  throw new Error('Truncated varint');
}

export class WireMsg {
  constructor(buf) {
    this.fields = new Map();
    if (!buf || !buf.length) return;
    let pos = 0;
    const len = buf.length;
    while (pos < len) {
      let tag;
      try {
        [tag, pos] = readVarint(buf, pos);
      } catch {
        break;
      }
      const fn = Number(tag >> 3n);
      const wt = Number(tag & 7n);
      let val;
      try {
        if (wt === 0) {
          [val, pos] = readVarint(buf, pos);
        } else if (wt === 2) {
          let l;
          [l, pos] = readVarint(buf, pos);
          const n = Number(l);
          if (n < 0 || pos + n > len) break;
          val = buf.subarray(pos, pos + n);
          pos += n;
        } else if (wt === 1) {
          if (pos + 8 > len) break;
          val = buf.subarray(pos, pos + 8);
          pos += 8;
        } else if (wt === 5) {
          if (pos + 4 > len) break;
          val = buf.subarray(pos, pos + 4);
          pos += 4;
        } else {
          break;
        }
      } catch {
        break;
      }
      let arr = this.fields.get(fn);
      if (!arr) {
        arr = [];
        this.fields.set(fn, arr);
      }
      arr.push(val);
    }
  }

  has(fn) {
    return this.fields.has(fn);
  }

  int(fn, signed32 = false) {
    const arr = this.fields.get(fn);
    if (!arr || !arr.length) return 0;
    let v = arr[arr.length - 1];
    if (typeof v !== 'bigint') return 0;
    if (signed32 && v >= 1n << 63n) v -= 1n << 64n;
    return Number(v);
  }

  bytes(fn) {
    const arr = this.fields.get(fn);
    if (!arr || !arr.length) return Buffer.alloc(0);
    const v = arr[arr.length - 1];
    return Buffer.isBuffer(v) || v instanceof Uint8Array ? Buffer.from(v.buffer, v.byteOffset, v.byteLength) : Buffer.alloc(0);
  }

  str(fn) {
    return this.bytes(fn).toString('utf8');
  }

  msg(fn) {
    return this.has(fn) ? new WireMsg(this.bytes(fn)) : null;
  }

  msgs(fn) {
    const arr = this.fields.get(fn);
    if (!arr || !arr.length) return [];
    return arr
      .filter((v) => Buffer.isBuffer(v) || v instanceof Uint8Array)
      .map((v) => new WireMsg(Buffer.from(v.buffer, v.byteOffset, v.byteLength)));
  }
}

// --- hook config & state persistence -----------------------------------------

export function readHookConfig() {
  const defaults = { agentAlert: 'tiered' };
  try {
    if (!existsSync(CONFIG_PATH)) return defaults;
    const parsed = JSON.parse(readFileSync(CONFIG_PATH, 'utf8'));
    const agentAlert = VALID_AGENT_ALERTS.has(parsed?.agentAlert) ? parsed.agentAlert : defaults.agentAlert;
    return { agentAlert };
  } catch {
    return defaults;
  }
}

export function writeHookConfig(patch) {
  const current = readHookConfig();
  const next = {
    agentAlert: VALID_AGENT_ALERTS.has(patch?.agentAlert) ? patch.agentAlert : current.agentAlert,
  };
  mkdirSync(PLUGIN_STATE_DIR, { recursive: true });
  writeFileSync(CONFIG_PATH, JSON.stringify(next, null, 2));
  return next;
}

function readAlertStateMap() {
  try {
    if (!existsSync(ALERT_STATE_PATH)) return {};
    const parsed = JSON.parse(readFileSync(ALERT_STATE_PATH, 'utf8'));
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function writeAlertStateMap(map) {
  try {
    mkdirSync(PLUGIN_STATE_DIR, { recursive: true });
    const entries = Object.entries(map || {})
      .sort((a, b) => (b[1]?.updatedAt || 0) - (a[1]?.updatedAt || 0))
      .slice(0, 100);
    writeFileSync(ALERT_STATE_PATH, JSON.stringify(Object.fromEntries(entries)));
  } catch {
    // Ignore transient state write errors in hooks.
  }
}

function fmtTok(n) {
  if (n >= 1e6) {
    const v = n / 1e6;
    return Number.isInteger(v) ? `${v}M` : `${v.toFixed(1)}M`;
  }
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n || 0);
}

const NATIVE_WINDOWS = [
  ['claude-opus-4-6', 250_000],
  ['claude-sonnet-4-6', 250_000],
  ['claude-', 1_000_000],
  ['gpt-oss-', 131_072],
  ['gemini-', 1_048_576],
];

function estimatePostTurnStepTokens(pb, md) {
  const ui = pb.msg(19);
  const pr = pb.msg(20);
  let bytesLen = 0;
  if (ui) {
    bytesLen = ui.bytes(2).length;
  } else if (pr) {
    bytesLen = pr.bytes(1).length;
  } else {
    const outTok = md.int(23, true);
    if (outTok > 0) return outTok;
    const tc = md.msg(4);
    bytesLen = tc ? tc.bytes(3).length : 0;
    const generic = pb.msg(140);
    const res = generic ? generic.msg(2) : null;
    if (res) bytesLen += res.bytes(1).length;
  }
  if (bytesLen <= 0) return 0;
  return Math.max(1, Math.trunc(bytesLen / 4));
}

function checkpointSummaryTokens(pb, md) {
  const retries = md.msgs(28);
  for (let i = retries.length - 1; i >= 0; i--) {
    const u = retries[i].msg(2);
    if (!u) continue;
    const resp = u.int(10);
    if (resp > 0) return resp;
    const out = u.int(3);
    const think = u.int(9);
    return think < out ? out - think : 0;
  }
  const ck = pb.msg(30);
  if (!ck) return 0;
  return Math.trunc((ck.bytes(4).length + ck.bytes(5).length + ck.bytes(6).length) / 4);
}

export function inspectConversationBudget(conversationId, dbPathOverride = '') {
  const dbPath = dbPathOverride || (UUID.test(conversationId || '') ? join(CONVERSATIONS, `${conversationId}.db`) : '');
  if (!dbPath || !existsSync(dbPath)) return null;

  let db;
  try {
    db = new DatabaseSync(dbPath, { readOnly: true });
    const rows = db.prepare('SELECT idx, step_type, status, metadata, step_payload FROM steps ORDER BY idx').all();
    const genRow = db.prepare('SELECT data FROM gen_metadata ORDER BY idx DESC LIMIT 1').get();
    const gen = genRow?.data ? new WireMsg(genRow.data) : null;

    const chat = gen ? gen.msg(1) : null;
    const chatStart = chat ? chat.msg(9) : null;
    const cwm = chatStart ? chatStart.msg(10) : null;
    const slug = gen?.msg(3)?.str(28) || '';
    let limitTokens = 0;
    for (const [prefix, tok] of NATIVE_WINDOWS) {
      if (slug.startsWith(prefix)) {
        limitTokens = tok;
        break;
      }
    }
    const reqBudget = (cwm ? cwm.int(4, true) : 0) || (limitTokens > 256_000 ? 256_000 : limitTokens);
    if (!limitTokens) limitTokens = reqBudget;
    if (!reqBudget || (limitTokens && reqBudget >= limitTokens)) {
      return { requestBudget: 0, limitTokens, budgetUsedTokens: 0, checkpointCount: 0, lastCheckpoint: null };
    }

    let ckptPos = -1;
    let checkpointCount = 0;
    let lastCheckpoint = null;
    const decoded = rows.map((r, pos) => {
      const pb = new WireMsg(r.step_payload);
      const type = pb.has(1) ? pb.int(1) : Number(r.step_type || 0);
      const status = pb.has(4) ? pb.int(4) : Number(r.status || 0);
      const md = pb.msg(5) || new WireMsg(r.metadata);
      const ck = pb.msg(30);
      if (ck && status === 3) {
        ckptPos = pos;
        checkpointCount += 1;
        const u = md.msg(9);
        const preTokens = u ? u.int(2) + u.int(5) + u.int(6) : 0;
        lastCheckpoint = {
          pos,
          stepIndex: Number(r.idx ?? pos),
          index: ck.int(1),
          start: ck.int(11),
          end: ck.int(12),
          preTokens,
          summaryTokens: checkpointSummaryTokens(pb, md),
        };
      }
      return { pos, idx: Number(r.idx ?? pos), type, status, pb, md, ck };
    });

    let posLast = -1;
    let prefixTok = 0;
    for (let i = decoded.length - 1; i > ckptPos; i--) {
      const s = decoded[i];
      if (!s.pb.has(20) && s.type !== 15) continue;
      const u = s.md.msg(9);
      if (!u) continue;
      const inp = u.int(2);
      const outTok = u.int(3);
      const cacheR = u.int(5);
      const cacheW = u.int(6);
      if (inp + cacheR + cacheW > 0) {
        posLast = i;
        prefixTok = inp + cacheR + cacheW + outTok;
        break;
      }
    }

    let budgetUsedTokens = 0;
    if (posLast >= 0) {
      let deltaTok = 0;
      for (let i = posLast + 1; i < decoded.length; i++) {
        const s = decoded[i];
        if (s.ck) continue;
        const u = s.md.msg(9);
        deltaTok += u && u.int(3) > 0 ? u.int(3) : estimatePostTurnStepTokens(s.pb, s.md);
      }
      budgetUsedTokens = prefixTok + deltaTok;
    } else {
      // Compacted or awaiting first generation after checkpoint: estimate from system breakdown + post-checkpoint steps.
      let sysTok = 0;
      const breakdown = cwm ? cwm.msg(3) : null;
      if (breakdown) {
        for (const g of breakdown.msgs(1)) {
          const gtype = g.int(2);
          if (gtype !== 7) sysTok += g.int(4, true);
        }
      }
      let postCkTok = lastCheckpoint ? lastCheckpoint.summaryTokens : 0;
      for (let i = ckptPos + 1; i < decoded.length; i++) {
        const s = decoded[i];
        if (s.ck) continue;
        postCkTok += estimatePostTurnStepTokens(s.pb, s.md);
      }
      budgetUsedTokens = sysTok + postCkTok;
    }

    return {
      requestBudget: reqBudget,
      limitTokens,
      budgetUsedTokens,
      budgetPct: reqBudget > 0 ? Number(((budgetUsedTokens / reqBudget) * 100).toFixed(1)) : 0,
      checkpointCount,
      lastCheckpoint,
    };
  } catch {
    return null;
  } finally {
    try { db?.close(); } catch { /* ignore */ }
  }
}

export function evaluateLifecycleHook(eventType, payload, options = {}) {
  if (eventType !== 'PostInvocation') return {};
  const config = options.config || readHookConfig();
  if (config.agentAlert === 'off') return {};

  const convId = payload?.conversationId || '';
  if (!UUID.test(convId) && !options.dbPath) return {};

  const info = inspectConversationBudget(convId, options.dbPath || '');
  if (!info || !info.requestBudget) return {};

  const stateMap = options.stateMap || readAlertStateMap();
  const stateKey = convId || 'default';
  const initialNumSteps = Number(payload?.initialNumSteps);
  const ckptJustHappenedInTurn =
    info.checkpointCount > 0 &&
    info.lastCheckpoint &&
    Number.isFinite(initialNumSteps) &&
    info.lastCheckpoint.stepIndex >= initialNumSteps;

  let prev = stateMap[stateKey];
  if (!prev) {
    prev = {
      checkpointCount: ckptJustHappenedInTurn ? info.checkpointCount - 1 : info.checkpointCount,
      tier: 'normal',
      updatedAt: Date.now(),
    };
  }

  const pct = (info.budgetUsedTokens / info.requestBudget) * 100;
  let currentTier = 'normal';
  if (pct >= 100) {
    currentTier = 'exceeded100';
  } else if (pct >= 95) {
    currentTier = 'crit95';
  } else if (pct >= 85 && config.agentAlert === 'tiered') {
    currentTier = 'warn85';
  }

  // 1. Check if a Compaction (Checkpoint) completed since the last hook check
  if (info.checkpointCount > (prev.checkpointCount ?? 0)) {
    stateMap[stateKey] = {
      checkpointCount: info.checkpointCount,
      tier: currentTier,
      updatedAt: Date.now(),
    };
    if (!options.stateMap) writeAlertStateMap(stateMap);

    const ck = info.lastCheckpoint;
    const rangeStr = ck ? ` (Step ${ck.start}–${ck.end})` : '';
    const dropStr = ck?.preTokens
      ? `，有效上下文已从 ${fmtTok(ck.preTokens)} 压缩至约 ${fmtTok(info.budgetUsedTokens)}`
      : '';
    return {
      injectSteps: [
        {
          ephemeralMessage:
            `[Context Usage] 已完成第 ${info.checkpointCount} 次自动上下文压缩${rangeStr}${dropStr}。` +
            `当前请求预算水位：${fmtTok(info.budgetUsedTokens)} / ${fmtTok(info.requestBudget)} (${pct.toFixed(1)}%)。`,
        },
      ],
    };
  }

  if (pct < 80 && prev.tier !== 'normal') {
    prev.tier = 'normal';
    stateMap[stateKey] = { ...prev, checkpointCount: info.checkpointCount, updatedAt: Date.now() };
    if (!options.stateMap) writeAlertStateMap(stateMap);
  }

  // 2. Check if budget usage crossed into a higher warning tier
  if (TIER_RANK[currentTier] > TIER_RANK[prev.tier || 'normal']) {
    stateMap[stateKey] = {
      checkpointCount: info.checkpointCount,
      tier: currentTier,
      updatedAt: Date.now(),
    };
    if (!options.stateMap) writeAlertStateMap(stateMap);

    const remaining = Math.max(0, info.requestBudget - info.budgetUsedTokens);
    if (currentTier === 'exceeded100') {
      return {
        injectSteps: [
          {
            ephemeralMessage:
              `[Context Usage] 🚨 请求预算已达压缩阈值：${fmtTok(info.budgetUsedTokens)} / ${fmtTok(info.requestBudget)} (${pct.toFixed(1)}%)。` +
              '系统将在进入下一回合或下一步工具调用前（PreInvocation）立即触发自动 Compaction。若有关键中间结论请确保已落盘或向用户交代。',
          },
        ],
      };
    }
    if (currentTier === 'crit95') {
      return {
        injectSteps: [
          {
            ephemeralMessage:
              `[Context Usage] ⚠️ 请求预算已逼近压缩临界线：${fmtTok(info.budgetUsedTokens)} / ${fmtTok(info.requestBudget)} (${pct.toFixed(1)}%)，` +
              `距离触发自动 Compaction 仅剩约 ${fmtTok(remaining)} tokens。若处于多步工具循环中，建议尽快收尾当前阶段或将重要状态写入 Artifact。`,
          },
        ],
      };
    }
    if (currentTier === 'warn85') {
      return {
        injectSteps: [
          {
            ephemeralMessage:
              `[Context Usage] 请求预算已进入高水位：${fmtTok(info.budgetUsedTokens)} / ${fmtTok(info.requestBudget)} (${pct.toFixed(1)}%)，` +
              `距离触发自动 Compaction 剩余约 ${fmtTok(remaining)} tokens。请注意精简后续大文件读取或命令输出。`,
          },
        ],
      };
    }
  }

  return {};
}

export async function runLifecycleHook(eventType) {
  let payload = {};
  try {
    const raw = readFileSync(0, 'utf8');
    if (raw && raw.trim()) payload = JSON.parse(raw);
  } catch {
    payload = {};
  }
  try {
    const out = evaluateLifecycleHook(eventType, payload);
    process.stdout.write(`${JSON.stringify(out)}\n`);
  } catch {
    process.stdout.write('{}\n');
  }
}

// --- python -------------------------------------------------------------------

const PYTHON_CANDIDATES = process.platform === 'win32'
  ? [['python', []], ['py', ['-3']], ['python3', []]]
  : [['python3', []], ['python', []], ['/opt/homebrew/bin/python3', []], ['/usr/local/bin/python3', []]];
let python = null;

function run(cmd, args, timeout = 20000) {
  return new Promise((resolvePromise, reject) => {
    execFile(cmd, args, {
      timeout,
      maxBuffer: 16 * 1024 * 1024,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      windowsHide: true,
    }, (err, stdout, stderr) => (err ? reject(new Error((stderr || err.message).trim())) : resolvePromise(stdout)));
  });
}

async function findPython() {
  if (python) return python;
  for (const [cmd, pre] of PYTHON_CANDIDATES) {
    try {
      const out = await run(cmd, [...pre, '-c', 'import sys; print(f"Python {sys.version_info[0]}.{sys.version_info[1]}")'], 5000);
      if (/^Python 3\./.test(out.trim())) {
        python = { cmd, pre };
        return python;
      }
    } catch { /* try the next candidate */ }
  }
  throw new Error('Python 3 not found. Install it from python.org and restart Antigravity.');
}

// --- live model catalog -------------------------------------------------------

let catalogFetchedAt = 0;

function lsRequest(method, body) {
  const addr = (process.env.ANTIGRAVITY_LS_ADDRESS || '').trim();
  const csrf = (process.env.ANTIGRAVITY_CSRF_TOKEN || '').trim();
  if (!addr || !csrf) return Promise.reject(new Error('no language server address'));
  const base = new URL(addr.startsWith('http') ? addr : `http://${addr}`);
  const data = JSON.stringify(body);
  const transport = base.protocol === 'https:' ? https : http;
  return new Promise((resolvePromise, reject) => {
    const req = transport.request({
      hostname: base.hostname,
      port: base.port,
      path: `/exa.language_server_pb.LanguageServerService/${method}`,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Connect-Protocol-Version': '1',
        'x-codeium-csrf-token': csrf,
        'Content-Length': Buffer.byteLength(data),
      },
      timeout: 5000,
      rejectUnauthorized: false,
    }, (res) => {
      let text = '';
      res.on('data', (chunk) => { text += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error(`${method}: HTTP ${res.statusCode}`));
          return;
        }
        try { resolvePromise(JSON.parse(text)); } catch (e) { reject(e); }
      });
    });
    req.on('timeout', () => req.destroy(new Error(`${method}: timeout`)));
    req.on('error', reject);
    req.end(data);
  });
}

async function refreshCatalog() {
  if (Date.now() - catalogFetchedAt < CATALOG_TTL_MS && existsSync(CATALOG_PATH)) return;
  catalogFetchedAt = Date.now();
  try {
    const resp = await lsRequest('GetAvailableModels', { metadata: { ideName: 'antigravity', extensionName: 'antigravity' } });
    const models = resp?.response?.models || resp?.models || {};
    const catalog = {};
    for (const [id, m] of Object.entries(models)) {
      const maxTokens = Number(m?.maxTokens) || 0;
      if (maxTokens > 0) catalog[id] = { maxTokens, displayName: m.displayName || '' };
    }
    if (Object.keys(catalog).length) {
      mkdirSync(DATA_DIR, { recursive: true });
      writeFileSync(CATALOG_PATH, JSON.stringify(catalog));
    }
  } catch (err) {
    console.error('[context] model catalog unavailable, using built-in limits:', err.message);
  }
}

// --- report -------------------------------------------------------------------

const cache = new Map(); // activeId -> { key, report }

function workspacePath(uri) {
  if (!uri || !uri.startsWith('file:')) return '';
  try {
    return fileURLToPath(uri);
  } catch {
    let p = decodeURIComponent(uri.replace(/^file:\/\//, ''));
    if (/^\/[A-Za-z]:/.test(p)) p = p.slice(1);
    return p;
  }
}

function fileStamp(path) {
  try {
    const s = statSync(path);
    return `${s.mtimeMs}:${s.size}`;
  } catch {
    return '-';
  }
}

async function fetchLiveTrajectorySummaries() {
  try {
    const resp = await lsRequest('GetAllCascadeTrajectories', {});
    return resp?.trajectorySummaries || resp?.response?.trajectorySummaries || {};
  } catch {
    return {};
  }
}

function subagentCacheSignature(activeId, liveSummaries, cachedSubagents) {
  const ids = new Set();
  if (Array.isArray(cachedSubagents)) {
    for (const s of cachedSubagents) {
      if (s?.conversation_id) ids.add(s.conversation_id);
    }
  }
  for (const [id, info] of Object.entries(liveSummaries)) {
    const meta = info?.trajectoryMetadata;
    if (meta?.parentConversationId === activeId || meta?.rootConversationId === activeId) {
      ids.add(id);
    }
  }
  const parts = [];
  for (const id of Array.from(ids).sort()) {
    const subDb = join(CONVERSATIONS, `${id}.db`);
    const live = liveSummaries[id];
    const liveSig = live ? `${live.status || ''}:${live.stepCount || 0}:${live.latestExecutorTerminationReason || ''}` : '';
    parts.push(`${id}@${fileStamp(subDb)}@${fileStamp(`${subDb}-wal`)}@${liveSig}`);
  }
  return parts.join(';');
}

const ACTIVE_SUBAGENT_STATES = new Set(['running', 'generating', 'starting']);

function mergeLiveSubagentStates(subagents, liveSummaries) {
  if (!Array.isArray(subagents) || !subagents.length) return;
  for (const s of subagents) {
    const live = liveSummaries[s.conversation_id];
    if (!live) continue;
    const status = live.status || '';
    const term = live.latestExecutorTerminationReason || '';
    if (status === 'CASCADE_RUN_STATUS_RUNNING' && s.state !== 'generating') {
      s.state = 'running';
    } else if (status === 'CASCADE_RUN_STATUS_IDLE') {
      if (term.endsWith('ERROR')) {
        s.state = 'error';
      } else if (s.has_reported) {
        s.state = 'reported';
      } else if (term.endsWith('USER_CANCELED')) {
        s.state = 'canceled';
      } else if (s.state === 'running' || s.state === 'generating') {
        s.state = 'idle';
      }
    }
  }
  subagents.sort((a, b) => {
    const aActive = ACTIVE_SUBAGENT_STATES.has(a.state);
    const bActive = ACTIVE_SUBAGENT_STATES.has(b.state);
    if (aActive !== bActive) return aActive ? -1 : 1;
    const aTs = aActive ? (a.created_at || 0) : (a.last_active_at || a.created_at || 0);
    const bTs = bActive ? (b.created_at || 0) : (b.last_active_at || b.created_at || 0);
    if (bTs !== aTs) return bTs - aTs;
    return (b.created_at || 0) - (a.created_at || 0);
  });
}

async function report(conversationId, workspaceUri, targetId) {
  const hostId = UUID.test(conversationId || '') ? conversationId : '';
  const activeId = UUID.test(targetId || '') ? targetId : hostId;
  if (!activeId) {
    return { error: 'no-conversation' };
  }
  const db = join(CONVERSATIONS, `${activeId}.db`);
  if (!existsSync(db)) {
    return { error: 'no-database' };
  }
  const [, liveSummaries] = await Promise.all([
    refreshCatalog(),
    fetchLiveTrajectorySummaries(),
  ]);
  const workspace = workspacePath(workspaceUri);
  const cached = cache.get(activeId);
  const subSig = subagentCacheSignature(activeId, liveSummaries, cached?.report?.subagents);
  const key = [fileStamp(db), fileStamp(`${db}-wal`), fileStamp(SCRIPT), workspace, fileStamp(CATALOG_PATH), subSig, hostId].join('|');
  if (cached && cached.key === key) {
    return { ...cached.report, hookConfig: readHookConfig() };
  }

  const { cmd, pre } = await findPython();
  const args = [...pre, SCRIPT, '--db', db, '--json'];
  if (existsSync(CATALOG_PATH)) args.push('--model-catalog', CATALOG_PATH);
  if (workspace) args.push('--workspace', workspace);
  const out = await run(cmd, args);
  const parsed = JSON.parse(out);
  mergeLiveSubagentStates(parsed.subagents, liveSummaries);
  const isChildOfHost =
    hostId &&
    hostId !== activeId &&
    (liveSummaries[activeId]?.trajectoryMetadata?.parentConversationId === hostId ||
      cache.get(hostId)?.report?.subagents?.some((s) => s.conversation_id === activeId));
  if (isChildOfHost && !parsed.parent_conversation_id) {
    parsed.parent_conversation_id = hostId;
  }
  const result = {
    ...parsed,
    host_conversation_id: hostId || activeId,
    hookConfig: readHookConfig(),
    updatedAt: new Date().toISOString(),
  };
  cache.set(activeId, { key, report: result });
  return result;
}

const isMain = Boolean(
  process.argv[1] && (
    process.platform === 'win32'
      ? fileURLToPath(import.meta.url).toLowerCase() === resolve(process.argv[1]).toLowerCase()
      : fileURLToPath(import.meta.url) === resolve(process.argv[1])
  )
);

if (isMain && !process.argv.includes('--hook')) {
  const { Response, SidecarApp } = await import('sidecar_sdk');
  const app = new SidecarApp();

  app.page('/', () => new Response(readFileSync(join(HERE, 'index.html'), 'utf8'), { contentType: 'text/html' }));

  app.api('/api/report', async (data) => {
    try {
      return await report(data.conversationId, data.workspace, data.targetId);
    } catch (err) {
      return { error: 'failed', message: err.message };
    }
  }, 'GET');

  app.api('/api/config', async () => {
    return { ok: true, hookConfig: readHookConfig() };
  }, 'GET');

  app.api('/api/config', async (data) => {
    try {
      return { ok: true, hookConfig: writeHookConfig(data || {}) };
    } catch (err) {
      return { ok: false, error: err.message };
    }
  }, 'POST');

  app.run();
  refreshCatalog();
}
