// Sidecar backend for the Context Usage panel.
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
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Response, SidecarApp } from 'sidecar_sdk';

const HERE = dirname(fileURLToPath(import.meta.url));
const SCRIPT = join(HERE, 'context_report.py');
const GEMINI_DIR = process.env.ANTIGRAVITY_GEMINI_DIR || join(homedir(), '.gemini');
const CONVERSATIONS = join(GEMINI_DIR, 'antigravity', 'conversations');
const DATA_DIR = process.env.ANTIGRAVITY_EXECUTABLE_DATA_DIR || join(HERE, '.data');
const CATALOG_PATH = join(DATA_DIR, 'model-catalog.json');
const CATALOG_TTL_MS = 10 * 60 * 1000;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

const app = new SidecarApp();

// --- python -------------------------------------------------------------------

const PYTHON_CANDIDATES = process.platform === 'win32'
  ? [['python', []], ['py', ['-3']], ['python3', []]]
  : [['python3', []], ['python', []], ['/opt/homebrew/bin/python3', []], ['/usr/local/bin/python3', []]];
let python = null;

function run(cmd, args, timeout = 20000) {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, {
      timeout,
      maxBuffer: 16 * 1024 * 1024,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
      windowsHide: true,
    }, (err, stdout, stderr) => (err ? reject(new Error((stderr || err.message).trim())) : resolve(stdout)));
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
  return new Promise((resolve, reject) => {
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
        try { resolve(JSON.parse(text)); } catch (e) { reject(e); }
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
  if (cached && cached.key === key) return cached.report;

  const { cmd, pre } = await findPython();
  const args = [...pre, SCRIPT, '--db', db, '--json'];
  if (existsSync(CATALOG_PATH)) args.push('--model-catalog', CATALOG_PATH);
  if (workspace) args.push('--workspace', workspace);
  const out = await run(cmd, args);
  const parsed = JSON.parse(out);
  mergeLiveSubagentStates(parsed.subagents, liveSummaries);
  if (hostId && hostId !== activeId && !parsed.parent_conversation_id) {
    parsed.parent_conversation_id = hostId;
  }
  const result = {
    ...parsed,
    host_conversation_id: hostId || activeId,
    updatedAt: new Date().toISOString(),
  };
  cache.set(activeId, { key, report: result });
  return result;
}

app.page('/', () => new Response(readFileSync(join(HERE, 'index.html'), 'utf8'), { contentType: 'text/html' }));

app.api('/api/report', async (data) => {
  try {
    return await report(data.conversationId, data.workspace, data.targetId);
  } catch (err) {
    return { error: 'failed', message: err.message };
  }
}, 'GET');

app.run();
refreshCatalog();
