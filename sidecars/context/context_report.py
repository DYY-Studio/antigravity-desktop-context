#!/usr/bin/env python3
"""Context window usage report for Antigravity 2.0 conversations.

A port of the agy CLI `/context` command (cli/model.ContextModel in agy 1.4.4).
The CLI reads the live trajectory from memory; this script reads the same
protobufs from the conversation database (`steps` and `gen_metadata` tables).
"""

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
from urllib.parse import unquote, urlparse

# --- protobuf ----------------------------------------------------------------


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while pos < len(buf):
        b = buf[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, pos
        shift += 7
    raise ValueError("Truncated varint")


class Msg:
    """Lazily decoded protobuf message: field number -> list of raw values."""

    def __init__(self, buf: bytes | None):
        self.fields: dict[int, list] = {}
        pos, n = 0, len(buf or b"")
        while pos < n:
            try:
                tag, pos = read_varint(buf, pos)
                fn, wt = tag >> 3, tag & 7
                if wt == 0:
                    val, pos = read_varint(buf, pos)
                elif wt == 2:
                    ln, pos = read_varint(buf, pos)
                    val = buf[pos : pos + ln]
                    pos += ln
                elif wt == 1:
                    val, pos = buf[pos : pos + 8], pos + 8
                elif wt == 5:
                    val, pos = buf[pos : pos + 4], pos + 4
                else:
                    break
            except ValueError:
                break
            self.fields.setdefault(fn, []).append(val)

    def has(self, fn: int) -> bool:
        return fn in self.fields

    def int(self, fn: int, signed32: bool = False) -> int:
        v = self.fields.get(fn, [0])[-1]
        v = v if isinstance(v, int) else 0
        if signed32 and v >= 1 << 63:
            v -= 1 << 64
        return v

    def bytes(self, fn: int) -> bytes:
        v = self.fields.get(fn, [b""])[-1]
        return v if isinstance(v, bytes) else b""

    def str(self, fn: int) -> str:
        return self.bytes(fn).decode("utf-8", errors="replace")

    def msg(self, fn: int) -> "Msg | None":
        return Msg(self.bytes(fn)) if self.has(fn) else None

    def msgs(self, fn: int) -> list["Msg"]:
        return [Msg(v) for v in self.fields.get(fn, []) if isinstance(v, bytes)]


# CortexStep oneof payload field numbers.
STEP_VIEW_FILE = 14
STEP_USER_INPUT = 19
STEP_PLANNER_RESPONSE = 20
STEP_CHECKPOINT = 30
STEP_CODE_ACTION = 10

STATUS_PENDING = 1
STATUS_RUNNING = 2
STATUS_DONE = 3
STATUS_INVALID = 4
STATUS_CLEARED = 5
STATUS_CANCELED = 6
STATUS_ERROR = 7
STATUS_GENERATING = 8
STATUS_WAITING = 9
STATUS_INTERRUPTED = 10
STATUS_HALTED = 11

STEP_STATUS_NAMES = {
    STATUS_PENDING: "pending",
    STATUS_RUNNING: "running",
    STATUS_DONE: "done",
    STATUS_INVALID: "error",
    STATUS_CANCELED: "canceled",
    STATUS_ERROR: "error",
    STATUS_GENERATING: "generating",
    STATUS_WAITING: "waiting",
    STATUS_INTERRUPTED: "canceled",
    STATUS_HALTED: "canceled",
}

TERM_REASON_NAMES = {
    1: "error",
    2: "user_canceled",
    3: "max_invocations",
    4: "no_tool_call",
    5: "max_forced_invocations",
    6: "early_continue",
    7: "terminal_step_type",
    8: "terminal_custom_hook",
    9: "injected_response",
    10: "max_token_budget_exceeded",
    11: "halted_step",
}

# TokenType values of TopLevelTokenGroup.type.
TOKEN_SYSTEM_PROMPT, TOKEN_TOOLS, TOKEN_RULES, TOKEN_MCP_TOOLS, TOKEN_SKILLS = 1, 2, 3, 4, 5
TOKEN_SUBAGENTS, TOKEN_CHAT_MESSAGES, TOKEN_WORKFLOWS, TOKEN_PLUGINS, TOKEN_FLOWS = 6, 7, 8, 9, 10

# CortexStepType names (exa.cortex_pb), used to parse "N: TYPE" chat item names.
STEP_TYPES = {
    "UNSPECIFIED": 0, "DUMMY": 1, "FINISH": 2, "PLAN_INPUT": 3, "MQUERY": 4, "CODE_ACTION": 5,
    "GIT_COMMIT": 6, "GREP_SEARCH": 7, "VIEW_FILE": 8, "LIST_DIRECTORY": 9, "COMPILE": 10,
    "VIEW_CODE_ITEM": 13, "USER_INPUT": 14, "PLANNER_RESPONSE": 15, "ERROR_MESSAGE": 17,
    "RUN_COMMAND": 21, "CHECKPOINT": 23, "PROPOSE_CODE": 24, "FIND": 25, "SUGGESTED_RESPONSES": 27,
    "COMMAND_STATUS": 28, "MEMORY": 29, "READ_URL_CONTENT": 31, "VIEW_CONTENT_CHUNK": 32,
    "SEARCH_WEB": 33, "RETRIEVE_MEMORY": 34, "MCP_TOOL": 38, "MANAGER_FEEDBACK": 39,
    "TOOL_CALL_PROPOSAL": 40, "TOOL_CALL_CHOICE": 41, "TRAJECTORY_CHOICE": 42, "CLIPBOARD": 45,
    "VIEW_FILE_OUTLINE": 47, "POST_PR_REVIEW": 49, "LIST_RESOURCES": 51, "READ_RESOURCE": 52,
    "LINT_DIFF": 53, "FIND_ALL_REFERENCES": 54, "BRAIN_UPDATE": 55, "OPEN_BROWSER_URL": 56,
    "RUN_EXTENSION_CODE": 57, "PROPOSAL_FEEDBACK": 59, "TRAJECTORY_SEARCH": 60,
    "EXECUTE_BROWSER_JAVASCRIPT": 61, "LIST_BROWSER_PAGES": 62, "CAPTURE_BROWSER_SCREENSHOT": 63,
    "CLICK_BROWSER_PIXEL": 64, "READ_TERMINAL": 65, "CAPTURE_BROWSER_CONSOLE_LOGS": 66,
    "READ_BROWSER_PAGE": 67, "BROWSER_GET_DOM": 68, "CODE_SEARCH": 73, "BROWSER_INPUT": 74,
    "BROWSER_MOVE_MOUSE": 75, "BROWSER_SELECT_OPTION": 76, "BROWSER_SCROLL_UP": 77,
    "BROWSER_SCROLL_DOWN": 78, "BROWSER_CLICK_ELEMENT": 79, "BROWSER_PRESS_KEY": 80,
    "TASK_BOUNDARY": 81, "NOTIFY_USER": 82, "CODE_ACKNOWLEDGEMENT": 83, "INTERNAL_SEARCH": 84,
    "BROWSER_SUBAGENT": 85, "FILE_CHANGE": 86, "MOVE": 87, "BROWSER_SCROLL": 88,
    "KNOWLEDGE_GENERATION": 89, "EPHEMERAL_MESSAGE": 90, "GENERATE_IMAGE": 91,
    "DELETE_DIRECTORY": 92, "COMPILE_APPLET": 93, "INSTALL_APPLET_DEPENDENCIES": 94,
    "INSTALL_APPLET_PACKAGE": 95, "BROWSER_RESIZE_WINDOW": 96, "BROWSER_DRAG_PIXEL_TO_PIXEL": 97,
    "CONVERSATION_HISTORY": 98, "KNOWLEDGE_ARTIFACTS": 99, "SEND_COMMAND_INPUT": 100,
    "SYSTEM_MESSAGE": 101, "WAIT": 102, "AGENCY_TOOL_CALL": 103, "CIDER_AGENT_DUMMY": 104,
    "BUILD_CLEANER": 105, "BLAZE_BUILD_TARGETS": 106, "BLAZE_TEST_TARGETS": 107,
    "SET_UP_FIREBASE": 108, "MOMA": 109, "RESTART_DEV_SERVER": 110, "DEPLOY_FIREBASE": 111,
    "SHELL_EXEC": 112, "BROWSER_MOUSE_WHEEL": 113, "LINT_APPLET": 114, "KI_INSERTION": 116,
    "RETRIEVE_CONTENT": 117, "CRITIQUE": 118, "FINDINGS": 119, "BROWSER_MOUSE_UP": 120,
    "BROWSER_MOUSE_DOWN": 121, "WORKSPACE_API": 122, "BROWSER_LIST_NETWORK_REQUESTS": 123,
    "BROWSER_GET_NETWORK_REQUEST": 124, "BROWSER_REFRESH_PAGE": 125, "EDIT_NOTEBOOK": 126,
    "INVOKE_SUBAGENT": 127, "WRITE_BLOB": 128, "READ_NOTEBOOK": 129, "PROPOSE_AI_COMMENTS": 130,
    "START_CODE_REVIEW": 131, "GENERIC": 132, "SET_UP_CLOUD_SQL": 133, "EXECUTE_NOTEBOOK": 134,
    "CLOUD_SQL_UPDATE_SCHEMA": 135, "RPC_ACTION": 136, "CLOUD_SQL_EXECUTE_SQL": 137,
    "ASK_QUESTION": 138, "DIRECTORY_RULES": 139, "TOOL_SEARCH": 140,
}

CAT_USER, CAT_AGENT, CAT_TOOL, CAT_CHECKPOINT, CAT_SYSTEM = 0, 1, 2, 3, 4

# --- formatting --------------------------------------------------------------


def fmt_tok(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000.0
        return f"{v:.0f}M" if v.is_integer() else f"{v:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000.0:.1f}k"
    return str(n)


def fmt_ctx_limit(n: int) -> str:
    return fmt_tok(n)


def go_div4(n: int) -> int:
    """Go's int division by 4 (truncates toward zero)."""
    return int(n / 4)


def shorten_path(uri: str, home_dir: str | None = None) -> str:
    p = uri
    if p.startswith("file:"):
        parsed = urlparse(p)
        p = unquote(parsed.path)
        if len(p) > 2 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        elif parsed.netloc and parsed.netloc != "localhost":
            p = f"//{parsed.netloc}{p}"
    home = str(home_dir if home_dir is not None else Path.home()).replace("\\", "/").rstrip("/")
    p = p.replace("\\", "/")
    if home:
        case_insensitive = os.name == "nt" or (len(home) >= 2 and home[1] == ":")
        p_cmp = p.lower() if case_insensitive else p
        h_cmp = home.lower() if case_insensitive else home
        if p_cmp == h_cmp:
            return "~"
        if p_cmp.startswith(h_cmp + "/"):
            return "~" + p[len(home) :]
    return p


# --- stats (model.ContextModel.collectStats) ---------------------------------


def step_category(step_type: int) -> int:
    if step_type == 14:
        return CAT_USER
    if step_type == 15:
        return CAT_AGENT
    if step_type == 23:
        return CAT_CHECKPOINT
    if step_type in (0, 2, 101):
        return CAT_SYSTEM
    return CAT_TOOL


STEP_GENERIC = 140
GENERIC_TYPE = 132
ORIGINAL_STEP_TYPE_URL = "/gemini_coder.Step"


def packed_original_step(generic: Msg) -> bytes | None:
    """The typed step a GENERIC step wraps (rehydration.originalStepAny)."""
    result = generic.msg(2)
    if result is None:
        return None
    payload = result.msg(6)
    if payload is not None and payload.str(1).endswith(ORIGINAL_STEP_TYPE_URL):
        return payload.bytes(2)
    for entry in result.msgs(5):
        if entry.str(1) == "__original_cortex_step":
            value = entry.msg(2)
            if value is not None and value.str(1).endswith(ORIGINAL_STEP_TYPE_URL):
                return value.bytes(2)
    return None


def is_skill_file(path: str) -> bool:
    return path.replace("\\", "/").endswith("/SKILL.md")


class Step:
    def __init__(self, pos: int, payload: bytes, step_type: int, status: int, metadata: bytes | None):
        self.pos = pos
        self.pb = Msg(payload)
        self.type = self.pb.int(1) if self.pb.has(1) else step_type
        self.status = self.pb.int(4) if self.pb.has(4) else status
        self.md = self.pb.msg(5) or Msg(metadata)
        self.generic_pb = self.pb.msg(STEP_GENERIC) if self.type == GENERIC_TYPE else None
        self.view_file_override: Msg | None = None
        if self.type == GENERIC_TYPE:
            self.rehydrate()

    def rehydrate(self):
        """Turn a GENERIC tool step back into its typed step, like the CLI's rehydration.RehydrateStep."""
        generic = self.pb.msg(STEP_GENERIC)
        if generic is None:
            return
        original = packed_original_step(generic)
        if original is not None:
            self.pb = Msg(original)
            self.type = self.pb.int(1)
            self.md = self.pb.msg(5) or Msg(b"")
            return
        tc = self.md.msg(4)
        if tc is None or tc.str(2) not in ("view_file", "read_file"):
            return
        try:
            args = json.loads(tc.str(3) or "{}")
        except json.JSONDecodeError:
            return
        path = next((args[k] for k in ("AbsolutePath", "Path", "FilePath") if isinstance(args.get(k), str) and args[k]), "")
        if path:
            self.type = 8  # VIEW_FILE
            self.view_file_override = Msg(
                b"\x0a" + _varint(len(path.encode())) + path.encode() + (b"\x88\x01\x01" if is_skill_file(path) else b"")
            )

    def payload(self, fn: int) -> Msg | None:
        if fn == STEP_VIEW_FILE and self.view_file_override is not None:
            return self.view_file_override
        return self.pb.msg(fn)


def _varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def estimate_step_tokens(step: Step) -> int:
    ui = step.payload(STEP_USER_INPUT)
    pr = step.payload(STEP_PLANNER_RESPONSE)
    if ui is not None:
        n = len(ui.bytes(2))  # user_response
    elif pr is not None:
        n = len(pr.bytes(1))  # response
    else:
        tc = step.md.msg(4)  # tool_call
        n = len(tc.bytes(3)) if tc else 0  # arguments_json
        out = step.md.int(23, signed32=True)  # tool_call_output_tokens
        if out > 0:
            return out
    if n == 0:
        return 0
    return max(1, go_div4(n))


def step_tokens(step: Step) -> int:
    usage = step.md.msg(9)  # model_usage
    if usage is not None:
        return usage.int(3)  # output_tokens
    return estimate_step_tokens(step)


def post_turn_step_tokens(step: Step) -> int:
    base = step_tokens(step)
    if step.md.int(23, signed32=True) <= 0 and step.generic_pb is not None:
        res = step.generic_pb.msg(2)
        if res is not None:
            base += go_div4(len(res.bytes(1)))
    return base


def effective_budget_tokens(steps: list[Step], gen: Msg | None, st: dict) -> int:
    """Actual token count checked against request_budget (modelReportedPrefixTokens + post-turn steps)."""
    if st.get("compacted"):
        return st["total"]
    ckpt_pos = -1
    for i, s in enumerate(steps):
        if s.payload(STEP_CHECKPOINT) is not None and s.status == STATUS_DONE:
            ckpt_pos = i
    pos_last = -1
    prefix_tok = 0
    for i in range(len(steps) - 1, ckpt_pos, -1):
        s = steps[i]
        if s.payload(STEP_PLANNER_RESPONSE) is None:
            continue
        u = s.md.msg(9)
        if u is None:
            continue
        inp, out_tok, cache_r, cache_w = u.int(2), u.int(3), u.int(5), u.int(6)
        if inp + cache_r + cache_w > 0:
            pos_last = i
            prefix_tok = inp + cache_r + cache_w + out_tok
            break
    if pos_last >= 0:
        delta_tok = sum(
            post_turn_step_tokens(s)
            for s in steps[pos_last + 1 :]
            if s.payload(STEP_CHECKPOINT) is None
        )
        return max(st["total"], prefix_tok + delta_tok)
    return st["total"]


def compaction_tokens(step: Step) -> int:
    for ri in reversed(step.md.msgs(28)):  # retry_infos
        usage = ri.msg(2)
        if usage is None:
            continue
        resp = usage.int(10)
        if resp:
            return resp
        out, thinking = usage.int(3), usage.int(9)
        return out - thinking if thinking < out else 0
    return 0


def checkpoint_summary_tokens(step: Step) -> int:
    t = compaction_tokens(step)
    if t:
        return t
    ck = step.payload(STEP_CHECKPOINT) or Msg(b"")
    return go_div4(len(ck.bytes(4)) + len(ck.bytes(5)) + len(ck.bytes(6)))


def checkpoint_newer_than_generation(steps: list[Step], chat_start: Msg | None) -> tuple[int, Step | None]:
    i = len(steps) - 1
    while i >= 0:
        s = steps[i]
        if s.payload(STEP_CHECKPOINT) is not None and s.status == STATUS_DONE:
            break
        i -= 1
    if i < 0:
        return -1, None
    for s in steps[i + 1 :]:
        if s.payload(STEP_PLANNER_RESPONSE) is not None and s.status == STATUS_DONE:
            return -1, None
    gen_ck = chat_start.int(2, signed32=True) if chat_start is not None else -1
    ck_idx = steps[i].payload(STEP_CHECKPOINT).int(1)
    if gen_ck >= ck_idx:
        return -1, None
    return i, steps[i]


def chat_tokens_by_category(items: list[Msg]) -> tuple[bool, list[int]]:
    out = [0, 0, 0, 0]  # user, agent, tool, system
    if not items:
        return False, out
    for it in items:
        left, sep, right = it.str(1).partition(": ")
        if not sep or not re.fullmatch(r"[+-]?[0-9]+", left) or right not in STEP_TYPES:
            return False, [0, 0, 0, 0]
        cat = step_category(STEP_TYPES[right])
        tok = it.int(3, signed32=True)
        if cat == CAT_USER:
            out[0] += tok
        elif cat in (CAT_AGENT, CAT_CHECKPOINT):
            out[1] += tok
        elif cat == CAT_TOOL:
            out[2] += tok
        else:
            out[3] += tok
    return True, out


def detect_system_files(home: str, workspace: str | None, app_dir: str) -> list[tuple[str, int]]:
    if not home:
        return []
    gemini = Path(home) / ".gemini"
    disp = "~/" + gemini.name
    found = []
    f = gemini / "GEMINI.md"
    if f.exists() and not f.is_dir():
        found.append(disp + "/GEMINI.md")
    for sub in ("skills", "rules", "agents", "global_workflows"):
        if (gemini / sub).is_dir():
            found.append(f"{disp}/{sub}/")
    if app_dir:
        for sub in ("rules", "skills", "agents", "global_workflows"):
            if (gemini / app_dir / sub).is_dir():
                found.append(f"{disp}/{app_dir}/{sub}/")
    if workspace:
        for root in (".agents", "_agents"):
            for sub in ("rules", "skills", "agents", "workflows"):
                if (Path(workspace) / root / sub).is_dir():
                    found.append(f"{root}/{sub}/")
            hooks = Path(workspace) / root / "hooks.json"
            if hooks.exists() and not hooks.is_dir():
                found.append(f"{root}/hooks.json")
    return [(p, 0) for p in found]


def collect_stats(steps: list[Step], gen: Msg | None, limit: int) -> dict:
    st = {k: 0 for k in (
        "user", "agent", "tool", "checkpoint", "system_messages", "system_prompt", "system_tools",
        "system_overhead", "rules", "skills", "subagents", "legacy_workflows", "workflows",
        "plugins", "mcp_tools",
    )}
    st["has_breakdown"] = False
    st["active_checkpoint"] = -1
    checkpoints, knowledge, artifacts, skill_files = [], [], [], []
    seen_k, seen_a, seen_s = set(), set(), set()

    chat = gen.msg(1) if gen is not None else None
    chat_start = chat.msg(9) if chat is not None else None
    newer_pos, newer_step = checkpoint_newer_than_generation(steps, chat_start)
    has_newer = newer_step is not None
    st["compacted"] = has_newer

    for pos, step in enumerate(steps):
        cat = step_category(step.type)
        tok = step_tokens(step)
        counted = 0 if has_newer and newer_pos > pos else tok
        if cat == CAT_USER:
            st["user"] += counted
        elif cat == CAT_AGENT:
            st["agent"] += counted
        elif cat == CAT_TOOL:
            st["tool"] += counted
        elif cat == CAT_CHECKPOINT:
            st["checkpoint"] += tok
            ck = step.payload(STEP_CHECKPOINT)
            if ck is not None:
                checkpoints.append({
                    "index": ck.int(1),
                    "start": ck.int(11),
                    "end": ck.int(12),
                    "intent": ck.str(4),
                })
        else:
            st["system_messages"] += counted

        ca = step.payload(STEP_CODE_ACTION)
        result = ca.msg(2) if ca is not None else None
        edit = result.msg(1) if result is not None else None
        if edit is not None:
            uri = edit.str(8)  # absolute_uri
            short = shorten_path(uri)
            if ca.int(21) and uri not in seen_a:  # is_artifact_file
                seen_a.add(uri)
                artifacts.append((short, tok))
            if ca.int(24) and uri not in seen_k:  # is_knowledge_file
                seen_k.add(uri)
                knowledge.append((short, tok))

        vf = step.payload(STEP_VIEW_FILE)
        if vf is not None and vf.int(17):  # is_skill_file
            uri = vf.str(1)
            if uri not in seen_s:
                seen_s.add(uri)
                skill_files.append((shorten_path(uri), tok))

    cwm = chat_start.msg(10) if chat_start is not None else None
    breakdown = cwm.msg(3) if cwm is not None else None
    if breakdown is not None:
        st["has_breakdown"] = True
        seen_prompt = seen_tools = False
        chat_tok, chat_items = 0, []
        for g in breakdown.msgs(1):
            gtype, gtok = g.int(2), g.int(4, signed32=True)
            if gtype == TOKEN_SYSTEM_PROMPT:
                extracted = 0
                for child in g.msgs(5):
                    name, ctok = child.str(1), child.int(3, signed32=True)
                    if name == "flows":
                        st["workflows"] += ctok
                    elif name == "skills":
                        st["skills"] += ctok
                    elif name == "subagents":
                        st["subagents"] += ctok
                    else:
                        continue
                    extracted += ctok
                st["system_prompt"] = gtok - extracted
                seen_prompt = True
            elif gtype == TOKEN_TOOLS:
                st["system_tools"] = gtok
                seen_tools = True
            elif gtype == TOKEN_RULES:
                st["rules"] += gtok
            elif gtype == TOKEN_MCP_TOOLS:
                st["mcp_tools"] += gtok
            elif gtype == TOKEN_SKILLS:
                st["skills"] += gtok
            elif gtype == TOKEN_SUBAGENTS:
                st["subagents"] += gtok
            elif gtype == TOKEN_CHAT_MESSAGES:
                chat_tok, chat_items = gtok, g.msgs(5)
            elif gtype == TOKEN_WORKFLOWS:
                st["legacy_workflows"] += gtok
            elif gtype == TOKEN_PLUGINS:
                st["plugins"] += gtok
            elif gtype == TOKEN_FLOWS:
                st["workflows"] += gtok

        if not seen_prompt and not seen_tools and chat_tok:
            usage = chat.msg(4)
            api_input = usage.int(2) + usage.int(5) if usage is not None else 0
            if chat_tok < api_input:
                st["system_overhead"] = api_input - chat_tok

        if chat_tok and not has_newer:
            ok, cats = chat_tokens_by_category(chat_items)
            if ok:
                st["user"], st["agent"], st["tool"], st["system_messages"] = cats
            else:
                est = st["user"] + st["agent"] + st["tool"]
                if est == 0:
                    st["agent"] = chat_tok
                else:
                    f = chat_tok / est
                    st["user"] = int(st["user"] * f)
                    st["agent"] = int(st["agent"] * f)
                    st["tool"] = int(st["tool"] * f)

    if chat_start is not None:
        st["active_checkpoint"] = chat_start.int(2, signed32=True)
    if has_newer:
        st["agent"] += checkpoint_summary_tokens(newer_step)
        st["active_checkpoint"] = newer_step.payload(STEP_CHECKPOINT).int(1)

    st["total"] = sum(st[k] for k in (
        "user", "agent", "tool", "system_messages", "system_prompt", "system_tools", "system_overhead",
        "rules", "skills", "subagents", "legacy_workflows", "workflows", "plugins", "mcp_tools",
    ))
    st["limit"] = limit
    st["checkpoints"] = checkpoints
    st["knowledge_files"] = knowledge
    st["artifact_files"] = artifacts
    st["skill_files"] = skill_files
    return st


def pb_seconds(m: Msg | None) -> float:
    if m is None:
        return 0.0
    return m.int(1) + m.int(2) / 1_000_000_000.0


def empty_cache_stats() -> dict:
    return {
        "turns": 0,
        "last_input_tokens": 0,
        "last_cache_read_tokens": 0,
        "last_prompt_tokens": 0,
        "last_hit_pct": 0.0,
        "total_input_tokens": 0,
        "total_cache_read_tokens": 0,
        "total_prompt_tokens": 0,
        "total_hit_pct": 0.0,
    }


def collect_cache_stats(steps: list[Step], gen: Msg | None = None) -> dict:
    turns = 0
    total_input = 0
    total_cache_read = 0
    last_input = 0
    last_cache_read = 0

    for s in steps:
        u = s.md.msg(9)
        if u is None:
            continue
        inp, out_tok, cache = u.int(2), u.int(3), u.int(5)
        if inp > 0 or out_tok > 0 or cache > 0:
            turns += 1
            total_input += inp
            total_cache_read += cache
            last_input = inp
            last_cache_read = cache

    chat = gen.msg(1) if gen is not None else None
    gen_u = chat.msg(4) if chat is not None else None
    if gen_u is not None:
        g_in, g_out, g_cache = gen_u.int(2), gen_u.int(3), gen_u.int(5)
        if (g_in > 0 or g_out > 0 or g_cache > 0) and (g_in, g_cache) != (last_input, last_cache_read):
            turns += 1
            total_input += g_in
            total_cache_read += g_cache
            last_input = g_in
            last_cache_read = g_cache

    last_prompt = last_input + last_cache_read
    total_prompt = total_input + total_cache_read
    return {
        "turns": turns,
        "last_input_tokens": last_input,
        "last_cache_read_tokens": last_cache_read,
        "last_prompt_tokens": last_prompt,
        "last_hit_pct": round(pct(last_cache_read, last_prompt), 1),
        "total_input_tokens": total_input,
        "total_cache_read_tokens": total_cache_read,
        "total_prompt_tokens": total_prompt,
        "total_hit_pct": round(pct(total_cache_read, total_prompt), 1),
    }


def empty_output_stats() -> dict:
    return {
        "turns": 0,
        "last_output_tokens": 0,
        "last_thinking_tokens": 0,
        "last_response_tokens": 0,
        "last_ttft_seconds": 0.0,
        "last_stream_seconds": 0.0,
        "last_duration_seconds": 0.0,
        "last_tps": 0.0,
        "total_output_tokens": 0,
        "total_duration_seconds": 0.0,
        "avg_tps": 0.0,
    }


def collect_output_stats(steps: list[Step], gen: Msg | None = None) -> dict:
    turns = 0
    total_out = 0
    timed_out = 0
    total_dur = 0.0
    last_out = 0
    last_think = 0
    last_resp = 0
    last_ttft = 0.0
    last_stream = 0.0
    last_dur = 0.0
    last_sig = (0, 0, 0)
    last_step_timed = False

    for s in steps:
        u = s.md.msg(9)
        if u is None:
            continue
        inp, out_tok, cache = u.int(2), u.int(3), u.int(5)
        if out_tok <= 0:
            continue
        think_tok, resp_tok = u.int(9), u.int(10)
        total_out += out_tok
        last_out = out_tok
        last_think = think_tok
        last_resp = resp_tok
        last_sig = (inp, out_tok, cache)

        t_create = pb_seconds(s.md.msg(1))
        t_view = pb_seconds(s.md.msg(6))
        t_end = pb_seconds(s.md.msg(7)) or pb_seconds(s.md.msg(8))
        if t_create > 0.0 and t_end > t_create:
            dur = t_end - t_create
            ttft = max(0.0, min(dur, t_view - t_create)) if t_view >= t_create else 0.0
            stream = max(0.0, dur - ttft)
            turns += 1
            timed_out += out_tok
            total_dur += dur
            last_ttft = ttft
            last_stream = stream
            last_dur = dur
            last_step_timed = True
        else:
            last_step_timed = False

    chat = gen.msg(1) if gen is not None else None
    gen_u = chat.msg(4) if chat is not None else None
    if gen_u is not None:
        g_in, g_out, g_cache = gen_u.int(2), gen_u.int(3), gen_u.int(5)
        g_ttft = pb_seconds(chat.msg(11))
        g_stream = pb_seconds(chat.msg(12))
        g_dur = g_ttft + g_stream
        if g_out > 0 and (g_in, g_out, g_cache) != last_sig:
            total_out += g_out
            last_out = g_out
            last_think = gen_u.int(9)
            last_resp = gen_u.int(10)
            if g_dur > 0.0:
                turns += 1
                timed_out += g_out
                total_dur += g_dur
                last_ttft = g_ttft
                last_stream = g_stream
                last_dur = g_dur
        elif g_out > 0 and not last_step_timed and g_dur > 0.0:
            turns += 1
            timed_out += last_out
            total_dur += g_dur
            last_ttft = g_ttft
            last_stream = g_stream
            last_dur = g_dur

    return {
        "turns": turns,
        "last_output_tokens": last_out,
        "last_thinking_tokens": last_think,
        "last_response_tokens": last_resp,
        "last_ttft_seconds": round(last_ttft, 2),
        "last_stream_seconds": round(last_stream, 2),
        "last_duration_seconds": round(last_dur, 2),
        "last_tps": round(last_out / last_dur, 1) if last_dur > 0.0 else 0.0,
        "total_output_tokens": total_out,
        "total_duration_seconds": round(total_dur, 2),
        "avg_tps": round(timed_out / total_dur, 1) if total_dur > 0.0 else 0.0,
    }


# --- rendering ---------------------------------------------------------------

CATEGORIES = [
    ("◉", "User messages", "user", True),
    ("◉", "Agent responses", "agent", True),
    ("◉", "Tool calls", "tool", True),
    ("⛁", "System prompt", "system_prompt", False),
    ("⛁", "System tools", "system_tools", False),
    ("⛁", "System overhead", "system_overhead", False),
    ("⛁", "Rules", "rules", False),
    ("⛁", "Skills", "skills", False),
    ("⛁", "Subagents", "subagents", False),
    ("⛁", "Legacy workflows", "legacy_workflows", False),
    ("⛁", "Workflows", "workflows", False),
    ("⛁", "Plugins", "plugins", False),
    ("⛁", "MCP tools", "mcp_tools", False),
    ("◉", "System messages", "system_messages", False),
]

GRID_COLORS = ["\x1b[90m", "\x1b[36m", "\x1b[35m", "\x1b[33m", "\x1b[34m", "\x1b[2m"]


def pct(n: int, limit: int) -> float:
    return n / limit * 100.0 if limit else 0.0


def stats_heading(st: dict) -> str:
    if st["compacted"]:
        return "Estimated after compaction · updates next turn"
    if st["has_breakdown"]:
        return "Token usage by category"
    return "Estimated usage (awaiting generation)"


def render_stats(st: dict, model: str) -> str:
    used, limit = st["total"], st["limit"]
    out = []
    if limit:
        out.append(f"{model} · {fmt_tok(used)}/{fmt_ctx_limit(limit)} tokens\n ({min(100.0, pct(used, limit)):.1f}%)\n")
    else:
        out.append(f"{model} · {fmt_tok(used)}/Unknown tokens\n")
    out.append(stats_heading(st) + "\n")
    for icon, name, key, always in CATEGORIES:
        tok = st[key]
        if always or tok:
            out.append(f"{icon} {name}: {fmt_tok(tok)} tokens ({pct(tok, limit):.1f}%)\n")
    free = limit - used if used < limit else 0
    out.append(f"□ Free space: {fmt_tok(free)} ({pct(free, limit):.1f}%)\n")
    if st["checkpoint"]:
        out.append(f"⊠ Checkpoint buffer: {fmt_tok(st['checkpoint'])} tokens (not counted in usage)\n")
    return "".join(out)


def grid_segments(st: dict) -> list[int]:
    """Token totals of the grid's five colored segments, in drawing order."""
    return [
        st["system_prompt"] + st["system_tools"] + st["system_overhead"],
        st["user"],
        st["agent"],
        st["tool"],
        sum(st[k] for k in ("rules", "skills", "subagents", "legacy_workflows", "workflows", "plugins", "mcp_tools")),
    ]


def render_grid(st: dict, width: int, rows: int, color: bool) -> list[str]:
    cols = max(10, width // 2)
    total_cells = max(1, rows) * cols
    limit = st["limit"]
    counts = [0 if not tok or not limit else max(1, int(tok / limit * total_cells)) for tok in grid_segments(st)]
    used = sum(counts)
    if used > total_cells:
        f = total_cells / used
        counts = [int(c * f) for c in counts]
    counts.append(total_cells - sum(counts))

    cells = []
    for i, n in enumerate(counts):
        glyph = "□" if i == 5 else "◉"
        cells.extend([f"{GRID_COLORS[i]}{glyph}\x1b[0m" if color else glyph] * n)
    return [" ".join(cells[r * cols : (r + 1) * cols]) for r in range(len(cells) // cols)]


def checkpoint_view(st: dict) -> dict | None:
    cks = st["checkpoints"]
    if not cks:
        return None
    active, historical = None, 0
    for ck in cks:
        if st["active_checkpoint"] >= 0 and ck["index"] == st["active_checkpoint"]:
            active = ck
        else:
            historical += 1
    if active is None:
        active, historical = cks[-1], len(cks) - 1
    text = active["intent"]
    if len(text.encode()) > 70:
        text = text.encode()[:67].decode("utf-8", errors="ignore") + "..."
    if not text:
        text = f"steps {active['start']}–{active['end']}"
    return {"count": len(cks), "active_number": active["index"] + 1, "active_text": text, "historical": historical}


def render_checkpoints(st: dict) -> str:
    view = checkpoint_view(st)
    if view is None:
        return ""
    out = [
        f"Checkpoints ({view['count']}) · /rewind\n",
        f"└ Checkpoint {view['active_number']} (active, in context): {view['active_text']}\n",
    ]
    if view["historical"]:
        out.append(f"  {view['historical']} historical checkpoint(s) (summarized, not in context)\n")
    return "".join(out)


def render_file_section(title: str, subtitle: str, items: list[tuple[str, int]]) -> str:
    if not items:
        return ""
    out = [f"\n{title} · {subtitle}\n"]
    for path, tok in items:
        out.append(f" └ {path}: {fmt_tok(tok)} tokens\n" if tok > 0 else f" └ {path}\n")
    return "".join(out)


def fmt_duration(seconds: int) -> str:
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    return f"{h}h {m}m" if m else f"{h}h"


def truncate_text(text: str, max_chars: int = 60) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max(1, max_chars - 1)].rstrip() + "…"


def render_subagents(subagents: list[dict]) -> str:
    if not subagents:
        return ""
    out = [f"\nSubagents ({len(subagents)})\n"]
    for s in subagents:
        role = s.get("role") or s.get("type_name") or s["conversation_id"][:8]
        tname = f" ({s['type_name']})" if s.get("type_name") and s.get("role") else ""
        state = s.get("state", "idle").upper()
        used = fmt_tok(s.get("used_tokens", 0))
        budget = s.get("request_budget_tokens") or s.get("limit_tokens") or 0
        budget_str = f"/{fmt_tok(budget)} budget ({s.get('budget_pct', 0.0):.1f}%)" if budget else ""
        ck = f" · Compacted ×{s['checkpoint_count']}" if s.get("checkpoint_count") else ""
        dur = fmt_duration(s.get("duration_seconds", 0))
        dur_str = f" · {dur}" if dur else ""
        cs = s.get("cache_stats") or {}
        cache_str = (
            f" · cache {fmt_tok(cs['last_cache_read_tokens'])} ({cs['last_hit_pct']:.1f}%)"
            if cs.get("total_cache_read_tokens", 0) > 0
            else ""
        )
        os_st = s.get("output_stats") or {}
        tps_str = (
            f" · {os_st['last_tps']:.1f} tok/s"
            if os_st.get("last_tps", 0) > 0
            else ""
        )
        action = f" — {s['current_action']}" if s.get("current_action") else ""
        out.append(
            f" └ [{state}] {role}{tname}: {used}{budget_str} · {s.get('step_count', 0)} steps{dur_str}{ck}{cache_str}{tps_str}{action}\n"
        )
    return "".join(out)


def render(
    st: dict,
    model: str,
    system_files: list[tuple[str, int]],
    term_width: int,
    color: bool,
    subagents: list[dict] | None = None,
) -> str:
    stats = render_stats(st, model)
    grid_width = max(20, (2 * term_width) // 5)
    if term_width >= 70 and term_width - grid_width - 6 >= 40:
        grid = render_grid(st, grid_width, max(3, stats.count("\n")), color)
        stat_lines = stats.rstrip("\n").split("\n")
        pad = max(10, grid_width // 2) * 2 - 1
        lines = []
        for i in range(max(len(grid), len(stat_lines))):
            g = grid[i] if i < len(grid) else " " * pad
            s = stat_lines[i] if i < len(stat_lines) else ""
            lines.append(f"{g}   {s}".rstrip())
        body = "\n".join(lines) + "\n"
    else:
        grid = render_grid(st, term_width if term_width >= 20 else 40, 3, color)
        body = stats + "\n" + "\n".join(grid) + "\n"

    parts = ["└ Context Usage\n", "\n", body]
    if subagents:
        parts.append(render_subagents(subagents))
    ck = render_checkpoints(st)
    if ck:
        parts += ["\n", ck]
    parts.append(render_file_section("Knowledge files", "/rules", st["knowledge_files"]))
    parts.append(render_file_section("Artifact files", "/artifact", st["artifact_files"]))
    parts.append(render_file_section("Skill files", "/skill", st["skill_files"]))
    parts.append(render_file_section("System files", "auto-loaded", system_files))
    parts.append("\nRelated: /artifact · /skill · /rewind")
    return "".join(parts)


# --- subagents & metadata ----------------------------------------------------


def load(db_path: Path) -> tuple[list[Step], Msg | None]:
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute("SELECT idx, step_type, status, metadata, step_payload FROM steps ORDER BY idx").fetchall()
        steps = [Step(pos, r[4] or b"", r[1], r[2], r[3]) for pos, r in enumerate(rows)]
        row = conn.execute("SELECT data FROM gen_metadata ORDER BY idx DESC LIMIT 1").fetchone()
    finally:
        conn.close()
    return steps, (Msg(row[0]) if row else None)


def load_meta(db_path: Path) -> tuple[Msg | None, int, int]:
    """Read trajectory_metadata_blob and latest executor_metadata termination info."""
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    tmb = None
    term_reason = 0
    term_step_idx = -1
    try:
        try:
            row = conn.execute("SELECT data FROM trajectory_metadata_blob LIMIT 1").fetchone()
            if row and row[0]:
                tmb = Msg(row[0])
        except sqlite3.Error:
            pass
        try:
            erow = conn.execute("SELECT data FROM executor_metadata ORDER BY idx DESC LIMIT 1").fetchone()
            if erow and erow[0]:
                em = Msg(erow[0])
                term_reason = em.int(1)
                term_step_idx = em.int(3)
        except sqlite3.Error:
            pass
    finally:
        conn.close()
    return tmb, term_reason, term_step_idx


def parse_trajectory_lineage(tmb: Msg | None, conv_id: str) -> dict:
    if tmb is None:
        return {
            "parent_conversation_id": "",
            "root_conversation_id": conv_id,
            "created_at": 0,
            "subagent_info": None,
        }
    parent_id = tmb.str(5)
    root_id = tmb.str(6) or parent_id or conv_id
    ts_msg = tmb.msg(2)
    created_at = ts_msg.int(1) if ts_msg is not None else 0
    spec = tmb.msg(8)
    subagent_info = None
    if parent_id or spec is not None:
        subagent_info = {
            "type_name": spec.str(1) if spec is not None else "",
            "role": spec.str(2) if spec is not None else "",
            "nesting_depth": tmb.int(17) or (1 if parent_id else 0),
        }
    return {
        "parent_conversation_id": parent_id,
        "root_conversation_id": root_id,
        "created_at": created_at,
        "subagent_info": subagent_info,
    }


def step_timestamps(steps: list[Step], created_at: int = 0) -> tuple[int, int, int]:
    start_ts = created_at
    if not start_ts and steps:
        for fn in (1, 6, 7, 8):
            m = steps[0].md.msg(fn)
            if m and m.int(1) > 0:
                start_ts = m.int(1)
                break
    end_ts = start_ts
    for step in reversed(steps):
        found = 0
        for fn in (8, 7, 6, 1):
            m = step.md.msg(fn)
            if m and m.int(1) > 0:
                found = m.int(1)
                break
        if found:
            end_ts = max(start_ts, found)
            break
    duration = max(0, end_ts - start_ts) if start_ts and end_ts else 0
    return start_ts, end_ts, duration


def extract_tool_call(step: Step) -> tuple[str, dict]:
    tc = step.md.msg(4)
    tool = tc.str(2) if tc is not None else ""
    args: dict = {}
    if tc is not None and tc.str(3):
        try:
            parsed = json.loads(tc.str(3))
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            pass
    if not args and step.generic_pb is not None:
        kv = {m.str(1): m.str(2) for m in step.generic_pb.msgs(1) if m.str(1)}
        if kv:
            args = {k: v.strip('"') if isinstance(v, str) else v for k, v in kv.items()}
            if not tool and ("Role" in args or "TypeName" in args):
                tool = "invoke_subagent"
    return tool, args


def summarize_step(step: Step) -> dict | None:
    status_name = STEP_STATUS_NAMES.get(step.status, "done")
    tool, args = extract_tool_call(step)

    if tool in ("read_file", "view_file") or (step.type == 8 and not tool):
        path = next(
            (args[k] for k in ("Path", "AbsolutePath", "FilePath") if isinstance(args.get(k), str) and args[k]),
            "",
        )
        if not path:
            vf = step.payload(STEP_VIEW_FILE)
            path = vf.str(1) if vf is not None else ""
        short = shorten_path(path) if path else "file"
        base = Path(short.rstrip("/")).name or short
        start, end = args.get("StartLine"), args.get("EndLine")
        rng = f" (L{start}-{end})" if start and end else (f" (L{start}+)" if start else "")
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": tool or "view_file",
            "label": f"Read {base}{rng}",
            "detail": short,
        }

    if tool == "run_command" or step.type == 21:
        summary = str(args.get("toolSummary") or "").strip()
        cmd = str(args.get("CommandLine") or "").strip()
        first_cmd = cmd.splitlines()[0].strip() if cmd else ""
        label = f"Run: {truncate_text(summary or first_cmd or 'command', 48)}"
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": "run_command",
            "label": label,
            "detail": truncate_text(first_cmd, 80),
        }

    if tool in ("edit_file", "write_file", "replace_file_content", "multi_replace_file_content") or step.type == 5:
        path = next(
            (args[k] for k in ("Path", "TargetFile", "AbsolutePath") if isinstance(args.get(k), str) and args[k]),
            "",
        )
        if not path:
            ca = step.payload(STEP_CODE_ACTION)
            res = ca.msg(2) if ca is not None else None
            ed = res.msg(1) if res is not None else None
            path = ed.str(8) if ed is not None else ""
        short = shorten_path(path) if path else "file"
        base = Path(short.rstrip("/")).name or short
        verb = "Write" if tool == "write_file" else "Edit"
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": tool or "code_action",
            "label": f"{verb} {base}",
            "detail": short,
        }

    if tool in ("grep_search", "find_by_name") or step.type in (7, 25):
        q = str(args.get("Query") or args.get("Pattern") or "").strip()
        label = f"Search: {truncate_text(q, 42)}" if q else "Search files"
        detail = shorten_path(str(args.get("SearchPath") or args.get("SearchDirectory") or ""))
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": tool or "search",
            "label": label,
            "detail": detail,
        }

    if tool == "send_message":
        msg = str(args.get("Message") or "").strip()
        first_line = next((ln.lstrip("# ").strip() for ln in msg.splitlines() if ln.strip()), "")
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": "send_message",
            "label": "Sent report to parent",
            "detail": truncate_text(first_line, 80),
        }

    if tool == "invoke_subagent" or step.type == 127:
        role = str(args.get("Role") or args.get("TypeName") or "subagent").strip().strip('"')
        tname = str(args.get("TypeName") or "").strip().strip('"')
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": "invoke_subagent",
            "label": f"Spawned: {truncate_text(role, 44)}",
            "detail": tname,
        }

    if tool == "call_mcp_tool" or step.type == 38:
        srv = str(args.get("ServerName") or "").strip()
        tn = str(args.get("ToolName") or "").strip()
        target = f"{srv}/{tn}".strip("/") or "mcp"
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": "call_mcp_tool",
            "label": f"MCP: {target}",
            "detail": "",
        }

    if tool:
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": tool,
            "label": f"Tool: {tool}",
            "detail": "",
        }

    if step.type == 23:
        ck = step.payload(STEP_CHECKPOINT)
        num = (ck.int(1) + 1) if ck is not None else 1
        intent = (ck.str(4) if ck is not None else "").strip()
        return {
            "idx": step.pos,
            "status": status_name,
            "tool": "checkpoint",
            "label": f"Compacted context (Checkpoint #{num})",
            "detail": truncate_text(intent, 72),
        }

    if step.type == 15:
        pr = step.payload(STEP_PLANNER_RESPONSE)
        text = (pr.str(1) or pr.str(8) or "").strip() if pr is not None else ""
        first_line = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.strip()), "")
        if step.status == STATUS_GENERATING:
            return {
                "idx": step.pos,
                "status": "generating",
                "tool": "response",
                "label": "Generating response…",
                "detail": truncate_text(first_line, 72),
            }
        if first_line:
            return {
                "idx": step.pos,
                "status": status_name,
                "tool": "response",
                "label": f"Response: {truncate_text(first_line, 44)}",
                "detail": truncate_text(first_line, 80),
            }
    return None


UUID_RE = re.compile(r'"conversationId"\s*:\s*"([0-9a-fA-F-]{36})"')


def discover_subagents(conv_id: str, steps: list[Step], conv_dir: Path) -> list[dict]:
    found: dict[str, dict] = {}
    order = 0

    for step in steps:
        tool, args = extract_tool_call(step)
        if tool != "invoke_subagent" and step.type != 127:
            continue
        sub_id = ""
        if step.generic_pb is not None:
            res = step.generic_pb.msg(2)
            out_text = res.str(1) if res is not None else ""
            m = UUID_RE.search(out_text)
            if m:
                sub_id = m.group(1)
        if not sub_id:
            continue
        role = str(args.get("Role") or "").strip().strip('"')
        type_name = str(args.get("TypeName") or "").strip().strip('"')
        if sub_id not in found:
            found[sub_id] = {
                "conversation_id": sub_id,
                "parent_conversation_id": conv_id,
                "root_conversation_id": conv_id,
                "nesting_depth": 1,
                "role": role,
                "type_name": type_name,
                "created_at": 0,
                "_order": order,
            }
            order += 1

    if conv_dir.is_dir():
        for other_db in sorted(conv_dir.glob("*.db"), key=lambda p: p.stat().st_mtime):
            sub_id = other_db.stem
            if sub_id == conv_id:
                continue
            tmb, _, _ = load_meta(other_db)
            if tmb is None:
                continue
            p_id = tmb.str(5)
            r_id = tmb.str(6)
            if p_id != conv_id and r_id != conv_id:
                continue
            lineage = parse_trajectory_lineage(tmb, sub_id)
            sinfo = lineage["subagent_info"] or {}
            entry = found.get(sub_id)
            if entry is None:
                found[sub_id] = {
                    "conversation_id": sub_id,
                    "parent_conversation_id": p_id or conv_id,
                    "root_conversation_id": r_id or conv_id,
                    "nesting_depth": sinfo.get("nesting_depth") or 1,
                    "role": sinfo.get("role") or "",
                    "type_name": sinfo.get("type_name") or "",
                    "created_at": lineage["created_at"],
                    "_order": order,
                }
                order += 1
            else:
                if not entry["role"] and sinfo.get("role"):
                    entry["role"] = sinfo["role"]
                if not entry["type_name"] and sinfo.get("type_name"):
                    entry["type_name"] = sinfo["type_name"]
                if sinfo.get("nesting_depth"):
                    entry["nesting_depth"] = sinfo["nesting_depth"]
                if lineage["created_at"]:
                    entry["created_at"] = lineage["created_at"]

    return sorted(found.values(), key=lambda x: (x["created_at"] or float("inf"), x["_order"]))


def collect_subagent_summary(meta: dict, conv_dir: Path, catalog: dict) -> dict:
    sub_id = meta["conversation_id"]
    db_path = conv_dir / f"{sub_id}.db"
    if not db_path.is_file():
        return {
            "conversation_id": sub_id,
            "parent_conversation_id": meta["parent_conversation_id"],
            "nesting_depth": meta["nesting_depth"],
            "role": meta["role"],
            "type_name": meta["type_name"],
            "model": "Unknown Model",
            "state": "starting",
            "termination_reason": "",
            "has_reported": False,
            "step_count": 0,
            "created_at": meta["created_at"],
            "last_active_at": meta["created_at"],
            "duration_seconds": 0,
            "used_tokens": 0,
            "budget_used_tokens": 0,
            "limit_tokens": 0,
            "used_pct": 0.0,
            "request_budget_tokens": 0,
            "budget_pct": 0.0,
            "checkpoint_count": 0,
            "compacted": False,
            "cache_stats": empty_cache_stats(),
            "output_stats": empty_output_stats(),
            "current_action": "Initializing…",
            "recent_steps": [],
        }

    steps, gen = load(db_path)
    tmb, term_reason_code, term_step_idx = load_meta(db_path)
    lineage = parse_trajectory_lineage(tmb, sub_id)
    sinfo = lineage["subagent_info"] or {}
    role = meta["role"] or sinfo.get("role") or ""
    type_name = meta["type_name"] or sinfo.get("type_name") or ""
    nesting_depth = sinfo.get("nesting_depth") or meta["nesting_depth"] or 1

    m_id = model_id(gen)
    cat_entry = catalog.get(m_id) or {}
    limit = int(cat_entry.get("maxTokens") or 0) or context_limit(gen)
    model = cat_entry.get("displayName") or model_name(gen)
    st = collect_stats(steps, gen, limit)
    cache_stats = collect_cache_stats(steps, gen)
    output_stats = collect_output_stats(steps, gen)
    req_budget = request_budget(gen) or (256_000 if limit > 256_000 else limit)
    budget_used = effective_budget_tokens(steps, gen, st)
    budget_pct = round(pct(budget_used, req_budget), 1) if req_budget else 0.0

    start_ts, end_ts, duration = step_timestamps(steps, lineage["created_at"] or meta["created_at"])

    summaries = []
    has_reported = False
    for s in steps:
        summary = summarize_step(s)
        if summary is not None:
            summaries.append(summary)
            if summary["tool"] == "send_message" and s.status == STATUS_DONE:
                has_reported = True

    last_step = steps[-1] if steps else None
    term_reason = TERM_REASON_NAMES.get(term_reason_code, "")
    executor_active = bool(steps and (term_step_idx < 0 or term_step_idx < len(steps) - 1))

    if last_step is not None and last_step.status == STATUS_GENERATING:
        state = "generating"
    elif last_step is not None and last_step.status in (STATUS_PENDING, STATUS_RUNNING, STATUS_WAITING):
        state = "running"
    elif last_step is not None and last_step.status in (STATUS_ERROR, STATUS_INVALID):
        state = "error"
    elif last_step is not None and last_step.status in (STATUS_CANCELED, STATUS_INTERRUPTED, STATUS_HALTED):
        state = "canceled"
    elif term_reason_code == 1:
        state = "error"
    elif has_reported:
        state = "reported"
    elif term_reason_code == 2:
        state = "canceled"
    elif executor_active:
        state = "running"
    elif not steps:
        state = "starting"
    else:
        state = "idle"

    if state == "generating":
        current_action = "Generating response…"
    elif summaries:
        if (
            len(summaries) >= 2
            and summaries[-1]["tool"] == "response"
            and summaries[-2]["tool"] == "send_message"
        ):
            current_action = summaries[-2]["label"]
        else:
            current_action = summaries[-1]["label"]
    elif not steps:
        current_action = "Initializing…"
    else:
        current_action = "Idle"

    return {
        "conversation_id": sub_id,
        "parent_conversation_id": lineage["parent_conversation_id"] or meta["parent_conversation_id"],
        "nesting_depth": nesting_depth,
        "role": role,
        "type_name": type_name,
        "model": model,
        "state": state,
        "termination_reason": term_reason,
        "has_reported": has_reported,
        "step_count": len(steps),
        "created_at": start_ts,
        "last_active_at": end_ts,
        "duration_seconds": duration,
        "used_tokens": st["total"],
        "budget_used_tokens": budget_used,
        "limit_tokens": limit,
        "used_pct": round(min(100.0, pct(st["total"], limit)), 1),
        "request_budget_tokens": req_budget,
        "budget_pct": budget_pct,
        "checkpoint_count": len(st["checkpoints"]),
        "compacted": st["compacted"],
        "cache_stats": cache_stats,
        "output_stats": output_stats,
        "current_action": current_action,
        "recent_steps": summaries[-4:],
    }


ACTIVE_SUBAGENT_STATES = ("running", "generating", "starting")


def _subagent_sort_key(item: tuple[int, dict]) -> tuple[int, int, int, int]:
    idx, s = item
    active = s.get("state") in ACTIVE_SUBAGENT_STATES
    primary_ts = (s.get("created_at") or 0) if active else (s.get("last_active_at") or s.get("created_at") or 0)
    return (
        0 if active else 1,
        -primary_ts,
        -(s.get("created_at") or 0),
        -idx,
    )


def sort_subagents(subagents: list[dict]) -> list[dict]:
    return [s for _, s in sorted(enumerate(subagents), key=_subagent_sort_key)]


# Native context windows from the model catalog, which the CLI fetches from the server
# (ModelInfo.max_tokens) and never writes to disk. Matched by model ID prefix; the first
# match wins. Add a row when a new model family ships.
NATIVE_CONTEXT_WINDOWS = [
    ("claude-opus-4-6", 250_000),
    ("claude-sonnet-4-6", 250_000),
    ("claude-", 1_000_000),
    ("gpt-oss-", 131_072),
    ("gemini-", 1_048_576),
]

EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")
WORD_LABELS = {"gpt": "GPT", "oss": "OSS"}


def model_id(gen: Msg | None) -> str:
    planner = gen.msg(3) if gen is not None else None
    return planner.str(28) if planner is not None else ""  # planner_config.model_name


def model_label(slug: str) -> str:
    """Catalog-style label from a model ID: gemini-3.8-flash-medium -> Gemini 3.8 Flash (Medium)."""
    parts = slug.split("-")
    effort = parts.pop().capitalize() if len(parts) > 1 and parts[-1] in EFFORT_LEVELS else ""
    words: list[str] = []
    for p in parts:
        if p.isdigit() and words and words[-1][-1:].isdigit():
            words[-1] += "." + p  # claude-opus-5-5 -> 5.5
        elif p in WORD_LABELS:
            words.append(WORD_LABELS[p])
        elif p[:1].isdigit():
            words.append(p.upper())  # 120b -> 120B
        else:
            words.append(p.capitalize())
    label = " ".join(words).replace("GPT OSS", "GPT-OSS")
    return f"{label} ({effort})" if effort else label


def model_name(gen: Msg | None) -> str:
    chat = gen.msg(1) if gen is not None else None
    if chat is not None and chat.str(21):  # model_display_name
        return chat.str(21)
    if model_id(gen):
        return model_label(model_id(gen))
    if chat is not None and chat.str(19):  # response_model
        return chat.str(19)
    return "Unknown Model"


def context_limit(gen: Msg | None) -> int:
    """The model's native window, like the CLI; else the request budget (max_context_tokens)."""
    slug = model_id(gen)
    for prefix, tokens in NATIVE_CONTEXT_WINDOWS:
        if slug.startswith(prefix):
            return tokens
    return request_budget(gen)


def request_budget(gen: Msg | None) -> int:
    chat = gen.msg(1) if gen is not None else None
    start = chat.msg(9) if chat is not None else None
    cwm = start.msg(10) if start is not None else None
    return cwm.int(4, signed32=True) if cwm is not None else 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Analyze Antigravity 2.0 conversation context usage.")
    parser.add_argument("--conversation-id", help="Conversation UUID (auto-detected if omitted)")
    parser.add_argument("--db", help="Path to a conversation .db file (e.g. one from ~/.gemini/antigravity-cli/conversations)")
    parser.add_argument("--workspace", help="Active workspace path", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Context window token limit override")
    parser.add_argument("--app-dir", default="antigravity", help="App data folder under ~/.gemini (default: antigravity)")
    parser.add_argument("--width", type=int, default=100, help="Terminal width used for the grid layout")
    parser.add_argument("--color", action="store_true", help="Color the grid segments with ANSI escapes")
    parser.add_argument("--json", action="store_true", help="Output JSON data")
    parser.add_argument("--model-catalog", help='JSON file of {"<model id>": {"maxTokens": n, "displayName": "..."}} '
                        "from the app's live model catalog; overrides the built-in limits and labels")
    args = parser.parse_args()

    if args.db:
        db_path = Path(args.db).expanduser()
        conv_id = db_path.stem
    else:
        gemini_env = os.environ.get("ANTIGRAVITY_GEMINI_DIR")
        gemini_dir = Path(gemini_env).expanduser() if gemini_env else Path.home() / ".gemini"
        conv_dir = gemini_dir / args.app_dir / "conversations"
        conv_id = args.conversation_id
        if not conv_id:
            dbs = sorted(conv_dir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not dbs:
                raise SystemExit("No conversation databases found.")
            conv_id = dbs[0].stem
        db_path = conv_dir / f"{conv_id}.db"
    if not db_path.is_file():
        raise SystemExit(f"Conversation database not found: {db_path}")

    steps, gen = load(db_path)
    tmb, _, _ = load_meta(db_path)
    lineage = parse_trajectory_lineage(tmb, conv_id)

    catalog = {}
    if args.model_catalog:
        with open(args.model_catalog, encoding="utf-8") as fh:
            catalog = json.load(fh)
    catalog_entry = catalog.get(model_id(gen)) or {}
    limit = args.limit if args.limit > 0 else int(catalog_entry.get("maxTokens") or 0) or context_limit(gen)
    st = collect_stats(steps, gen, limit)
    cache_stats = collect_cache_stats(steps, gen)
    output_stats = collect_output_stats(steps, gen)
    model = catalog_entry.get("displayName") or model_name(gen)
    system_files = detect_system_files(os.path.expanduser("~"), args.workspace, args.app_dir)

    subagent_metas = discover_subagents(conv_id, steps, db_path.parent)
    subagents = sort_subagents([collect_subagent_summary(m, db_path.parent, catalog) for m in subagent_metas])

    if args.json:
        req_budget = request_budget(gen)
        budget_used = effective_budget_tokens(steps, gen, st)
        print(json.dumps({
            "conversation_id": conv_id,
            "parent_conversation_id": lineage["parent_conversation_id"],
            "root_conversation_id": lineage["root_conversation_id"],
            "subagent_info": lineage["subagent_info"],
            "model": model,
            "heading": stats_heading(st),
            "grid_segments": grid_segments(st),
            "checkpoint_view": checkpoint_view(st),
            "text": render(st, model, system_files, args.width, False, subagents),
            "model_id": model_id(gen),
            "request_budget_tokens": req_budget,
            "budget_used_tokens": budget_used,
            "budget_pct": round(pct(budget_used, req_budget), 1) if req_budget else 0.0,
            "used_tokens": st["total"],
            "limit_tokens": limit,
            "used_pct": round(min(100.0, pct(st["total"], limit)), 1),
            "free_tokens": limit - st["total"] if st["total"] < limit else 0,
            "has_breakdown": st["has_breakdown"],
            "compacted": st["compacted"],
            "checkpoint_buffer": st["checkpoint"],
            "cache_stats": cache_stats,
            "output_stats": output_stats,
            "categories": [
                {"icon": icon, "name": name, "key": key, "tokens": st[key], "pct": round(pct(st[key], limit), 1)}
                for icon, name, key, always in CATEGORIES
                if always or st[key]
            ],
            "active_checkpoint": st["active_checkpoint"],
            "checkpoints": st["checkpoints"],
            "knowledge_files": [{"path": p, "tokens": t} for p, t in st["knowledge_files"]],
            "artifact_files": [{"path": p, "tokens": t} for p, t in st["artifact_files"]],
            "skill_files": [{"path": p, "tokens": t} for p, t in st["skill_files"]],
            "system_files": [{"path": p} for p, _ in system_files],
            "subagents": subagents,
        }, indent=2))
        return

    print(render(st, model, system_files, args.width, args.color, subagents))


if __name__ == "__main__":
    main()
