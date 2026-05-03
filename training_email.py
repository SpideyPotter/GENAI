#!/usr/bin/env python3
"""
Email periodic snapshots of the QA synthesis log (synthesis.py → QAdata/qa_synthesis.log).

Config (first file found wins):
  $TRAINING_EMAIL_CONFIG (e.g. export TRAINING_EMAIL_CONFIG=.training_email_config),
  else ~/.training_email_config, else <repo>/.training_email_config
  (KEY=value per line, # comments ok). Relative paths use the process cwd.

  SENDER_EMAIL=you@gmail.com
  SENDER_PASSWORD=app-password
  RECIPIENT_EMAIL=you@gmail.com

Optional:
  SYNTHESIS_LOG=/path/to/qa_synthesis.log   # preferred
  TRAINING_LOG=...                          # legacy alias for SYNTHESIS_LOG
  PROCESS_MATCH=synthesis.py                # pgrep filter (default: synthesis.py)
  TRAIN_MATCH=...                           # legacy alias for PROCESS_MATCH
  INTERVAL_SECONDS=3600                     # daemon: seconds between emails (default 3600)

Usage:
  python training_email.py              # one email now
  python training_email.py once
  python training_email.py daemon       # repeat every INTERVAL_SECONDS

Cron (hourly):
  0 * * * * cd /path/to/repo && /path/to/python training_email.py once
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import smtplib


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_config_path() -> Path:
    env = os.environ.get('TRAINING_EMAIL_CONFIG', '').strip()
    if env:
        return Path(env).expanduser()
    home_cfg = Path.home() / '.training_email_config'
    if home_cfg.is_file():
        return home_cfg
    return _repo_root() / '.training_email_config'


def load_config(path: Path | None = None) -> dict:
    p = path or _default_config_path()
    if not p.is_file():
        sys.stderr.write(
            f'Missing email config. Tried TRAINING_EMAIL_CONFIG, '
            f'{Path.home() / ".training_email_config"}, '
            f'{_repo_root() / ".training_email_config"}\n'
            'Create one with SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL '
            '(optional: SYNTHESIS_LOG, PROCESS_MATCH, INTERVAL_SECONDS).\n')
        sys.exit(1)
    cfg: dict[str, str] = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    for key in ('SENDER_EMAIL', 'SENDER_PASSWORD', 'RECIPIENT_EMAIL'):
        if key not in cfg:
            sys.stderr.write(f'Config missing required key: {key}\n')
            sys.exit(1)

    style = (cfg.get('REPORT_STYLE') or cfg.get('TRAIN_TYPE') or '').strip().lower()
    if style in ('mmdet', 'mmseg', 'train'):
        sys.stderr.write(
            'training_email.py only sends QA synthesis log reports now; '
            f'ignoring REPORT_STYLE={style!r}.\n',
        )
    return cfg


def send_email(cfg: dict, subject: str, body: str) -> bool:
    msg = MIMEMultipart()
    msg['From'] = cfg['SENDER_EMAIL']
    msg['To'] = cfg['RECIPIENT_EMAIL']
    msg['Subject'] = f'[QA Synthesis] {subject}'
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(cfg['SENDER_EMAIL'], cfg['SENDER_PASSWORD'])
            server.send_message(msg)
        print(f'Email sent: {subject}', flush=True)
        return True
    except Exception as e:
        print(f'Email failed: {e}', file=sys.stderr, flush=True)
        return False


def read_tail(path: Path, max_bytes: int = 384_000) -> str:
    if not path.is_file():
        return ''
    size = path.stat().st_size
    with open(path, 'rb') as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        data = f.read().decode('utf-8', errors='replace')
    return data


CRASH_PATTERNS = re.compile(
    r'(Traceback \(most recent call last\):|'
    r'FileNotFoundError:|CUDA out of memory|OutOfMemoryError|'
    r'RuntimeError:|AssertionError:|KeyError:|ValueError:|'
    r'nan |loss is nan|Killed|SIGKILL|Segmentation fault)',
    re.IGNORECASE,
)


def detect_crash_snippets(text: str, context: int = 4) -> list[str]:
    lines = text.splitlines()
    hits = []
    for i, line in enumerate(lines):
        if CRASH_PATTERNS.search(line):
            start = max(0, i - context)
            end = min(len(lines), i + context + 12)
            block = '\n'.join(lines[start:end])
            hits.append(block)
    return hits[-3:]


def synthesis_process_status(match: str) -> str:
    try:
        r = subprocess.run(
            ['pgrep', '-af', 'python'],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 'Could not run pgrep.'
    if r.returncode != 0 and not r.stdout:
        return 'No python processes matched pgrep.'
    lines = [ln for ln in r.stdout.strip().splitlines() if match in ln]
    if not lines:
        return (
            f'No process matching "{match}" in pgrep -af python.\n'
            'Synthesis may be finished or PROCESS_MATCH / TRAIN_MATCH needs adjusting.'
        )
    return 'Matching process(es):\n' + '\n'.join(lines[:8])


def gpu_snapshot() -> str:
    try:
        r = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader',
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return '(nvidia-smi unavailable)'
        return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return '(nvidia-smi unavailable)'


def gpu_snapshot_full(max_lines: int = 80) -> str:
    try:
        r = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=12,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or '').strip()
            return f'(nvidia-smi failed: {err[:500]})' if err else '(nvidia-smi failed)'
        lines = r.stdout.splitlines()
        if len(lines) > max_lines:
            head = lines[: max_lines - 2]
            return '\n'.join(head + ['...', f'({len(lines) - max_lines + 2} more lines truncated)'])
        return r.stdout.rstrip() or '(empty nvidia-smi output)'
    except FileNotFoundError:
        return '(nvidia-smi not installed)'
    except subprocess.TimeoutExpired:
        return '(nvidia-smi timed out)'


def _synthesis_log_path(cfg: dict) -> Path:
    raw = (cfg.get('SYNTHESIS_LOG') or cfg.get('TRAINING_LOG') or '').strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_repo_root() / 'QAdata' / 'qa_synthesis.log').resolve()


def _process_match(cfg: dict) -> str:
    return (cfg.get('PROCESS_MATCH') or cfg.get('TRAIN_MATCH') or 'synthesis.py').strip()


def extract_synthesis_log_highlights(text: str, *, max_lines: int = 80) -> list[str]:
    """Lines that matter for synthesis.py / qa_synthesis.log (INFO/WARN/ERROR, tqdm, milestones)."""
    keys = (
        '[ERROR]', '[WARNING]', '[CRITICAL]',
        'ERROR', 'WARNING', 'Traceback',
        'QA Synthesis', 'Inference settings', 'dtype=', 'batch_size',
        'Resuming', 'checkpoint', 'Skipping', 'Nothing left to generate',
        'Loading model', 'Model loaded', 'Found ', 'chunk files', 'Total chunks loaded',
        'Generating QA', 'It/s', 's/it', 'Inference complete', 'Pipeline finished',
        'Saved ', 'QA pairs', 'Stats:', 'OOM', 'OutOfMemory', 'transformers',
        'progress:', 'new chunks/min',
    )
    lines = text.splitlines()
    picked: list[str] = []
    for line in lines:
        if any(k in line for k in keys):
            picked.append(line)
    if picked:
        return picked[-max_lines:]
    return lines[-min(max_lines, len(lines)) :]


def build_synthesis_report(cfg: dict) -> tuple[str, str]:
    log_path = _synthesis_log_path(cfg)
    match = _process_match(cfg)

    parts: list[str] = []
    parts.append(f'Time (host): {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    parts.append(f'SYNTHESIS_LOG: {log_path}')
    parts.append('')

    parts.append('=== synthesis.py process ===')
    parts.append(synthesis_process_status(match))
    parts.append('')

    parts.append('=== nvidia-smi (CSV) ===')
    parts.append(gpu_snapshot())
    parts.append('')
    parts.append('=== nvidia-smi (full) ===')
    parts.append(gpu_snapshot_full(100))
    parts.append('')

    tail = read_tail(log_path) if log_path.is_file() else ''
    if tail:
        parts.append(f'=== Log highlights (tail of {log_path.name}) ===')
        highlights = extract_synthesis_log_highlights(tail, max_lines=90)
        parts.extend(highlights)
        parts.append('')
        parts.append(f'=== Raw tail (last ~28 KiB of {log_path.name}) ===')
        tail_snip = tail[-28_672:] if len(tail) > 28_672 else tail
        parts.append(tail_snip)
        crash_blocks = detect_crash_snippets(tail)
    else:
        parts.append(f'=== Synthesis log ===\nFile missing or empty: {log_path}\n')
        crash_blocks = []

    parts.append('')
    parts.append('=== Crash / error scan (log tail) ===')
    if crash_blocks:
        parts.append('ALERT: Possible failure signatures in recent log tail.')
        for b in crash_blocks:
            parts.append('---')
            parts.append(b)
        subj = 'HOURLY — crash suspected (check log)'
    else:
        parts.append('No Traceback / OOM / FileNotFoundError patterns in tail window.')
        subj = 'Hourly QA synthesis log'

    body = '\n'.join(parts)
    return subj, body


def main():
    # Legacy: python training_email.py "Subject line" "Plain body"
    if len(sys.argv) == 3 and sys.argv[1] not in (
            'once', 'daemon', '-h', '--help') and not sys.argv[1].startswith('-'):
        cfg = load_config()
        send_email(cfg, sys.argv[1], sys.argv[2])
        return

    parser = argparse.ArgumentParser(
        description='Email QA synthesis status (tail of qa_synthesis.log + GPU snapshot).',
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='once',
        choices=('once', 'daemon'),
        help='once: send one report; daemon: repeat every INTERVAL_SECONDS',
    )
    parser.add_argument('--config', type=Path, default=None, help='Override config path')
    parser.add_argument('--subject', default=None, help='Override email subject tag')
    parser.add_argument('--body', default=None, help='Send only this body (no auto report)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.body is not None:
        subj = args.subject or 'Manual notice'
        send_email(cfg, subj, args.body)
        return

    interval = int(cfg.get('INTERVAL_SECONDS', '3600'))

    def one_shot():
        subj, body = build_synthesis_report(cfg)
        if args.subject:
            subj = args.subject
        send_email(cfg, subj, body)

    if args.mode == 'once':
        one_shot()
        return

    print(f'Daemon: sending every {interval}s. Ctrl+C to stop.', flush=True)
    while True:
        one_shot()
        time.sleep(interval)


if __name__ == '__main__':
    main()
