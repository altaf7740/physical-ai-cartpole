"""MCP access tokens: named, scoped, optionally expiring; created and revoked from the web UI.

Only a SHA-256 of each token is stored (``[mcp] tokens_file``, mode 0600, not in git, kept by ``make
deploy``), so a token is shown once, when it is created. Same model as I-Vision's ``McpToken`` rows, in a
JSON file instead of a database.

A token from the earlier single-token setup (``mcp_token`` next to the tokens file) is imported once as
"default" with the operate scope, so an agent already set up with it keeps working.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from mcp_tools import SCOPES

TOKEN_PREFIX = "cpmcp_"
LEGACY_FILE = "mcp_token"
_TOUCH_EVERY_S = 60.0  # how often last_used_at is written back


class TokenError(ValueError):
    pass


@dataclass
class Token:
    id: str
    name: str
    prefix: str  # the first characters, to tell tokens apart in the UI
    scope: str
    token_hash: str
    created_at: float
    expires_at: float | None = None
    last_used_at: float | None = None

    def public(self) -> dict:
        data = asdict(self)
        del data["token_hash"]
        data["expired"] = self.expires_at is not None and self.expires_at <= time.time()
        return data


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._tokens: list[Token] = []
        self._saved_at = 0.0
        if self.path.exists():
            raw = json.loads(self.path.read_text() or "[]")
            self._tokens = [Token(**t) for t in raw]
        self._import_legacy(self.path.with_name(LEGACY_FILE))

    def _import_legacy(self, legacy: Path) -> None:
        if not legacy.exists():
            return
        token = legacy.read_text().strip()
        if token.startswith(TOKEN_PREFIX) and not any(t.token_hash == _hash(token) for t in self._tokens):
            self._tokens.append(self._row("default", "operate", token, None, legacy.stat().st_mtime))
        self._save()
        legacy.unlink()

    @staticmethod
    def _row(name: str, scope: str, token: str, expires_at: float | None, created_at: float) -> Token:
        return Token(id=secrets.token_hex(4), name=name, prefix=token[: len(TOKEN_PREFIX) + 6], scope=scope,
                     token_hash=_hash(token), created_at=created_at, expires_at=expires_at)

    def _save(self) -> None:
        """Called with the lock held (or before the store is shared). Atomic, private."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump([asdict(t) for t in self._tokens], f, indent=1)
        os.replace(tmp, self.path)
        self._saved_at = time.monotonic()

    # ---- management (the web UI) ---------------------------------------------------------------------

    def create(self, name: str, scope: str, expires_in_days: float | None = None) -> tuple[Token, str]:
        """A new token. Returns the row and the token itself, which is not stored and cannot be shown again."""
        name = (name or "").strip()
        if not 1 <= len(name) <= 60:
            raise TokenError("name must be 1-60 characters")
        if scope not in SCOPES:
            raise TokenError(f"scope must be one of {SCOPES}")
        if expires_in_days is not None and not 0 < expires_in_days <= 3650:
            raise TokenError("expires_in_days must be 1-3650 (or empty for never)")
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        now = time.time()
        row = self._row(name, scope, token, now + expires_in_days * 86400 if expires_in_days else None, now)
        with self._lock:
            if any(t.name == name for t in self._tokens):
                raise TokenError(f"a token named {name!r} already exists")
            self._tokens.append(row)
            self._save()
        return row, token

    def revoke(self, token_id: str) -> Token:
        with self._lock:
            row = next((t for t in self._tokens if t.id == token_id), None)
            if row is None:
                raise TokenError("token not found")
            self._tokens.remove(row)
            self._save()
            return row

    def list(self) -> list[dict]:
        with self._lock:
            return [t.public() for t in sorted(self._tokens, key=lambda t: -t.created_at)]

    # ---- the MCP endpoint ----------------------------------------------------------------------------

    def authenticate(self, authorization: str | None) -> Token | None:
        """The token an ``Authorization: Bearer cpmcp_...`` header carries, if it is valid and not expired."""
        scheme, _, token = (authorization or "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token.startswith(TOKEN_PREFIX):
            return None
        digest, now = _hash(token), time.time()
        with self._lock:
            row = next((t for t in self._tokens if hmac.compare_digest(t.token_hash, digest)), None)
            if row is None or (row.expires_at is not None and row.expires_at <= now):
                return None
            row.last_used_at = now
            if time.monotonic() - self._saved_at > _TOUCH_EVERY_S:
                try:
                    self._save()
                except OSError:
                    pass
            return row
