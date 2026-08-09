"""Security validation of scripts, arguments, and stdin.

The validator applies a fixed rule set: a dangerous-command substring table,
compiled dangerous patterns, argument-injection patterns, network-access and
reverse-shell signatures for scripts, and embedded-shell-command detection for
stdin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ValidationFailure:
    """One structured security validation failure."""

    error_type: str  # dangerous_command / dangerous_pattern / shell_injection / ...
    pattern: str  # the pattern that matched
    context: str  # where it was found
    message: str  # human-readable description


@dataclass(frozen=True)
class ValidationResult:
    """All validation errors found by one validation call."""

    valid: bool
    errors: list[ValidationFailure] = field(default_factory=list)


#: Command-substitution patterns, compiled once. They run per argument in
#: ``validate_args``, so recompiling on every call would waste work.
_COMMAND_SUBSTITUTION_PATTERNS = (
    re.compile(r"\$\([^)]+\)"),  # $(command)
    re.compile(r"`[^`]+`"),  # `command`
    re.compile(r"\$\{[^}]*\$\([^)]*\)"),  # ${...$(command)
)

_NETWORK_ACCESS_PATTERNS = (
    r"\bcurl\b",
    r"\bwget\b",
    r"\bnc\b",
    r"\bnetcat\b",
    r"\btelnet\b",
    r"\bssh\b",
    r"\bscp\b",
    r"\brsync\b",
    r"\bftp\b",
    r"\bsftp\b",
    r"socket\.connect",
    r"urllib\.request",
    r"requests\.get",
    r"requests\.post",
    r"http\.client",
    r"httplib",
    r"fetch\s*\(",
    r"axios",
    r"XMLHttpRequest",
)
_NETWORK_ACCESS_RE = [re.compile(f"(?i){pattern}") for pattern in _NETWORK_ACCESS_PATTERNS]

_REVERSE_SHELL_PATTERNS = (
    r"/dev/tcp/",
    r"/dev/udp/",
    r"bash\s+-i",
    r"sh\s+-i",
    r"/bin/bash\s+-i",
    r"/bin/sh\s+-i",
    r"python.*pty\.spawn",
    r"perl.*-e.*socket",
    r"ruby.*-rsocket",
    r"socat.*exec",
    r"mkfifo",
    r"mknod.*p",
    r"0<&196",  # File descriptor redirection trick
    r"196>&0",
    r"/inet/tcp/",
    r"bash.*>&.*0>&1",
    r"nc.*-e",
    r"ncat.*-e",
    r"netcat.*-e",
)
_REVERSE_SHELL_RE = [re.compile(f"(?i){pattern}") for pattern in _REVERSE_SHELL_PATTERNS]

#: Stdin patterns stay case-sensitive to avoid over-blocking ordinary text.
_EMBEDDED_SHELL_PATTERNS = (
    re.compile(r"\$\(.*\)"),  # Command substitution
    re.compile(r"`.*`"),  # Backtick substitution
    re.compile(r"\n\s*[;&|]"),  # Newline followed by shell operators
    re.compile(r"\\n.*[;&|]"),  # Escaped newline followed by shell operators
)

_SHELL_OPERATORS = (
    "&&",
    "||",
    ";",
    "|",
    "\n",
    "\r",
    "$(",
    "`",
    ">",
    "<",
    ">>",
    "2>",
    "&>",
)

_ARG_INJECTION_PATTERNS = (
    r"\.\./",
    r"\.\.\\",
    r"\$\{[A-Z_]+\}",
    r"\$[A-Z_]+",
    r"\$\(",
    r"`",
    r"\n",
    r"\r",
)


def _compile_patterns(patterns: tuple[str, ...]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        try:
            compiled.append(re.compile(f"(?i){pattern}"))
        except re.error:
            continue
    return compiled


class ScriptValidator:
    """Validates scripts, arguments, and stdin for security."""

    def __init__(self) -> None:
        self._dangerous_commands = _default_dangerous_commands()
        self._dangerous_patterns = _compile_patterns(_default_dangerous_patterns())
        self._arg_injection_patterns = _compile_patterns(_ARG_INJECTION_PATTERNS)

    def validate_script(self, content: str) -> ValidationResult:
        """Validate script content for dangerous commands and patterns."""
        errors: list[ValidationFailure] = []
        for cmd in self._dangerous_commands:
            if cmd in content:
                errors.append(
                    ValidationFailure(
                        error_type="dangerous_command",
                        pattern=cmd,
                        context=extract_context(content, cmd),
                        message=f"Script contains dangerous command: {cmd}",
                    )
                )
        lower = content.lower()
        for pattern in self._dangerous_patterns:
            match = pattern.search(lower)
            if match is not None:
                errors.append(
                    ValidationFailure(
                        error_type="dangerous_pattern",
                        pattern=pattern.pattern,
                        context=extract_context(content, match.group(0)),
                        message=f"Script contains dangerous pattern: {match.group(0)}",
                    )
                )
        if has_network_access(content):
            errors.append(
                ValidationFailure(
                    error_type="network_access",
                    pattern="network commands",
                    context="script content",
                    message="Script attempts to access network resources",
                )
            )
        if has_reverse_shell_pattern(content):
            errors.append(
                ValidationFailure(
                    error_type="reverse_shell",
                    pattern="reverse shell pattern",
                    context="script content",
                    message="Script contains potential reverse shell pattern",
                )
            )
        return ValidationResult(valid=not errors, errors=errors)

    def validate_args(self, args: list[str]) -> ValidationResult:
        """Validate command-line arguments for injection attempts."""
        errors: list[ValidationFailure] = []
        for i, arg in enumerate(args):
            context = f"arg[{i}]: {truncate(arg, 50)}"
            if has_shell_operators(arg):
                errors.append(
                    ValidationFailure(
                        error_type="shell_injection",
                        pattern="shell operators",
                        context=context,
                        message="Argument contains shell command operators",
                    )
                )
            if has_command_substitution(arg):
                errors.append(
                    ValidationFailure(
                        error_type="command_substitution",
                        pattern="command substitution",
                        context=context,
                        message="Argument contains command substitution syntax",
                    )
                )
            for pattern in self._arg_injection_patterns:
                if pattern.search(arg) is not None:
                    errors.append(
                        ValidationFailure(
                            error_type="arg_injection",
                            pattern=pattern.pattern,
                            context=context,
                            message="Argument matches injection pattern",
                        )
                    )
        return ValidationResult(valid=not errors, errors=errors)

    def validate_stdin(self, stdin: str) -> ValidationResult:
        """Validate stdin content for embedded shell commands."""
        errors: list[ValidationFailure] = []
        if has_embedded_shell_commands(stdin):
            errors.append(
                ValidationFailure(
                    error_type="stdin_injection",
                    pattern="embedded shell commands",
                    context=truncate(stdin, 100),
                    message="Stdin contains embedded shell command patterns",
                )
            )
        return ValidationResult(valid=not errors, errors=errors)

    def validate_all(self, script_content: str, args: list[str], stdin: str) -> ValidationResult:
        """Perform comprehensive validation over script, args, and stdin."""
        errors: list[ValidationFailure] = []
        if not self.validate_script(script_content).valid:
            errors.extend(self.validate_script(script_content).errors)
        if not self.validate_args(args).valid:
            errors.extend(self.validate_args(args).errors)
        if stdin != "" and not self.validate_stdin(stdin).valid:
            errors.extend(self.validate_stdin(stdin).errors)
        return ValidationResult(valid=not errors, errors=errors)


def has_shell_operators(value: str) -> bool:
    """Return whether ``value`` contains a shell command-chaining operator."""
    return any(op in value for op in _SHELL_OPERATORS)


def has_command_substitution(value: str) -> bool:
    """Return whether ``value`` contains command-substitution syntax."""
    return any(pattern.search(value) is not None for pattern in _COMMAND_SUBSTITUTION_PATTERNS)


def has_network_access(content: str) -> bool:
    """Return whether ``content`` references network-access tools or libraries."""
    return any(pattern.search(content) is not None for pattern in _NETWORK_ACCESS_RE)


def has_reverse_shell_pattern(content: str) -> bool:
    """Return whether ``content`` contains a common reverse-shell signature."""
    return any(pattern.search(content) is not None for pattern in _REVERSE_SHELL_RE)


def has_embedded_shell_commands(content: str) -> bool:
    """Return whether ``content`` embeds shell command substitution."""
    return any(pattern.search(content) is not None for pattern in _EMBEDDED_SHELL_PATTERNS)


def _default_dangerous_commands() -> tuple[str, ...]:
    """Return commands that must never appear inside a script."""
    return (
        # System modification - various forms of dangerous rm
        "rm -rf /",
        "rm -fr /",
        "rm -rf/*",
        "rm -rf *",
        # Filesystem destruction
        "mkfs",
        "dd if=/dev/zero",
        "dd if=/dev/random",
        # Fork bombs (various forms)
        ":(){ :|:& };:",
        ":(){:|:&};:",
        "bomb(){ bomb|bomb& };bomb",
        # Process and system control
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init 0",
        "init 6",
        "killall",
        "pkill",
        # Permission escalation
        "chmod 777 /",
        "chown root",
        "setuid",
        "setgid",
        "passwd",
        # Credential access
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        ".ssh/",
        "id_rsa",
        "id_ed25519",
        # Environment manipulation
        "export PATH=",
        "export LD_PRELOAD",
        "export LD_LIBRARY_PATH",
        # Cron manipulation
        "crontab",
        "/etc/cron",
        # Service manipulation
        "systemctl",
        "service",
        # Module/kernel manipulation
        "insmod",
        "modprobe",
        "rmmod",
        # Container escape attempts
        "docker",
        "kubectl",
        "nsenter",
        "unshare",
        "capsh",
    )


def _default_dangerous_patterns() -> tuple[str, ...]:
    """Return regex patterns indicating dangerous operations."""
    return (
        # Base64 encoded payloads (often used to hide malicious code)
        r"base64\s+(-d|--decode)",
        r"echo\s+.*\|\s*base64\s+-d",
        # Hex encoded payloads
        r"xxd\s+-r",
        r"echo\s+-e\s+.*\\x",
        # Code download and execution
        r"curl.*\|\s*(bash|sh)",
        r"wget.*\|\s*(bash|sh)",
        r"python.*http\.server",
        # Eval and exec patterns (code injection)
        r"eval\s*\(",
        r"exec\s*\(",
        r"os\.system\s*\(",
        r"subprocess\.call\s*\(.*shell\s*=\s*True",
        r"subprocess\.Popen\s*\(.*shell\s*=\s*True",
        r"os\.popen\s*\(",
        r"commands\.getoutput\s*\(",
        r"commands\.getstatusoutput\s*\(",
        # History/log manipulation
        r"history\s+-c",
        r"unset\s+HISTFILE",
        r"export\s+HISTSIZE=0",
        # Python dangerous functions
        r"__import__\s*\(",
        r"importlib\.import_module",
        r"compile\s*\(.*exec",
        # Pickle deserialization (can execute arbitrary code)
        r"pickle\.loads?\s*\(",
        r"cPickle\.loads?\s*\(",
        # YAML unsafe loading
        r"yaml\.load\s*\([^,]+\)",  # Without Loader argument
        r"yaml\.unsafe_load",
        # Fork bomb patterns (function recursion with backgrounding)
        r":\s*\(\s*\)\s*\{\s*:",  # :() { : pattern
        r"\(\)\s*\{\s*\w+\s*\|\s*\w+\s*&",  # () { x | x & pattern
        # Dangerous rm patterns
        r"rm\s+-[rf]+\s+/",  # rm -rf / or rm -fr /
        r"rm\s+--no-preserve-root",
    )


def extract_context(content: str, match: str) -> str:
    """Extract up to 20 characters of context around a match."""
    idx = content.lower().find(match.lower())
    if idx == -1:
        return ""
    start = idx - 20
    if start < 0:
        start = 0
    end = idx + len(match) + 20
    if end > len(content):
        end = len(content)
    context = content[start:end]
    if start > 0:
        context = f"...{context}"
    if end < len(content):
        context = f"{context}..."
    return context


def truncate(value: str, max_len: int) -> str:
    """Truncate ``value`` to ``max_len`` characters with an ellipsis."""
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}..."


__all__ = [
    "ScriptValidator",
    "ValidationFailure",
    "ValidationResult",
    "extract_context",
    "has_command_substitution",
    "has_embedded_shell_commands",
    "has_network_access",
    "has_reverse_shell_pattern",
    "has_shell_operators",
    "truncate",
]
