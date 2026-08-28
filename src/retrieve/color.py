"""ANSI color helpers matching Ch's CLI color scheme.

Ch uses raw ANSI escape codes (see its internal/ui/ui.go). This module
mirrors that palette so Smart Search output looks consistent:

  blue    prompts (>, user:)
  green   assistant/content, success, ON
  red     errors, destructive, OFF
  yellow  info, spinner, help
  cyan    labels, filenames, paths
  magenta model names
  gray    metadata, timestamps, reasoning

Color is auto-disabled when stdout is not a terminal (piped/redirected),
matching Ch's IsPipedOutput behavior.
"""

import sys

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
GRAY = "\033[90m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"

_is_tty = sys.stdout.isatty()


def _wrap(text, color):
    if not _is_tty or not text:
        return text
    return f"{color}{text}{RESET}"


def red(text):
    return _wrap(text, RED)


def green(text):
    return _wrap(text, GREEN)


def yellow(text):
    return _wrap(text, YELLOW)


def blue(text):
    return _wrap(text, BLUE)


def magenta(text):
    return _wrap(text, MAGENTA)


def cyan(text):
    return _wrap(text, CYAN)


def gray(text):
    return _wrap(text, GRAY)


def bold(text):
    return _wrap(text, BOLD)


def underline(text):
    return _wrap(text, UNDERLINE)
