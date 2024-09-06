#!/usr/bin/env python3

import argparse
from collections import defaultdict, namedtuple
import os
import re
import signal
import subprocess
import sys
import threading
from typing import IO, Set, TextIO

from .diagnostic_formatter import (
    DiagnosticFormatter,
    CompactDiagnosticFormatter,
    FancyDiagnosticFormatter,
    GithubActionWorkflowCommandDiagnosticFormatter,
)
from .pylspclient.json_rpc_endpoint import JsonRpcEndpoint
from .pylspclient.lsp_endpoint import LspEndpoint
from .pylspclient.lsp_client import LspClient
from .pylspclient.lsp_structs import TextDocumentItem, LANGUAGE_IDENTIFIER
from .version import __version__

__all__ = ["main_cli"]


class ReadPipe(threading.Thread):
    def __init__(self, pipe: IO[bytes], out: TextIO):
        threading.Thread.__init__(self)
        self.pipe = pipe
        self.out = out

    def run(self):
        line = self.pipe.readline().decode("utf-8")
        while line:
            print(line, file=self.out)
            line = self.pipe.readline().decode("utf-8")


def kill_child_process(sig, _, child_processes, pbar):
    """Kill child processes on SIGINT"""
    assert sig == signal.SIGINT
    if pbar is not None:
        pbar.close()
    for child in child_processes:
        print(f"Terminating child process {child.pid}...", file=sys.stderr)
        child.terminate()
        child.wait()
        print(f"Child process {child.pid} terminated.", file=sys.stderr)
    sys.exit(1)


class FileExtensionFilter:
    def __init__(self, extensions: Set[str]):
        self.extensions = extensions

    def __call__(self, file_path):
        return os.path.splitext(file_path)[1][1:] in self.extensions


def _file_uri(path: str):
    return "file://" + path


def _uri_file(uri: str):
    if not uri.startswith("file://"):
        raise ValueError("Not a file URI: " + uri)
    return uri[7:]


def _is_output_supports_color(output: TextIO):
    return hasattr(output, "isatty") and output.isatty()


class DiagnosticCollector:
    SEVERITY_INT = {
        "error": 1,
        "warn": 2,
        "info": 3,
        "hint": 4,
    }

    def __init__(self):
        self.diagnostics = {}
        self.requested_files = set()
        self.cond = threading.Condition()

    def handle_publish_diagnostics(self, args):
        file = _uri_file(args["uri"])
        if file not in self.requested_files:
            return
        self.cond.acquire()
        self.diagnostics[file] = args["diagnostics"]
        self.cond.notify()
        self.cond.release()

    def request_diagnostics(self, lsp_client: LspClient, file_path: str):
        file_path = os.path.abspath(file_path)
        languageId = LANGUAGE_IDENTIFIER.CPP
        version = 1
        text = open(file_path, "r").read()
        self.requested_files.add(file_path)
        lsp_client.didOpen(
            TextDocumentItem(_file_uri(file_path), languageId, version, text)
        )

    def check_failed(self, fail_on_severity: str) -> bool:
        severity_level = self.SEVERITY_INT[fail_on_severity]
        for diagnostics in self.diagnostics.values():
            for diagnostic in diagnostics:
                if diagnostic["severity"] <= severity_level:
                    return True
        return False

    def format_diagnostics(self, formatter: DiagnosticFormatter) -> str:
        return formatter.format(sorted(self.diagnostics.items())).rstrip()


def main_cli():
    DEFAULT_ALLOW_EXTENSIONS = [
        "c",
        "h",
        "cpp",
        "cc",
        "cxx",
        "hpp",
        "hh",
        "hxx",
        "cu",
        "cuh",
    ]

    parser = argparse.ArgumentParser(
        prog="clangd-tidy",
        description="Run clangd with clang-tidy and output diagnostics. This aims to serve as a faster alternative to clang-tidy.",
        epilog="Find more information on https://github.com/lljbash/clangd-tidy.",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "-p",
        "--compile-commands-dir",
        default="build",
        help="Specify a path to look for compile_commands.json. If the path is invalid, clangd will look in the current directory and parent paths of each source file. [default: build]",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of async workers used by clangd. Background index also uses this many workers. [default: 1]",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Output file for diagnostics. [default: stdout]",
    )
    parser.add_argument(
        "--clangd-executable",
        default="clangd",
        help="Path to clangd executable. [default: clangd]",
    )
    parser.add_argument(
        "--allow-extensions",
        default=DEFAULT_ALLOW_EXTENSIONS,
        help=f"A comma-separated list of file extensions to allow. [default: {','.join(DEFAULT_ALLOW_EXTENSIONS)}]",
    )
    parser.add_argument(
        "--fail-on-severity",
        metavar="SEVERITY",
        choices=DiagnosticCollector.SEVERITY_INT.keys(),
        default="hint",
        help=f"On which severity of diagnostics this program should exit with a non-zero status. Candidates: {', '.join(DiagnosticCollector.SEVERITY_INT)}. [default: hint]",
    )
    parser.add_argument(
        "--tqdm", action="store_true", help="Show a progress bar (tqdm required)."
    )
    parser.add_argument(
        "--github",
        action="store_true",
        help="Append workflow commands for GitHub Actions to output.",
    )
    parser.add_argument(
        "--git-root",
        default=os.getcwd(),
        help="Root directory of the git repository. Only works with --github. [default: current directory]",
    )
    parser.add_argument(
        "-c",
        "--compact",
        action="store_true",
        help="Print compact diagnostics (legacy).",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=2,
        help="Number of additional lines to display on both sides of each diagnostic. This option is ineffective with --compact. [default: 2]",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Colorize the output. This option is ineffective with --compact. [default: auto]",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output from clangd."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes suggested by clangd.",
    )
    parser.add_argument(
        "filename",
        nargs="+",
        help="Files to check. Files whose extensions are not in ALLOW_EXTENSIONS will be ignored.",
    )
    args = parser.parse_args()

    ext_filter = FileExtensionFilter(set(map(str.strip, args.allow_extensions)))
    files = list(filter(ext_filter, args.filename))
    run(
        files=files,
        compile_commands_dir=args.compile_commands_dir,
        clangd_executable=args.clangd_executable,
        jobs=args.jobs,
        output=args.output,
        fail_on_severity=args.fail_on_severity,
        tqdm=args.tqdm,
        github=args.github,
        git_root=args.git_root,
        compact=args.compact,
        context=args.context,
        color=args.color,
        verbose=args.verbose,
        fix=args.fix,
    )


def run(
    files,
    compile_commands_dir,
    clangd_executable,
    jobs,
    output,
    fail_on_severity,
    tqdm,
    github,
    git_root,
    compact,
    context,
    color,
    verbose,
    fix,
):
    for file in files:
        if not os.path.isfile(file):
            print(f"File not found: {file}", file=sys.stderr)
            sys.exit(1)

    files_to_process = files

    while files_to_process:
        collector = collect_diagnostics(
            files_to_process,
            compile_commands_dir,
            clangd_executable,
            jobs,
            verbose,
            tqdm,
        )

        if fix:
            files_to_process, collector.diagnostics = apply_fixes(collector.diagnostics)
            if files_to_process:
                print(
                    f"Reprocessing {len(files_to_process)} files due to overlapping changes",
                    file=sys.stderr,
                )
        else:
            files_to_process = []

    formatter = (
        FancyDiagnosticFormatter(
            extra_context=context,
            enable_color=(
                _is_output_supports_color(output)
                if color == "auto"
                else color == "always"
            ),
        )
        if not compact
        else CompactDiagnosticFormatter()
    )
    print(collector.format_diagnostics(formatter), file=output)
    if github:
        print(
            collector.format_diagnostics(
                GithubActionWorkflowCommandDiagnosticFormatter(
                    git_root
                )
            ),
            file=output,
        )
    if collector.check_failed(fail_on_severity):
        exit(1)


def collect_diagnostics(
    files, compile_commands_dir, clangd_executable, jobs, verbose, tqdm
):
    clangd_command = [
        f"{clangd_executable}",
        f"--compile-commands-dir={compile_commands_dir}",
        "--clang-tidy",
        f"-j={jobs}",
        "--background-index",
        "--background-index-priority=normal",
        "--pch-storage=memory",
        "--enable-config",
        "--offset-encoding=utf-16",
    ]

    p = subprocess.Popen(
        clangd_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert p.stderr is not None
    read_pipe = ReadPipe(p.stderr, verbose and sys.stderr or open(os.devnull, "w"))
    read_pipe.start()

    # Kill clangd subprocess on SIGINT
    pbar = None  # use to close progress bar if it exists
    signal.signal(signal.SIGINT, lambda sig, _: kill_child_process(sig, _, [p], pbar))

    collector = DiagnosticCollector()

    json_rpc_endpoint = JsonRpcEndpoint(p.stdin, p.stdout)
    lsp_endpoint = LspEndpoint(
        json_rpc_endpoint,
        notify_callbacks={
            "textDocument/publishDiagnostics": lambda args: collector.handle_publish_diagnostics(
                args
            ),
        },
        timeout=5,
    )
    lsp_client = LspClient(lsp_endpoint)

    root_path = os.path.abspath(".")
    root_uri = _file_uri(root_path)
    workspace_folders = [{"name": "foo", "uri": root_uri}]

    lsp_client.initialize(
        processId=p.pid,
        rootPath=None,
        rootUri=root_uri,
        initializationOptions=None,
        capabilities={
            "textDocument": {
                "publishDiagnostics": {
                    "codeActionsInline": True,  # only supported by clangd
                },
            }
        },
        trace="off",
        workspaceFolders=workspace_folders,
    )
    lsp_client.initialized()

    for file in files:
        collector.request_diagnostics(lsp_client, file)

    if tqdm:
        try:
            from tqdm import tqdm
        except ImportError:
            print(
                "tqdm not found. Please install tqdm to enable progress bar.",
                file=sys.stderr,
            )
            tqdm = False

    if tqdm:
        from tqdm import tqdm

        with tqdm(total=len(files)) as pbar:
            collector.cond.acquire()
            while len(collector.diagnostics) < len(files):
                pbar.update(len(collector.diagnostics) - pbar.n)
                collector.cond.wait()
            pbar.update(len(collector.diagnostics) - pbar.n)
            collector.cond.release()
    else:
        collector.cond.acquire()
        while len(collector.diagnostics) < len(files):
            collector.cond.wait()
        collector.cond.release()

    lsp_client.shutdown()
    lsp_client.exit()
    lsp_endpoint.join()
    os.wait()
    if read_pipe.is_alive():
        read_pipe.join()

    return collector


Position = namedtuple("Position", ["line", "character"])
Range = namedtuple("Range", ["start", "end"])
Change = namedtuple("Change", ["range", "new_text"])
ChangeSet = namedtuple("ChangeSet", ["action", "changes"])


def range_from_dict(d):
    return Range(
        Position(d["start"]["line"], d["start"]["character"]),
        Position(d["end"]["line"], d["end"]["character"]),
    )


def apply_fixes(file_diagnostics):
    remaining_diagnostics = {}
    all_change_sets = defaultdict(list)
    applied_changes = defaultdict(set)
    for file, diagnostics in file_diagnostics.items():
        unfixed_diagnostics = []
        for diagnostic in diagnostics:
            for action in diagnostic.get("codeActions", []):
                assert action["kind"] == "quickfix", action
                assert "edit" in action, action
                assert list(action["edit"].keys()) == ["changes"], action
                assert len(action["edit"]["changes"]) > 0, action
                title = action["title"]
                if title == "remove all unused includes":
                    # Use the other actions to remove the includes
                    continue
                for document, edits in action["edit"]["changes"].items():
                    edit_file = document.removeprefix("file://")
                    assert (
                        edit_file in file_diagnostics
                    ), f"Got a change for a file that was not requested: {edit_file}"
                    changes = [
                        Change(
                            range=range_from_dict(edit["range"]),
                            new_text=edit["newText"],
                        )
                        for edit in edits
                    ]
                    # Sort changes by end position in descending order so that we don't change the position of
                    # subsequent changes when applying them
                    changes.sort(key=lambda change: change.range.end, reverse=True)
                    for change1, change2 in zip(changes, changes[1:]):
                        # Ensure that there is no overlap between changes within the same action
                        assert change1.range.end >= change2.range.start, (
                            change1,
                            change2,
                        )
                    # Check that we don't apply the same action multiple times (e.g. if adding the same header is
                    # suggested for multiple diagnostics)
                    key = tuple(changes)
                    if key in applied_changes[edit_file]:
                        continue
                    applied_changes[edit_file].add(key)
                    all_change_sets[edit_file].append(
                        ChangeSet(action=f'"{title}"', changes=changes)
                    )
                # Ensure we only apply one action per diagnostic
                break
            else:
                unfixed_diagnostics.append(diagnostic)
        if unfixed_diagnostics:
            remaining_diagnostics[file] = unfixed_diagnostics

    files_to_reprocess = set()

    for file, change_sets in all_change_sets.items():
        # Sort change sets by the end position of the last change in descending order so that we don't change the
        # position of subsequent non-overlapping change sets when applying them
        change_sets.sort(
            key=lambda change_set: change_set.changes[-1].range.end, reverse=True
        )
        merged_change_sets = []
        for change_set in change_sets:
            if not merged_change_sets:
                merged_change_sets.append(change_set)
                continue
            for change in change_set.changes:
                for merged_change in merged_change_sets[-1].changes:
                    if (
                        change.range.end >= merged_change.range.start
                        and change.range.start <= merged_change.range.end
                    ):
                        # Overlapping changes, can't merge
                        merged_change_sets.append(change_set)
                        break
                else:
                    # No overlap for this change
                    continue
                break
            else:
                # No overlap for any change in the set
                merged_change_set = ChangeSet(
                    action=merged_change_sets[-1].action + " + " + change_set.action,
                    changes=merged_change_sets[-1].changes + change_set.changes,
                )
                merged_change_set.changes.sort(
                    key=lambda change: change.range.end, reverse=True
                )
                merged_change_sets[-1] = merged_change_set
                continue

        with open(file, "r") as f:
            # Read the file as UTF-16 because clangd outputs diagnostics with UTF-16 offsets (le is for little-endian
            # and doesn't matter but without it Python will add a byte order mark to the beginning of the string)
            content = f.read().encode("utf-16-le")

        print(f"Applying changes for {file}:", file=sys.stderr)
        line_indices = [0] + [
            m.end() for m in re.finditer("\n".encode("utf-16-le"), content)
        ]
        pos_to_index = (
            lambda pos: line_indices[pos.line] + pos.character * 2
        )  # 2 bytes per character
        last_start = None
        for change_set in merged_change_sets:
            # Skip change sets that overlap with the previous one
            if last_start is not None and change_set.changes[0].range.end >= last_start:
                print(
                    f"  Skipping {change_set.action} because it overlaps with the previous change set",
                    file=sys.stderr,
                )
                files_to_reprocess.add(file)
                continue
            last_start = change_set.changes[-1].range.start
            print(f"  {change_set.action}:", file=sys.stderr)
            for change in change_set.changes:
                start = change.range.start
                end = change.range.end
                old_text = content[pos_to_index(start) : pos_to_index(end)].decode(
                    "utf-16-le"
                )
                print(
                    f"    {start.line + 1}:{start.character + 1}-{end.line + 1}:{end.character + 1}: "
                    f"'{old_text}' -> '{change.new_text}'",
                    file=sys.stderr,
                )
                content = (
                    content[: pos_to_index(start)]
                    + change.new_text.encode("utf-16-le")
                    + content[pos_to_index(end) :]
                )
        print("  Done", file=sys.stderr)
        with open(file, "w") as f:
            f.write(content.decode("utf-16-le"))

    return list(sorted(files_to_reprocess)), remaining_diagnostics


if __name__ == "__main__":
    main()
