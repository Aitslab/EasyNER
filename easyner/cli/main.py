#!/usr/bin/env python3
"""EasyNER Command Line Tool.

A unified CLI for all EasyNER operations.

Usage:
    easyner [command] [options]

Available Commands:
    config      Configuration management operations

Future commands may include:
    ner         Named entity recognition operations
    preprocess  Data preprocessing operations
"""

import argparse
import sys

import argcomplete

from easyner.cli import config_cli


def main() -> int:
    """Main entry point for the EasyNER CLI."""
    parser = argparse.ArgumentParser(
        description="EasyNER Command Line Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Setup subparsers for top-level commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management operations",
    )
    config_cli.setup_parsers(config_parser)

    # Future commands can be added here
    # ner_parser = subparsers.add_parser("ner", help="Named entity recognition operations")
    # ner_cli.setup_parsers(ner_parser)

    # Enable autocomplete
    try:
        argcomplete.autocomplete(parser)
    except ImportError:
        # argcomplete is not installed, continue without autocomplete
        pass

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == "config":
        return config_cli.handle_command(args)
    # Additional commands would be handled here
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
