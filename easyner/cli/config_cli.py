#!/usr/bin/env python3
"""Configuration CLI Tool.

This tool provides a command-line interface for working with
EasyNER configuration files.
It leverages the OOP-based configuration management system to
provide a unified interface for all configuration operations.

Examples:
    # Validate all configuration files
    easyner config validate

    # Generate a template configuration file
    easyner config template

    # Ensure a config.json file exists (create if needed)
    easyner config ensure

    # View the current configuration (entire or a section)
    easyner config view
    easyner config view --section ner

    # Backup the current configuration
    easyner config backup create

    # List available backups
    easyner config backup list

    # Restore a backup
    easyner config backup restore --name backup_20250423_120000.json

"""

import argparse
import json
import sys

from easyner.config import config_manager


def setup_validate_parser(subparsers) -> None:
    """Setup the validate command parser."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate configuration files",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")


def setup_template_parser(subparsers) -> None:
    """Setup the template command parser."""
    parser = subparsers.add_parser(
        "template",
        help="Generate a template configuration",
    )
    parser.add_argument(
        "--skip-prettier",
        action="store_true",
        help="Skip formatting with prettier",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")


def setup_ensure_parser(subparsers) -> None:
    """Setup the ensure command parser."""
    parser = subparsers.add_parser(
        "ensure",
        help="Ensure config.json exists (create if needed)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")


def setup_view_parser(subparsers) -> None:
    """Setup the view command parser."""
    parser = subparsers.add_parser("view", help="View configuration contents")
    parser.add_argument("--section", type=str, help="View a specific section")
    parser.add_argument(
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (pretty or json)",
    )


def setup_backup_parser(subparsers):
    """Setup the backup command parser and its subcommands."""
    parser = subparsers.add_parser(
        "backup",
        help="Backup management operations",
    )
    backup_subparsers = parser.add_subparsers(dest="backup_cmd", required=True)

    # Create backup command
    create_parser = backup_subparsers.add_parser(
        "create",
        help="Create a new backup",
    )
    create_parser.add_argument(
        "--name",
        type=str,
        help="Description for the backup",
    )

    # List backups command
    backup_subparsers.add_parser("list", help="List available backups")

    # Restore backup command
    restore_parser = backup_subparsers.add_parser(
        "restore",
        help="Restore a backup",
    )
    restore_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Backup name or path to restore",
    )

    # Delete backup command
    delete_parser = backup_subparsers.add_parser(
        "delete",
        help="Delete a backup",
    )
    delete_parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Backup name or path to delete",
    )


def setup_parsers(parser):
    """Setup the config command parser and its subcommands."""
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    setup_validate_parser(subparsers)
    setup_template_parser(subparsers)
    setup_ensure_parser(subparsers)
    setup_view_parser(subparsers)
    setup_backup_parser(subparsers)


def handle_validate(args):
    """Handle the validate command."""
    # Update quiet flag if needed
    if args.quiet:
        config_manager.quiet = True

    result = config_manager.validate_all()
    if not args.quiet:
        if result:
            print("✓ All configuration files are valid")
        else:
            print("✗ Validation found issues with configuration files")

    return 0 if result else 1


def handle_template(args) -> int:
    """Handle the template command."""
    # Update quiet flag if needed
    if args.quiet:
        config_manager.quiet = True

    result = config_manager.generate_template(skip_prettier=args.skip_prettier)
    if not args.quiet:
        if result:
            print(
                f"✓ Template generated successfully at {config_manager.template_path}",
            )
        else:
            print("✗ Template generation completed with issues")

    return 0 if result else 1


def handle_ensure(args):
    """Handle the ensure command."""
    # Update quiet flag if needed
    if args.quiet:
        config_manager.quiet = True

    result = config_manager.ensure_config_exists()
    if not args.quiet:
        if result:
            print(f"✓ Configuration exists at {config_manager.config_path}")
        else:
            print("✗ Failed to ensure configuration exists")

    return 0 if result else 1


def handle_view(args):
    """Handle the view command."""
    try:
        if args.section:
            # View a specific section using dictionary access
            try:
                data = config_manager[args.section]
                if not getattr(args, "quiet", False):
                    print(f"=== Configuration Section: {args.section} ===")
            except KeyError:
                print(
                    f"Error: Section '{args.section}' not found in configuration",
                )
                return 1
        else:
            # View the entire configuration
            data = config_manager.get_config()
            if not getattr(args, "quiet", False):
                print("=== Complete Configuration ===")

        # Output the data in the requested format
        if args.format == "pretty":
            print(json.dumps(data, indent=2, sort_keys=False))
        else:
            print(json.dumps(data))

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def handle_backup_create(args):
    """Create a backup of the current configuration using the ConfigManager."""
    try:
        # Create backup using the ConfigManager
        backup_path = config_manager.create_backup(description=args.name)
        print(f"✓ Backup created: {backup_path}")
        return 0
    except Exception as e:
        print(f"Error creating backup: {e}")
        return 1


def handle_backup_list(args) -> int:
    """List all available configuration backups using ConfigManager."""
    try:
        # Get all backups
        backups = config_manager.list_backups()

        if not backups:
            print("No backups found.")
            return 0

        # Print backup information
        print(f"=== Available Backups ({len(backups)}) ===")
        for i, backup in enumerate(backups, 1):
            size = backup.path.stat().st_size / 1024  # Size in KB

            # Format backup information
            print(f"{i}. {backup.name}")
            if backup.description:
                print(f"   Description: {backup.description}")
            print(
                f"   Created: {backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            print(f"   Size: {size:.2f} KB")
            print()

        return 0
    except Exception as e:
        print(f"Error listing backups: {e}")
        return 1


def handle_backup_restore(args):
    """Restore a configuration from a backup using the ConfigManager."""
    backup_name = args.name

    try:
        # Restore the backup
        result = config_manager.restore_backup(backup_name)
        if result:
            print(f"✓ Configuration restored from backup: {backup_name}")
            return 0
        else:
            print(f"Error: Failed to restore backup '{backup_name}'")
            return 1
    except FileNotFoundError:
        print(f"Error: Backup '{backup_name}' not found")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error restoring backup: {e}")
        return 1


def handle_backup_delete(args):
    """Delete a backup using the ConfigManager."""
    backup_name = args.name

    try:
        # Delete the backup
        result = config_manager.delete_backup(backup_name)
        if result:
            print(f"✓ Backup deleted: {backup_name}")
            return 0
        else:
            print(f"Error: Failed to delete backup '{backup_name}'")
            return 1
    except FileNotFoundError:
        print(f"Error: Backup '{backup_name}' not found")
        return 1
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error deleting backup: {e}")
        return 1


def handle_backup(args):
    """Handle the backup command and its subcommands."""
    if args.backup_cmd == "create":
        return handle_backup_create(args)
    elif args.backup_cmd == "list":
        return handle_backup_list(args)
    elif args.backup_cmd == "restore":
        return handle_backup_restore(args)
    elif args.backup_cmd == "delete":
        return handle_backup_delete(args)
    else:
        print(f"Unknown backup command: {args.backup_cmd}")
        return 1


def handle_command(args):
    """Handle the config command and its subcommands."""
    if args.subcommand == "validate":
        return handle_validate(args)
    elif args.subcommand == "template":
        return handle_template(args)
    elif args.subcommand == "ensure":
        return handle_ensure(args)
    elif args.subcommand == "view":
        return handle_view(args)
    elif args.subcommand == "backup":
        return handle_backup(args)
    else:
        print(f"Unknown config subcommand: {args.subcommand}")
        return 1


def main():
    """Main entry point for direct execution of the configuration CLI tool.

    This is kept for backwards compatibility but the preferred way is to use
    the unified CLI with: easyner config [command]
    """
    parser = argparse.ArgumentParser(
        description="EasyNER Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Setup subparsers for commands
    setup_parsers(parser)

    # Parse arguments
    args = parser.parse_args()

    # Handle commands by passing to the command handler
    return handle_command(args)


if __name__ == "__main__":
    sys.exit(main())
