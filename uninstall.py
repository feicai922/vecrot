"""Uninstall only the plugin; models, environments and source remain intact.

PowerShell 7: python -X utf8 uninstall.py [--qgis-profile vecrot-dev]
"""
import sys

from install import main


if __name__ == "__main__":
    sys.exit(main(["--uninstall", *sys.argv[1:]]))
