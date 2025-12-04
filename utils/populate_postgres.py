"""Compatibility shim - original CLI location.

This module delegates to `refactx.cli.populate_postgres.main` so scripts that
call `python utils/populate_postgres.py` continue to work while the canonical
implementation lives in `refactx.cli`.
"""

from refactx.cli.populate_postgres import main

if __name__ == "__main__":
    main()