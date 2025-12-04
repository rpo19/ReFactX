"""Compatibility shim - original CLI location.

This module delegates to `refactx.cli.debug_postgres.main` so scripts that
call `python utils/debug_postgres.py` continue to work while the canonical
implementation lives in `refactx.cli`.
"""

from refactx.cli.debug_postgres import main

if __name__ == "__main__":
    main()
--postgres-url, --table-name, --rootkey, --switch-parameter, --end-of-triple, --model-name
Actual values:
--index-module {index_module}
# or
"""Compatibility shim - original CLI location.

This module delegates to `refactx.cli.debug_postgres.main` so scripts that
call `python utils/debug_postgres.py` continue to work while the canonical
implementation lives in `refactx.cli`.
"""

from refactx.cli.debug_postgres import main

if __name__ == "__main__":
    main()




