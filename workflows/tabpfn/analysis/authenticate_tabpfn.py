"""Cache a Prior Labs API key entered privately in the terminal.

Sign in and accept the model license at https://platform.priorlabs.ai first.
Manual key entry avoids TabPFN 9.2.0's Windows browser-callback input polling.
Keys use the upstream package's user-local cache, outside the repository.
"""

import argparse
from getpass import getpass
import os
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--api-key', action='store_true',
                      help='Prompt privately for a key (the default; no localhost callback)')
    mode.add_argument('--browser', action='store_true',
                      help='Use upstream browser callback instead; may fail on Windows')
    args = parser.parse_args()
    if not args.browser:
        print('Open https://platform.priorlabs.ai in your browser and sign in.')
        print('Accept the TabPFN-3.5 license under Licenses, then copy your API key')
        print('from your account. This flow does not use a localhost redirect.')
        print('Paste the key at the following hidden prompt, then press Enter.')
        token = getpass('Prior Labs API key: ').strip()
        if not token:
            parser.error('No key supplied; nothing was saved')
        # Exact upstream token-cache location. Never write credentials to the repo.
        path = Path.home() / '.cache' / 'tabpfn' / 'auth_token'
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(token)
        print('API key cached locally. License access will be checked before downloading weights.')
    else:
        # Set before importing TabPFN because its settings are initialized on import.
        os.environ.setdefault('TABPFN_AUTH_GUI_URL', 'https://platform.priorlabs.ai')
        from tabpfn.browser_auth import ensure_license_accepted

        os.environ.pop('TABPFN_NO_BROWSER', None)
        ensure_license_accepted(hf_repo_id='tabpfn_3_5')
        print('TabPFN-3.5 access verified. You can now run analysis/compare_tabpfn.py.')
