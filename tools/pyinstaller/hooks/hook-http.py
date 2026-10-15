"""http imports ssl only for HTTPS and TLS, and the app makes no network connections.

Without ssl the bundle leaves out OpenSSL (libssl and libcrypto).
"""

excludedimports = ["ssl"]
