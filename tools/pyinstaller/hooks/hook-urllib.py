"""urllib imports ssl for HTTPS and ftplib for ftp:// URLs; the app opens neither.

Without ssl the bundle leaves out OpenSSL (libssl and libcrypto).
"""

excludedimports = ["ftplib", "ssl"]
