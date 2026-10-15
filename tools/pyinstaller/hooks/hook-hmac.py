"""hmac falls back to CPython's built-in hashes when _hashlib (OpenSSL) is absent."""

excludedimports = ["_hashlib"]
