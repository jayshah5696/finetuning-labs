# Historical first bulk attempt

This run exposed malformed responses, duplicate variants, and long documents below 1,000 tokens.
Its derived release also preceded the final medium-length gate, so it fails the stricter current
audit. Do not train on that derived release. The final run reuses only compatible short and medium
calls that pass the current gates. Raw requests and responses are preserved without editing.
