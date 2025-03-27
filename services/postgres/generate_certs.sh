#!/bin/bash

set -e  # Exit on error

domain=$1
userid=$2  # e.g., 999

# === Step 1: Generate CA ===
echo "Generating CA..."
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 365 -key ca.key -out ca.crt -subj "/CN=$domain"

# === Step 2: Generate Server Key ===
echo "Generating server key..."
openssl genrsa -out server.key 4096

# === Step 3: Generate CSR (Certificate Signing Request) ===
echo "Generating CSR..."
openssl req -new -key server.key -out server.csr -subj "/CN=$domain"

# === Step 4: Sign the Server Certificate with the CA ===
echo "Signing server certificate with CA..."
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt -days 365 -sha256

# === Step 5: Set Permissions ===
echo "Setting permissions..."
chmod 600 server.key ca.key
chmod 644 server.crt ca.crt
chown $userid:$userid server.* ca.*

# Cleanup CSR and Serial File
rm server.csr

echo "Certificate generation completed!"
echo "Keep server.key and ca.key secret!"
echo "The servers (postgresql and redis) requires server.crt and server.key."
echo "ca.crt must be shared with the clients to verify the TLS connection with the server."
