# Postgres index services
This folder contains the services setup with docker compose.

The services are:
- PostgreSQL db to save the tree representation.
- API for serving a postgreSQL db over http(s).
- Nginx for handling https and password protection.

You can follow both the Unsecure setup (for local/development environment) or the TLS one.

#### Copy sample configs
```
cp env_sample.txt .env
cp postgresql_sample.conf postgresql.conf
cp docker-compose-sample.yml docker-compose.yml
```

Edit them according to your needs.

Suggestion for `.env`: use 127.0.0.1 as listen address for unsecure services.

#### Start the services
```
sudo docker compose up -d
```
## HTTPS
#### Create certs
Replace example.com with your domain. 999 is the user id of the postgres container.
```
sudo ./generate_certs.sh example.com 999
```

#### Edit compose file
Uncomment API and Nginx services in the `docker-compose.yml`

#### Restart services
```
sudo docker compose restart
```

### Keep secrets
Keep `server.key` and `ca.key` secret.
