# Postgres index services
This folder contains the services setup with docker compose.

The services are:
- PostgreSQL db to save the tree representation.
- Redis for caching.

You can follow both the Unsecure setup (for local/development environment) or the TLS one.

## Unsecure
#### Create .env file
You can start from the example `env_sample.txt`.
Suggestion: use 127.0.0.1 as listen addr for unsecure services.

#### Create redis.conf
Copy `redis.conf.sample` to `redis.conf` and edit the password at the end of the file.

#### Start the services
```
sudo docker compose -f docker-compose-insecure.yml up -d
```
## TLS
In this setup we will enable TLS encryption without certificate verification
### PostreSQL with TLS
To use TLS we will initialize the db with a normal postgres instance without port forwarding.
Then remove the containers and create them with `docker-compose-tls.yml`.
#### Create .env file
You can start from the example `env_sample.txt`.
Suggestion: until TLS is enabled use 127.0.0.1 as listen addr for unsecure services.
#### Create certs
Replace example.com with your domain. 999 is the user id of the postgres container.
```
sudo ./generate_certs.sh example.com 999
```
#### Start postgres without TLS
```
sudo docker compose -f docker-compose-insecure.yml up -d postgres
```
Now the db (empty) should have been created.
#### Stop and delete the container
```
sudo docker compose -f docker-compose-insecure.yml down
```
The db is persisted in the volume.
#### Start the docker-compose-tls
Modify .env with the final listen address, then create and start it.
```
sudo docker compose -f docker-compose-tls.yml up -d postgres
```
#### Create normal user
Run with bash.
```
source .env && sudo docker compose -f docker-compose-tls.yml exec -T postgres psql -U postgres -v username="$POSTGRES_NORMAL_USER_NAME"  -v userpass="'$POSTGRES_NORMAL_USER_PASSWORD'" < create_normal_user.sql
```
#### Check if TLS is working
Connect from somewhere else and run `SHOW ssl;` (Replace configs like user and host).
```
psql "sslmode=verify-ca sslrootcert=ca.crt dbname=postgres host=host port=5432 user=user"
# then run
SHOW ssl;
```
#### Check if unsecure connection is forbidden
Same as before except `sslmode=disable`.
```
psql "sslmode=disable dbname=postgres host=host port=5432 user=user"
# should give error
```

### Redis with TLS
#### Create redis.conf and set a strong password
Copy `redis.conf.sample` to `redis.conf` and edit the password at the end of the file.
#### Enable TLS in redis.conf
Uncomment the lines after `TLS` at the end of the file and comment the Unsecure `port 6379`.
#### Start redis
It will use the same `server.crt` and `server.key` as postgres.
```
sudo docker compose -f docker-compose-tls.yml up -d redis
```
#### Check if TLS is working
Connect from somewhere else and run some commands. (Replace configs like user and host).
```
redis-cli --tls -h host -p port --askpass --cacert ca.crt
# then run some commands
set mykey 12
get mykey
```
### Modify Listen Address
Now that TLS is enabled you may want to reach postgres and redis from other machines. Just modify the listen address in the `.env` file to e.g. `0.0.0.0` for listening on all interfaces.
### Setup clients
Now copy the CA certificate `ca.crt` to the client machines and use it to verify the server certificate.
### Keep secrets
Keep `server.key` and `ca.key` secret.
