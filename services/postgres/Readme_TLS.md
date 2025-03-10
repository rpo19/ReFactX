# TLS
In this setup we will enable TLS encryption without certificate verification
## PostreSQL with TLS
To use TLS initialize the db with a normal postgres instance without port forwarding.
Then remove the containers and create them with `docker-compose-tls.yml`.
### Create .env file
You can start from the example `env_sample.txt`.
Suggestion: until TLS is enabled use 127.0.0.1 as listen addr for postgres.
### Create certs
```
sudo ./generate_certs.sh
```
### Start postgres without TLS
```
sudo docker-compose up -d postgres
```
Now the db (empty) should have been created.
### Stop and delete the container
```
sudo docker-compose down
```
The db is persisted in the volume.
### Start the docker-compose-tls
Modify .env with the final listen address, then create and start it.
```
sudo docker-compose -f docker-compose-tls.yml up -d postgres
```
### Create normal user
```
source .env && sudo docker-compose exec -T postgres psql -U postgres -v username="$POSTGRES_NORMAL_USER_NAME"  -v userpass="'$POSTGRES_NORMAL_USER_PASSWORD'" < create_normal_user.sql
```
### Check if TLS is working
Connect from somewhere else and run `SHOW ssl;` (Replace configs like user and host).
```
psql "sslmode=verify-ca sslrootcert=ca.crt dbname=postgres host=host port=5432 user=user"
# then run
SHOW ssl;
```
### Check if unsecure connection is forbidden
Same as before except `sslmode=disable`.
```
psql "sslmode=disable dbname=postgres host=host port=5432 user=user"
# should give error
```

## Redis with TLS
### Set a strong password
Open `redis-tls.conf` and look for `requirepass`.
### Start redis
It will use the same `server.crt` and `server.key` as postgres.
```
sudo docker-compose -f docker-compose-tls.yml up -d redis
```
### Check if TLS is working
Connect from somewhere else and run some commands. (Replace configs like user and host).
```
redis-cli --tls -h host -p port --askpass --cacert ca.crt
# then run some commands
set mykey 12
get mykey
```