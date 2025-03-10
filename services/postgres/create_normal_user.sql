-- Create the read-only user
CREATE USER :username WITH PASSWORD :userpass;

-- Grant read-only permissions
GRANT CONNECT ON DATABASE postgres TO :username;
GRANT USAGE ON SCHEMA public TO :username;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO :username;

-- Ensure future tables are readable
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO :username;
