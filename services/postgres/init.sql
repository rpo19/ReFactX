CREATE TABLE ctrie (
    id BIGINT GENERATED ALWAYS AS IDENTITY, -- Automatically generates unique values for id -- later make it PRIMARY KEY
    key BYTEA NOT NULL,
    children BYTEA,
    subtree BYTEA
);

