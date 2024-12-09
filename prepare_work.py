import psycopg
import sys

postgres_connect = sys.argv[1]
data_size = int(sys.argv[2])
n_workers = int(sys.argv[3])
delete = sys.argv[4] if len(sys.argv) >= 5 else 'no'

with psycopg.connect(postgres_connect) as conn:
    with conn.cursor() as cur:
        if delete == 'delete':
            cur.execute('DELETE FROM work; DELETE FROM relations; DELETE FROM entitylabels; DELETE FROM predlabels; DELETE FROM attributes;')

        cur.execute("SELECT * FROM work;")
        results = cur.fetchone()

        assert results is None, 'ERROR: the work is already initialized'

        worker_data = data_size // n_workers

        worker_work = [[s,s+worker_data] for s in range(0, data_size, worker_data)]
        worker_work[-1][1] = data_size


        for work in worker_work:
            cur.execute('''INSERT INTO work (bytestart, byteend, lastbyteprocessed, actualbytestart, actualbyteend)
            VALUES (%s, %s, %s, %s, %s);''', (work[0], work[1], -1, work[0], work[1]))

        cur.execute("SELECT id, bytestart, byteend, lastbyteprocessed, actualbytestart, actualbyteend FROM work;")
        results = cur.fetchall()
        for item in results:
            print(item)

    conn.commit()
