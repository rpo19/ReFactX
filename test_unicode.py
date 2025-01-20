import bz2
wiki_dump = '/mnt/data/jupyterlab/rpozzi/latest-truthy-2024-12-11.nt.bz2'
with bz2.BZ2File(wiki_dump, 'r') as fd:
    for count, bline in enumerate(fd):
        line = bline.decode('unicode_escape') # TODO check if it fixes
        u8line = bline.decode('utf-8')
        if '\\u' in u8line: #<happiness> <different from> <Fel' in line:
            breakpoint()