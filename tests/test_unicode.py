import bz2
import sys

wiki_dump = sys.argv[1]
with bz2.BZ2File(wiki_dump, 'r') as fd:
    for count, bline in enumerate(fd):
        line = bline.decode('unicode_escape') # TODO check if it fixes
        u8line = bline.decode('utf-8')
        double_decode = bline.decode('unicode_escape').encode('utf-8').decode('utf-8')
        if '\\u' in u8line: #<happiness> <different from> <Fel' in line:
            breakpoint()
