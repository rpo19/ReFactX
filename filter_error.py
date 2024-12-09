import sys

startpattern = 'Failed to convert Literal lexical form to value. Datatype=http://www.w3.org/2001/XMLSchema#dateTime, Converter=<function parse_datetime'
endpattern = 'isodate.isoerror.ISO8601Error'

dest = sys.stdout

inside = False
for line in sys.stdin:
    if inside:
        pass
        if line.startswith(endpattern):
            inside = False
    else:
        if line.startswith(startpattern):
            inside = True
        else:
            print(line, end='', file=dest)