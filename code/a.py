with open("/home/lawson/bleu_base.txt",'r') as f:
    res = []
    line = f.readline()
    while(line):
        res.append( "==".join(line.strip("\n").split()))
        print(res[-1])
        line = f.readline()

with open("/home/lawson/requirements.txt",'w') as f:
    for line in res:
        f.write(line+"\n")

        