#-*-encoding:utf-8-*-
#!/usr/bin/env python

import pickle

f = pickle.load(open('./wiki_train.pkl','r'))

i = 0
for line in f:
    if(i>400):
        break
    print line[0],line[2]
    i += 1

if __name__ == '__main__':
    pass
