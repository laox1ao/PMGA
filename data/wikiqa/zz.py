#-*-encoding:utf-8-*-
#!/usr/bin/env python

import pickle

f = pickle.load(open('./wiki_train.pkl','r'))
fo = open('visible_data.pkl','w')
result = []
i = 0
for line in f:
#    if(i>400):
#        break
    if(line[2]==1 and len(line[0])<=10 and len(line[1])<=10):
        result.append(line)
print len(result)
pickle.dump(result,fo)

if __name__ == '__main__':
    pass
