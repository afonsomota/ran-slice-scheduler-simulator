#f = open("mcs.da","r")
#
#d = {}
#
#for line in f.readlines():
#  arr = line.split()
#  d[int(arr[0])]={'Qm':int(arr[1]),'R':float(arr[2]),'se':float(arr[3])}
#
#print d

f = open("tbs.da","r")

a = []
for line in f.readlines():
  a.append(int(line))
print a
