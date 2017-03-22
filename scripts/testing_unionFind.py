import utils.extUnionFind as extUF

prova = extUF.UnionFind(10)




# # Check inconsistency:
# prova.extend(2)
# prova.union(1,10)
# prova.shrink(2)
# print prova.find(1)

# Check shrinking
prova.extend(2)
prova.union(11,3)
prova.union(10,2)
print prova.find(9), prova.find(8)
out = prova.shrink(2,2)
print prova.find(9), prova.find(8)
print out
