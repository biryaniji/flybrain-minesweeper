"""Exact optimal win rate for 4x4 Minesweeper, 2 mines, safe first click.
Searches over every consistent mine layout. Run: python optimal.py (takes a few seconds)."""
import itertools, sys
from functools import lru_cache
N=4; CELLS=range(16)
def nbrs(i):
    r,c=divmod(i,N)
    return [rr*N+cc for rr in range(r-1,r+2) for cc in range(c-1,c+2) if 0<=rr<N and 0<=cc<N and (rr,cc)!=(r,c)]
NB=[nbrs(i) for i in CELLS]
def count(i,m): return sum(j in m for j in NB[i])
def reveal(R,c,m):
    R=set(R); q=[c]; R.add(c)
    while q:
        x=q.pop()
        if count(x,m)==0:
            for y in NB[x]:
                if y not in R: R.add(y); q.append(y)
    return frozenset(R)
@lru_cache(maxsize=None)
def V(S,R):
    best=0.0
    for c in CELLS:
        if c in R: continue
        groups={}
        for m in S:
            if c in m: continue
            R2=reveal(R,c,m)
            key=(R2,tuple((i,count(i,m)) for i in sorted(R2)))
            groups.setdefault(key,[]).append(m)
        tot=0.0
        for (R2,_),g in groups.items():
            if len(R2)==16-2: tot+=len(g)
            else: tot+=len(g)*V(frozenset(g),R2)
        best=max(best,tot/len(S))
        if best==1.0: break
    return best
res=[]
for a in CELLS:
    S=[frozenset(p) for p in itertools.combinations([x for x in CELLS if x!=a],2)]
    tot=0.0; groups={}
    for m in S:
        R2=reveal(frozenset(),a,m)
        key=(R2,tuple((i,count(i,m)) for i in sorted(R2)))
        groups.setdefault(key,[]).append(m)
    for (R2,_),g in groups.items():
        tot+= len(g) if len(R2)==14 else len(g)*V(frozenset(g),R2)
    res.append(tot/len(S))
print([round(x,4) for x in res]); print("optimal:",max(res), "best first click:",res.index(max(res)))