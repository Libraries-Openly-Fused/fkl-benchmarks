# The E19 crop-sweep crop table: edges, alignment, overlaps per N (no GPU)
SX=[0,320,0,320,192]; SY=[0,0,200,200,100]; W=H=256; FW,FH=3840,2160
def table(n): return [(64+600*(i%6)+SX[i//30], 64+400*((i//6)%5)+SY[i//30]) for i in range(n)]
t=table(150)
assert all(t[i]==(64+600*(i%6),64+400*(i//6)) for i in range(30))
assert len(set(t))==150
print("largest right edge", max(x+W for x,y in t), "largest bottom edge", max(y+H for x,y in t), "smallest x, y", min(x for x,y in t), min(y for x,y in t))
print("source byte offset mod 16 per crop equal to its E19 crop (i mod 30):", all(((y*FW+x)*3)%16==((t[i%30][1]*FW+t[i%30][0])*3)%16 for i,(x,y) in enumerate(t)))
def ov(a,b): return abs(a[0]-b[0])<W and abs(a[1]-b[1])<H
print("N,overlapping_pairs,crops_overlapping_another,max_crops_on_one_pixel")
for n in [1]+list(range(10,151,10)):
    c=table(n)
    # the most crops on one pixel: for axis-aligned rectangles it is reached at a point (a left edge, a top edge)
    most=max(sum(x<=px<x+W and y<=py<y+H for x,y in c) for px in {x for x,_ in c} for py in {y for _,y in c})
    pairs=sum(ov(c[i],c[j]) for i in range(n) for j in range(i))
    some=sum(any(ov(c[i],c[j]) for j in range(n) if j!=i) for i in range(n))
    print(f"{n},{pairs},{some},{most}")
