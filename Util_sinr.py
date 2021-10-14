
class Util:

  def reference_position(pos,prec,start=(0,0)):
    xa = pos[0] - start[0]
    ya = pos[1] - start[1]
    x = xa - (xa % prec) + start[0] + prec/2
    y = ya - (ya % prec) + start[1] + prec/2
    return (x,y)
