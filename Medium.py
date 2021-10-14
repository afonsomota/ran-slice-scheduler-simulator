import sys
import pickle
import math
import numpy as np
from scipy.spatial import ConvexHull
import xml.etree.ElementTree as ET

from Util_sinr import Util

class Node:
  def __init__(self,height,nf,gain,txpower):
    self.height = height
    self.nf = nf #noise factor
    self.gain = gain #antenna isotropic gain
    self.txpower = txpower

  def directional_gain(self,tx_pos,rx_pos,cell_lobe):
    return self.gain

class UE(Node):
  def __init__(self,hue = 1.5):
    super().__init__(hue,7,0,23)

class ENB(Node):

  def __init__(self,hbs = 25):
    self.angle3db = 65
    self.maximum_attenuation = 30
    super().__init__(hbs,5,8,46)

  def horizontal_gain(self,angle,ma):
    """Calculate horizontal gain based in horizontal angle and maximum attenuation (ma)"""
    return -min(12*(angle/self.angle3db)**2,ma)

  def vertical_gain(self,angle,ma):
    """Calculate vertical gain based in vertical angle and maximum attenuation (ma)"""
    return -min(12*((angle-90)/self.angle3db)**2,ma)

  def directional_gain(self,tx_pos,rx_pos,cell_lobe):
    rp = (rx_pos[0] - tx_pos[0],rx_pos[1] - tx_pos[1],rx_pos[2] - tx_pos[2])
    clr = math.radians(cell_lobe)
    d = math.sqrt(rp[0]**2+rp[1]**2+rp[2]**2)
    hangle = math.degrees(math.atan2(rp[0],rp[1]))-cell_lobe
    if hangle > 180: hangle -= 360
    if hangle < -180: hangle += 360
    vangle = math.degrees(math.asin(rp[2]/d)) + 90
    hgain = self.horizontal_gain(hangle,self.maximum_attenuation)
    vgain = self.vertical_gain(vangle,self.maximum_attenuation)
    return -min(-abs(vgain+hgain),self.maximum_attenuation)
#hangle = math.radians(math.atan2(rp[0],rp[1])) - cell_lobe

class Medium:
  def __init__(self):
    self.carrier = 4
    self.tn = 174 #thermal noise
    self.sw = 20 #street width

class Building:
  def __init__(self,vertex,height):
    self.importAsConvexHull(vertex)
    self.height=height
    self.faceNormals=[]
    self.faceVertex=[]
    self.calculateFacesNormals()
    self.calculateFacesVertex()

  def importAsConvexHull(self,vertex):
    vertex2d = list(map(lambda x: [x[0],x[1]],vertex))
    hull = ConvexHull(vertex2d)
    cw_vertices = list(hull.vertices)
    cw_vertices.reverse()
    self.vertex = list(map(lambda i: [hull.points[i][0],hull.points[i][1],0],cw_vertices))
    return

  def calculateFacesNormals(self):
    normals = []
    for i in range(0,len(self.vertex)):
      ni = i + 1 if i != len(self.vertex) - 1 else 0
      vi0 = np.array(self.vertex[i])
      vih = np.array([self.vertex[i][0], self.vertex[i][1], self.height]) #vertex top
      vni0 = np.array(self.vertex[ni])
      nrm = np.cross(vih-vi0, vni0-vi0)
      if not np.array_equal(nrm,np.array([0,0,0])):
        nrm = nrm / np.linalg.norm(nrm)
      normals.append(nrm)
    normals.append(np.array([0,0,1])) #normal to roof
    normals.append(np.array([0,0,-1])) #normal to roof
    self.faceNormals = normals
    return

  def calculateFacesVertex(self):
    fvertex = list(map(np.array,self.vertex))
    fvertex.append(np.array([self.vertex[0][0],self.vertex[0][1],self.height])) #add one roof vertice
    fvertex.append(np.array(self.vertex[0]))
    self.faceVertex = fvertex
    return

  def getFacesNormals(self):
    return self.faceNormals

  def getFacesVertex(self):
    return self.faceVertex

  def calculateIntersection(self,tx_pos,rx_pos):
    """Intersection of segment with polyhedron. Source: http://geomalgorithms.com/a13-_intersect-4.html"""
    p0 = np.array(tx_pos)
    p1 = np.array(rx_pos)

    if np.array_equal(p1,p0): return False

    te = 0 #max entering
    tl = 1 #max leaving
    ds = p1-p0
    n = self.getFacesNormals()
    v = self.getFacesVertex()

    #for each face
    for i in range(0,len(v)):
      N = - np.dot(p0-v[i],n[i])
      D = np.dot(ds,n[i])

      #line parallel to face
      if D == 0:
        #if n is 0, segnent will never touch polyhedron
        if N < 0: return False
        continue

      t=N/D
      if D < 0:
        te = max(te,t)
        if te > tl: return False

      if D > 0:
        tl = min(tl,t)
        if tl < te: return False

    return True

    
    


class SNRCalculator:
  def __init__(self, bmap=None, bs=None, ue=None):
    self.ue = ue if ue != None else UE()
    self.enb = bs if bs != None else ENB()
    self.medium = Medium()
    self.building_height = 15
    if bmap != None:
      bmapf=bmap[0]
      bmapc=bmap[1]
      self.loadMap(bmapf,bmapc)

  def addObstacle(self,obs):
    self.obstacles.append(obs)

  def clearObstacles(self):
    self.obstacles.clear()

  #2500,1250
  def loadMap(self,bmap,center):
    """Load building map into an obstacle list"""
    self.obstacles = []
    tree = ET.parse(bmap)
    root = tree.getroot()
    polys = root.findall('.//poly')
    for p in polys:
      t = p.get('type')
      if 'building' in t:
        vertex = list(map(lambda x: [float(x.split(",")[0])-center[0],float(x.split(",")[1])-center[1],0] ,p.get('shape').split(" ")))
        obs = Building(vertex,self.building_height)
        self.addObstacle(obs)
    return

  def checkLOS(self,tx_pos,rx_pos):
    """Check obstacles in the way of the transmission, return 1 for LOS and 0 for NLOS"""
    for obs in self.obstacles:
      obj_int = obs.calculateIntersection(tx_pos,rx_pos)
      if obj_int: 
        return (False,obs.height)
    return (True,0)

  def getPathLossUMa(self,tx_pos,rx_pos,pre_los=None, pre_h=0):
    """Get the Path Loss between two nodes. Returns pathloss and shadow fading std-dev"""
    if pre_los == None: 
      (los,h) = self.checkLOS(tx_pos,rx_pos)
    else:
      los = pre_los
      h = pre_h
    d3 = math.sqrt((tx_pos[0]-rx_pos[0])**2 +
                   (tx_pos[1]-rx_pos[1])**2 +
                   (tx_pos[1]-rx_pos[1])**2
                  )
    d2 = math.sqrt((tx_pos[0]-rx_pos[0])**2 +
                   (tx_pos[1]-rx_pos[1])**2
                  )
    htx = tx_pos[2]
    hrx = rx_pos[2]
    if htx > hrx:
      hbs = htx
      hue = hrx
    else:
      hbs = hrx
      hue = htx
    dbp = 4 * (htx-1) * (hrx-1) * self.medium.carrier *10 / 3.

    if d2 < dbp:
      pl_los = 28 + 22*math.log10(d3) + 20 * math.log10(self.medium.carrier)
    elif d2 >= dbp:
      pl_los = 40*math.log10(d3) +  28 + 20*math.log10(self.medium.carrier) - \
               9*math.log10((dbp)**2+(htx-hrx)**2)

    if los:
      pl = pl_los
      sf = 4
      h = 0
    else:
      pl_nlos = 161.04 - 7*math.log10(self.medium.sw) + 7.5*math.log10(h) - \
                (24.37-3.7*(h/hbs)**2)*math.log10(hbs) + \
                (43.42-3.1*math.log10(hbs))*(math.log10(d3)-3) + \
                20*math.log10(self.medium.carrier) - \
                (3.2*(math.log10(17.625))**2-4.97) - \
                0.6*(hue-1.5)
      pl_nlos = 13.54 + 39.08 * math.log10(d3) + 20 * math.log10(self.medium.carrier) - 0.6*(hue-1.5)
      pl = max(pl_los, pl_nlos)
      sf = 6

    return (pl,sf,los,h)


  def getSNR(self,tx_pos,rx_pos,cell_lobe,direction="downlink",pre_los=None,pre_h=20):
    """Get the SINR in a direction  between two nodes according to their position"""
    if direction == "downlink":
      src=self.enb
      dst=self.ue
    else:
      src=self.ue
      dst=self.enb

    pl = self.getPathLossUMa(tx_pos,rx_pos,pre_los,pre_h)

    pr = src.txpower + src.directional_gain(tx_pos,rx_pos,cell_lobe) + dst.directional_gain(rx_pos,tx_pos,cell_lobe) - pl[0]

    snr = pr - dst.nf + self.medium.tn - 10*math.log10(20*1000000)

    return (snr,pl[1],pl[2],pl[3])



class Simulation:

  def __init__(self, step = 10, nBS=19, isd=500, hbs = 25, hue = 1.5, bmp = None, out = "."):
    self.nBS = nBS
    self.isd = isd
    self.step = step
    self.center = [0,0,0]
    self.sumo_center = [0,0,0]
    self.hbs = hbs
    self.hue = hue
    self.bs = ENB(hbs)
    self.ue = UE(hue)
    self.snr_calculator = SNRCalculator(bmp,self.bs,self.ue)
    if bmp != None:
      self.sumo_center = bmp[1]
    self.deployBSs()
    self.out = out
    self.cells = {}

  def deployBSs(self):
    h = self.hbs
    assert self.nBS == 1 or self.nBS == 7 or self.nBS == 19

    bsp = []

    bsp.append([0,0,h])

    if self.nBS == 1:
      self.BSlist = bsp
      return

    bsp.append([0,self.isd,h])
    bsp.append([0,-self.isd,h])
    bsp.append([-math.sqrt(3)*self.isd/2,self.isd/2,h])
    bsp.append([-math.sqrt(3)*self.isd/2,-self.isd/2,h])
    bsp.append([math.sqrt(3)*self.isd/2,self.isd/2,h])
    bsp.append([math.sqrt(3)*self.isd/2,-self.isd/2,h])

    if self.nBS == 7:
      self.BSlist = bsp
      return

    bsp.append([math.sqrt(3)*self.isd,0,h])
    bsp.append([math.sqrt(3)*self.isd,self.isd,h])
    bsp.append([math.sqrt(3)/2*self.isd,3/2*self.isd,h])
    bsp.append([0,2*self.isd,h])
    bsp.append([-math.sqrt(3)/2*self.isd,3/2*self.isd,h])
    bsp.append([-math.sqrt(3)*self.isd,self.isd,h])
    bsp.append([-math.sqrt(3)*self.isd,0,h])
    bsp.append([-math.sqrt(3)*self.isd,-self.isd,h])
    bsp.append([-math.sqrt(3)/2*self.isd,-3/2*self.isd,h])
    bsp.append([0,-2*self.isd,h])
    bsp.append([math.sqrt(3)/2*self.isd,-3/2*self.isd,h])
    bsp.append([math.sqrt(3)*self.isd,-self.isd,h])

    self.BSlist = bsp
    return

#  def setCenter(self,center_p):
#    self.center = center_p
#    for bsp in self.BSlist:
#      for i in range(0,bsp):
#        bsp[i] += center_p[i]

  def addObstacle(self,obs):
    self.snr_calculator.addObstacle(obs)

  def run(self,direction="downlink"):
    x_min = int(self.center[0] - 3*math.sqrt(3)*self.isd/2)
    x_max = int(self.center[0] + 3*math.sqrt(3)*self.isd/2)
    y_min = int(self.center[1] - 5*self.isd/2)
    y_max = int(self.center[1] + 5*self.isd/2)
    lobes = [-120,0,120]
    snr_per_pos={}
    max_snr_per_pos = {}
    i = 0
    c = 0
    max_distance = math.sqrt(3)*self.isd/2
    for b in self.BSlist:
      for lobe in lobes:
        f=open(self.out + "/map-cell_"+str(c)+"-bs_"+str(i)+"-lobe_"+str(lobe)+".out","w")
        for xa in np.arange(x_min,x_max,self.step):
          for ya in np.arange(y_min,y_max,self.step):
            (x,y) = Util.reference_position((xa,ya),self.step,self.center)
            if x not in snr_per_pos:
              snr_per_pos[x] = {}
              max_snr_per_pos[x] = {}
            if y not in snr_per_pos[x]: 
              snr_per_pos[x][y] = {}
            if math.sqrt((x-b[0])**2 + (y-b[1])**2) > max_distance:
              continue
            if direction == "downlink":
              snr=self.snr_calculator.getSNR(b,[x,y,self.ue.height],lobe,"downlink")
            else:
              snr=self.snr_calculator.getSNR([x,y,self.bs.height],b,lobe,"uplink")
            print(x,y,snr[0],snr[1],snr[2],snr[3],sep=",",file=f)
            snr_per_pos[x][y][c]=snr
        f.close()
        self.cells[c] = {'pos':b, 'lobe':lobe, 'id':c}
        c += 1
      i += 1
    f=open(self.out + "/cells.pkl","wb")
    pickle.dump(self.cells,f)
    f.close()
    f=open(self.out + "/snr_per_pos.pkl","wb")
    pickle.dump(snr_per_pos,f)
    f.close()
    f=open(self.out + "/map-max.out","w")
    for xa in np.arange(x_min,x_max,self.step):
      for ya in np.arange(y_min,y_max,self.step):
        (x,y) = Util.reference_position((xa,ya),self.step,self.center)
        if len(snr_per_pos[x][y]) == 0:
          continue
        max_snr = -9999
        max_cell = -1
        for c in snr_per_pos[x][y]:
          if max_cell == -1 or snr_per_pos[x][y][c][0] > max_snr:
            max_snr = snr_per_pos[x][y][c][0]
            max_cell = c
        max_sf = snr_per_pos[x][y][max_cell][1]
        max_los = snr_per_pos[x][y][max_cell][2]
        max_h = snr_per_pos[x][y][max_cell][3]

        max_snr_per_pos[x][y] = {
          'x':x,
          'y':y,
          'cell':max_cell,
          'snr':max_snr,
          'sf':max_sf,
          'los':max_los,
          'h':max_h
        }

        print(x,y,max_snr,max_cell,max_sf,max_los,max_h,sep=",",file=f)
    f.close()
    f=open(self.out + "/max_snr_per_pos.pkl","wb")
    pickle.dump(max_snr_per_pos,f)
    f.close()


    self.snr_calculator.clearObstacles()
    exprt = {
      'x_min': x_min,
      'y_min': y_min,
      'x_max': x_max,
      'y_max': y_max,
      'max_distance': max_distance,
      'snr_calculator': self.snr_calculator,
      'precision': self.step,
      'center': self.center,
      'sumo_center': self.sumo_center
    }
    fconfig = open(self.out+"/sim.config","wb")
    pickle.dump(exprt,fconfig)
    fconfig.close()

if __name__ == '__main__':

  bfile = 'mafra/osm.poly.xml'
  center = [2500,1250]

  if len(sys.argv) > 1:
    bfile = sys.argv[1]
  if len(sys.argv) > 2:
    xc = float(sys.argv[2])
    yc = float(sys.argv[3])
    center = [xc, yc]

  sim = Simulation(step=10,nBS=19,bmp=(bfile,center),out="out")
  sim.run()
  #b1 = Building([[50,20,0],[50,40,0],[90,40,0],[90,20,0]],20)
  #b3 = Building([[50,-20,0],[50,-40,0],[90,-40,0],[90,-20,0]],15)
  ##b2 = Building([[-50,20,0],[-90,20,0],[-90,40,0],[-50,40,0],[-70,1]],10)
  ##b2 = Building([[-50,20,0],[-90,20,0],[-90,40,0],[-50,40,0]],10)
  #b2 = Building([[-50,20,0],[-90,10,0],[-90,40,0],[-50,30,0]],10)
  ##b2 = Building([[-50,20,0],[-90,20,0],[-90,40,0]],10)
  #sim.addObstacle(b1)
  #sim.addObstacle(b2)
  #sim.addObstacle(b3)
