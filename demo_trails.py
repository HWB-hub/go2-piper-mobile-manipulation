"""Non-physical visual trails for the target ground projection and actual base path."""
import numpy as np
from pxr import Gf, UsdGeom, Vt


def target_positions(parameters, duration):
    t = np.linspace(0.,duration,240)
    u = np.maximum(t-2.,0.)
    tau = np.where(u<2.,u*u/4.,u-1.)
    w = parameters["omega"]
    a = tau*w
    ax,ay,az = parameters["amplitude"]
    px,py,pz = parameters["phase"]
    v,m = parameters["speed"],parameters["modulation"]
    x = v*(tau+m*np.sin(a)/w)
    y = ay*(np.sin(a+py)-np.sin(py))
    if parameters["kind"] in (1,2):
        x = ax*(np.sin(a+px)-np.sin(px))
    if parameters["kind"]==1:
        y = ay*(np.cos(px)-np.cos(a+px))
    if parameters["kind"]==2:
        y = ay*np.sin(2*a)
    z = az*np.sin(.7*a+pz)
    return np.array(parameters["start_w"])+np.stack((x,y,z),-1)


class DemoTrails:
    def __init__(self, stage, parameters, duration):
        self.reference = self._curve(stage,"/World/TargetGroundProjection",(.88,.45,.12))
        points = target_positions(parameters,duration)
        points[:,2] = .013
        self._set_points(self.reference,points)
        self.actual = self._curve(stage,"/World/ActualBasePath",(.08,.43,.80))
        self.points = []

    @staticmethod
    def _curve(stage,path,color):
        curve = UsdGeom.BasisCurves.Define(stage,path)
        curve.CreateTypeAttr("linear")
        curve.CreateWrapAttr("nonperiodic")
        curve.CreateWidthsAttr([.018])
        curve.SetWidthsInterpolation("constant")
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        return curve

    @staticmethod
    def _set_points(curve,points):
        curve.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in points]))
        curve.GetCurveVertexCountsAttr().Set([len(points)])

    def update(self,base):
        self.points.append((float(base[0]),float(base[1]),.019))
        if len(self.points)>1:
            self._set_points(self.actual,self.points)
