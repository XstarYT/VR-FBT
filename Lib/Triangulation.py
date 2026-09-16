"""Small-view consensus triangulation with confidence and baseline checks."""
import itertools
import math
import numpy as np


def triangulate_consensus(views, projections, camera_positions, inlier_px=12.0):
    """Return point/confidence/error using at least two agreeing cameras.

    Test all camera pairs and the full set (at most three cameras). Prefer
    support from more views, then confident rays with a useful baseline.
    Conflicting high-confidence pixels cannot pull all views into one average.
    """
    if len(views) < 2:
        return None

    def solve(selected):
        rows=[]
        for source,u,v,visibility in selected:
            p=projections[source];weight=math.sqrt(visibility)
            rows.extend((weight*(u*p[2]-p[0]),weight*(v*p[2]-p[1])))
        _,_,vt=np.linalg.svd(rows)
        return None if abs(vt[-1,3])<1e-8 else vt[-1,:3]/vt[-1,3]

    def supporters(point):
        result=[]
        for view in views:
            source,u,v,_=view;p=projections[source]@np.append(point,1.)
            if p[2]<=1e-8:continue
            error=math.hypot(p[0]/p[2]-u,p[1]/p[2]-v)
            if error<=inlier_px:result.append((view,error))
        return result

    def geometry(point, selected):
        best=0.
        for first,second in itertools.combinations(selected,2):
            a=point-camera_positions[first[0]];b=point-camera_positions[second[0]]
            denominator=np.linalg.norm(a)*np.linalg.norm(b)
            if denominator<=1e-8:continue
            cosine=float(np.clip(np.dot(a,b)/denominator,-1,1))
            best=max(best,math.sqrt(max(0,1-cosine*cosine)))
        return best

    candidates=[views,*itertools.combinations(views,2)] if len(views)>2 else [views]
    winner=None
    try:
        for candidate in candidates:
            point=solve(candidate)
            if point is None or not np.isfinite(point).all():continue
            inliers=supporters(point)
            if len(inliers)<2:continue
            selected=[item[0] for item in inliers]
            baseline=geometry(point,selected)
            if baseline<math.sin(math.radians(3)):continue
            confidence=sum(view[3] for view in selected)/len(selected)
            error=sum(item[1] for item in inliers)/len(inliers)
            score=(len(inliers),confidence*baseline,-error)
            if winner is None or score>winner[0]:winner=(score,selected)
        if winner is None:return None
        point=solve(winner[1])
        if point is None:return None
        inliers=supporters(point)
        if len(inliers)<2:return None
        selected=[item[0] for item in inliers]
        error=sum(item[1] for item in inliers)/len(inliers)
        baseline=geometry(point,selected)
        if baseline<math.sin(math.radians(3)):return None
        confidence=(sum(view[3] for view in selected)/len(selected))*min(1.,baseline/math.sin(math.radians(20)))*max(.25,1-error/50)
        return point,min(1.,confidence),error
    except (ValueError,np.linalg.LinAlgError):
        return None
