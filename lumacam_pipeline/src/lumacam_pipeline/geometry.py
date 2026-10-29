"""Estimate a fixed-diameter electrode centre from visible boundary contrast."""
import numpy as np


def estimate_electrode_center(images, roi, *, pixel_size_um, diameter_mm, max_shift_px=40, step_px=2):
    """Fit a known-radius circle to temporal edge contrast, never to bright area.

    Only registration uses spatial smoothing. Incomplete/ambiguous boundaries
    return a candidate with accepted=False; they do not redefine the ROI.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates, median_filter
    size, diameter = float(pixel_size_um), float(diameter_mm)
    if not np.isfinite([size,diameter,max_shift_px,step_px]).all() or min(size,diameter,max_shift_px,step_px)<=0:
        raise ValueError('Known pixel size, electrode diameter and positive search parameters are required.')
    radius=diameter*1000/(2*size)
    x0,y0=float(roi['x0']),float(roi['y0'])
    indices=sorted(set(np.linspace(0,len(images)-1,min(15,len(images)),dtype=int)))
    selected=np.asarray([images[i] for i in indices],dtype=float)
    valid=np.isfinite(selected).all(axis=(1,2))
    selected=selected[valid]
    if len(selected)<2: raise ValueError('Centre estimation needs at least two finite images with temporal contrast.')
    contrast=gaussian_filter(median_filter(np.percentile(selected,85,axis=0)-np.percentile(selected,10,axis=0),size=3),2)
    contrast=np.maximum(contrast,0)
    nonzero=contrast[contrast>0]
    cap=float(np.percentile(nonzero,99)) if len(nonzero) else 0.
    angle=np.linspace(0,2*np.pi,180,endpoint=False);cx,sy=np.cos(angle),np.sin(angle)
    candidates=[]
    band=min(4.,radius/4)
    for y in np.arange(y0-max_shift_px,y0+max_shift_px+.01,step_px):
        for x in np.arange(x0-max_shift_px,x0+max_shift_px+.01,step_px):
            if x-radius<0 or y-radius<0 or x+radius>=contrast.shape[1] or y+radius>=contrast.shape[0]: continue
            inside=map_coordinates(contrast,[y+(radius-band)*sy,x+(radius-band)*cx],order=1,mode='nearest')
            outside=map_coordinates(contrast,[y+(radius+band)*sy,x+(radius+band)*cx],order=1,mode='nearest')
            edge=inside-outside
            score=float(np.clip(edge,0,cap).mean())
            support=float(np.mean(edge>.1*cap)) if cap>0 else 0.
            candidates.append((score,x,y,support))
    if not candidates: raise ValueError('The fixed-radius search does not fit inside the image.')
    score,x,y,support=max(candidates)
    shift=float(np.hypot(x-x0,y-y0))
    accepted=bool(cap>0 and support>=.35 and shift<=.75*max_shift_px)
    candidate={'x0':float(x),'y0':float(y),'r0':float(radius)}
    retained={'x0':x0,'y0':y0,'r0':float(radius)}
    return {'method':'fixed known radius; temporal boundary contrast', 'diameter_mm':diameter,
            'pixel_size_um':size,'initial_roi':dict(roi),'candidate_roi':candidate,
            'roi':candidate if accepted else retained,'accepted':accepted,
            'arc_support_fraction':support,'shift_px':shift,'search_radius_px':float(max_shift_px),
            'grid_step_px':float(step_px),'score':score,
            'reason':'Image-supported centre estimate; verify diagnostic overlay.' if accepted else
                     'Insufficient boundary support or candidate near search limit; historical centre retained.',
            'note':'Acceptance is a registration heuristic, not independent electrode-boundary confirmation. Science images are unchanged.'}
