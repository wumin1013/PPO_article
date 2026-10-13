"""COPP adapter v4c: unchanged S2 geometry without unused cruise-cap solving.

The existing geometry builder is reused; only its final container is replaced.
COPP does not consume S2 curve-constant cruise caps. This removes that unrelated
enclosure prerequisite without changing a control point, trim, or tolerance.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import copp_path_adapter_v4 as adapter
from copp_time_projection_v4 import speed_peak
import baselines.legacy_s2 as legacy

ROOT=adapter.ROOT
_prepare=legacy.prepare_geometry
_sample=adapter.sample


def prepare_geometry_only(reference,**kwargs):
    module_factory=legacy._module

    def factory(name,path):
        module=module_factory(name,path)
        if name=="jmp_legacy_s2_geometry":
            def container(pieces,item,ref,*,settings):
                # This is exactly the old final-container merging rule.
                merged=[]
                for piece in pieces:
                    if piece.length<=0:continue
                    if merged and isinstance(piece,module.Line) and isinstance(merged[-1],module.Line):
                        prev=merged[-1]
                        if np.linalg.norm(prev.end-piece.start)>module.MATCH_TOLERANCE_MM or np.linalg.norm(prev.tangent-piece.tangent)>1e-10:
                            raise ValueError("Noncollinear remnants cannot be merged")
                        merged[-1]=module.Line(prev.start,piece.end,prev.source_start,piece.source_end)
                    else:merged.append(piece)
                if not merged or not isinstance(merged[0],module.Line) or not isinstance(merged[-1],module.Line):
                    raise ValueError("Original S2 entry/exit straight prerequisites retained")
                for left,right in zip(merged[:-1],merged[1:]):
                    if np.linalg.norm(left.position([left.length])[0]-right.position([0.])[0])>module.MATCH_TOLERANCE_MM or abs(left.source_end-right.source_start)>module.MATCH_TOLERANCE_MM:
                        raise ValueError("Geometric/source interval discontinuity")
                curve=module.Compound(merged,metadata=dict(path_id=item["id"],closed=item["closed"],
                    source_reference_sha256=item["reference_sha256"],
                    source_order_semantics="One monotone complete source traversal",
                    geometry_only=True,unused_curve_constant_caps_computed=False,
                    geometry_settings={key:settings[key] for key in ["corner_error_mm","fit","inversion_tolerance_mm"]}))
                curve.reference=ref
                if np.linalg.norm(curve.position([0.])[0]-ref[0])>module.MATCH_TOLERANCE_MM or np.linalg.norm(curve.position([curve.length])[0]-ref[-1])>module.MATCH_TOLERANCE_MM:
                    raise ValueError("Original finite traversal endpoints changed")
                return curve
            module.GeometryPath=container
        return module

    legacy._module=factory
    try:return _prepare(reference,**kwargs)
    finally:legacy._module=module_factory


def projected_sample(curve,s,profile,dt,settings):
    peak=speed_peak(s,profile.a,profile.b,profile.num_stationary)*settings["spatial_scale_mm"]
    dilation=1.01*max(1.,peak/(100.*settings["velocity_reserve"]))
    scaled=adapter.copp.Profile3rd(profile.a/dilation**2,profile.b/dilation**2,profile.num_stationary)
    pos,t,progress,meta=_sample(curve,s,scaled,dt,settings)
    meta.update(time_dilation=dilation,raw_interpolated_peak_speed_mm_s=peak,
        adaptation="Exact interval-speed projection and fixed additional 1% time reserve",
        wrapper_sha256=adapter.sha(__file__))
    return pos,t,progress,meta


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--case",default="all",
        choices=["all","dense","validation_trapezoid","validation_zigzag","train_sine"])
    parser.add_argument("--output",type=Path,default=ROOT/"results/copp_development_v4c")
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    adapter.prepare_geometry=prepare_geometry_only
    adapter.sample=projected_sample
    names=["dense","validation_trapezoid","validation_zigzag","train_sine"]
    if args.case!="all":names=[args.case]
    cases=[]
    for name in names:
        source=ROOT/"baseline_records/supplementary_reproduction/cases/dense" if name=="dense" else ROOT/"data"
        path=source/("reference.csv" if name=="dense" else name+".csv")
        geometry=json.loads((source/"geometry.json").read_text("utf-8")) if name=="dense" else None
        cases.append((name,path,geometry))
    registration=dict(version="v4c-geometry-only-and-time-projection",settings=adapter.SETTINGS,
        source_sha256=adapter.sha(__file__),core_sha256=adapter.sha(adapter.__file__),
        speed_projection_sha256=adapter.sha(ROOT/"src/copp_time_projection_v4.py"),
        input_sha256={name:adapter.sha(path) for name,path,_ in cases},
        adaptation="Retain S2 control points and trims; skip unused constant-cruise cap computation; exact interval speed projection plus fixed 1% time reserve.",
        final_iteration_rule="Fixed third iteration; all intermediate trajectories/failures retained",
        final_test_paths_executed=0,command_periods_s=[.001,.0005],limits=[100.,2000.,62500.])
    reg=args.output/("registration_"+args.case+".json")
    if reg.exists():raise FileExistsError("Preserve previous evidence")
    adapter.write(reg,registration)
    records=[]
    for name,path,geometry in cases:
        ref=np.loadtxt(path,delimiter=",",skiprows=1)
        chord=max((p.get("chord_error_bound_mm",0.) for p in geometry["primitives"]),default=0.) if geometry else 0.
        records.append(adapter.run_case(name,ref,1.25-chord,geometry,args.output))
    adapter.write(args.output/("summary_"+args.case+".json"),records)


if __name__=="__main__":main()
