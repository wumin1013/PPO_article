"""Curvature diagnostics from raw physical velocity and acceleration, without XY resampling."""
import numpy as np


def intrinsic_metrics(rows):
    speed=np.linalg.norm(rows[:,2:4],axis=1)
    moving=speed>5.  # Existing physical-heading validity threshold.
    cross=rows[:,2]*rows[:,5]-rows[:,3]*rows[:,4]
    curvature=np.zeros_like(speed);curvature[moving]=cross[moving]/speed[moving]**3
    peak=int(np.argmax(abs(curvature)))
    return dict(peak_curvature_mm_inv=float(abs(curvature[peak])),peak_ms=peak,
        bending_integral_mm_inv=float(np.sum(curvature[moving]**2*speed[moving])*.001),
        valid_speed_gt_mm_s=5.,valid_duration_s=float(moving.sum()*.001))


def main():
    import argparse
    from study_io import BASE,write_json
    from search_complete_return import read
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True)
    args=parser.parse_args();stage=BASE/args.stage
    source=read(stage/'raw_result_audit.json');records=[]
    for row in source['changed_metrics']:
        with np.load(row['before']['trace']) as z:before=intrinsic_metrics(z['rows'])
        with np.load(row['after']['trace']) as z:after=intrinsic_metrics(z['rows'])
        r=dict(path=row['path'],before=before,after=after,
            peak_change=after['peak_curvature_mm_inv']-before['peak_curvature_mm_inv'],
            energy_change=after['bending_integral_mm_inv']-before['bending_integral_mm_inv'])
        records.append(r)
        print('INTRINSIC_CURVE',row['path'],r['peak_change'],r['energy_change'],before['peak_ms'],after['peak_ms'],flush=True)
    write_json(stage/'intrinsic_curve_audit.json',dict(records=records,new_physical_runs=0,
        original_acceptance_unchanged=True,interpretation='Raw1ms V/A curvature at the existing '
        'speed>5 heading-validity threshold. Independent of spatial resampling; this does '
        'not replace full XY, spatial curvature, low-speed/start/stop or physical checks.'))


if __name__=='__main__':main()
