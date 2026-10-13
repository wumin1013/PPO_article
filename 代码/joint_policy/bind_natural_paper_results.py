"""Freeze completed evidence and generate the only numerical input to the manuscript."""
import json,re,shutil
from datetime import datetime,timezone
import numpy as np
from study_io import BASE,sha,write_json
from natural_case_metrics import metrics
from natural_case_audit import analytic_command_audit


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    case=BASE/'paper_evidence_review/natural_closed_case';paper=BASE/'manuscript_joint_v1'
    geometry=read(case/'geometry.json');half=read(case/'resolution_half_ms/result.json')
    assert half['success']
    mainr=next(r for r in read(case/'support_calibration/results.json') if r['success'])
    maintrace=case/'support_calibration/remaining_median.npz'
    z=np.load(maintrace);mainr['analytic']=analytic_command_audit(z['reference'][0],z['rows'],geometry)
    assert mainr['analytic']['analytic_corridor_pass']
    write_json(case/'support_calibration/primary_result.json',mainr)
    fixed=[dict(label='Historical D=3, original',result=case/'fixed_historical/result.json',trace=case/'fixed_historical/trace.npz',D=3.,role='independent historical PPO model')]
    for r in read(case/'fixed_support_calibration/results.json'):
        label=r['spec']['label'];p=case/'fixed_support_calibration'/label
        # The complete result includes analytic audit, which the raw evaluator JSON lacks.
        write_json(p.with_name(label+'_complete_result.json'),r)
        fixed.append(dict(label='Historical D=3, '+label,result=p.with_name(label+'_complete_result.json'),trace=p.with_suffix('.npz'),D=3.,role='historical PPO plus support calibration'))
    r=read(case/'fixed_conditional_training/full_result.json');spec=read(case/'fixed_conditional_training/selected_policy.json')
    fixed.append(dict(label=f"Conditional D={spec['fixed_d']:g}",result=case/'fixed_conditional_training/full_result.json',trace=case/'fixed_conditional_training/full_trace.npz',D=spec['fixed_d'],role='same-backbone conditional training and global-distance selection'))
    eligible=[f for f in fixed if read(f['result'])['success']]
    best=min(eligible,key=lambda f:read(f['result'])['complete_time_s'])
    sources={'Joint':dict(result=case/'support_calibration/primary_result.json',trace=maintrace),
        'S2':dict(result=case/'s2_complete/result.json',trace=case/'s2_complete/trace.npz'),'Fixed':best}
    binding=dict(created_utc=datetime.now(timezone.utc).isoformat(),primary_methods={},
        primary_fixed_label=best['label'],primary_fixed_D_mm=best['D'],fixed_opportunities=[],
        reference_sha256=sha(case/'reference.csv'),geometry_sha256=sha(case/'geometry.json'),
        resolution=half,scope=dict(N_path=1,main_models=1,independent_full_pipeline_replications=0,
                                 data_role='disclosed development case',sealed_data_used=False))
    for name,entry in sources.items():
        result=read(entry['result']);assert result['success']
        z=np.load(entry['trace']);m=metrics(z['reference'],z['rows'],geometry)
        assert m['all34_windows_complete'] and m['projection_raw_backward_steps']==0
        target=case/(name.lower()+'_paper_metrics.json');write_json(target,m)
        binding['primary_methods'][name]=dict(trace=str(entry['trace'].relative_to(BASE)),trace_sha256=sha(entry['trace']),
            result=str(entry['result'].relative_to(BASE)),result_sha256=sha(entry['result']),
            metrics=str(target.relative_to(BASE)),metrics_sha256=sha(target))
    for f in fixed:
        r=read(f['result']);binding['fixed_opportunities'].append(dict(label=f['label'],D=f['D'],role=f['role'],
            success=r['success'],complete_time_s=r['complete_time_s'],result=str(f['result'].relative_to(BASE)),result_sha256=sha(f['result'])))
    write_json(case/'paper_result_binding.json',binding)
    mm={k:read(BASE/v['metrics']) for k,v in binding['primary_methods'].items()}
    rr={k:read(BASE/v['result']) for k,v in binding['primary_methods'].items()}
    t=rr['Joint']['complete_time_s'];tf=rr['Fixed']['complete_time_s'];ts=rr['S2']['complete_time_s']
    gain=100*(tf-t)/tf
    text=f"It is {abs(gain):.3f}\\% {'faster' if gain>0 else 'slower'} than the strongest complete fixed-preview control ({tf:.3f} s), and {t-ts:.3f} s slower than S2 ({ts:.3f} s)."
    cm=mm['Joint']['cruise']['99'];cf=mm['Fixed']['cruise']['99'];cs=mm['S2']['cruise']['99']
    deficit=mm['Joint']['corner_total_speed_deficit_mm'];df=mm['Fixed']['corner_total_speed_deficit_mm'];ds=mm['S2']['corner_total_speed_deficit_mm']
    commands=dict(MainTime=f'{t:.3f}',FixedTime=f'{tf:.3f}',S2Time=f'{ts:.3f}',MainGainFixed=f'{gain:.3f}',
        FixedD=f"{best['D']:g}",FixedLabel=best['label'],AbstractComparison=text,
        MainCruise=f"{cm['time_s']:.3f}",MainCruisePct=f"{100*cm['time_fraction']:.3f}",
        MainCoverage=f"{100*cm['distance_fraction']:.3f}",MainLongest=f"{cm['longest_time_s']:.3f}",
        MainLongestDistance=f"{cm['longest_distance_mm']/1000:.3f}",FixedCoverage=f"{100*cf['distance_fraction']:.3f}",
        FixedCruise=f"{cf['time_s']:.3f}",FixedLongest=f"{cf['longest_time_s']:.3f}",
        S2Coverage=f"{100*cs['distance_fraction']:.3f}",S2Cruise=f"{cs['time_s']:.3f}",S2Longest=f"{cs['longest_time_s']:.3f}",
        MainDeficit=f'{deficit:.3f}',FixedDeficit=f'{df:.3f}',S2Deficit=f'{ds:.3f}',
        MainBelow=f"{mm['Joint']['corner_total_below95_s']:.3f}",FixedBelow=f"{mm['Fixed']['corner_total_below95_s']:.3f}",
        S2Below=f"{mm['S2']['corner_total_below95_s']:.3f}",HalfTime=f"{half['complete_time_s']:.3f}",
        MainGeometry=f"{mainr['analytic']['maximum_evaluated_distance_mm']:.6f}",
        HalfGeometry=f"{half['analytic']['maximum_evaluated_distance_mm']:.6f}",
        ConditionalTime=f"{r['complete_time_s']:.3f}" if r['success'] else 'not completed')
    # Use the actual conditional result, independent of the last loop variable.
    conditional=read(case/'fixed_conditional_training/full_result.json')
    commands['ConditionalTime']=f"{conditional['complete_time_s']:.3f}" if conditional['success'] else 'not completed'
    commands['CornerComparisonText']=(f'The total corner speed deficit is {deficit:.3f} mm for Joint, {df:.3f} mm for the selected fixed control, and {ds:.3f} mm for S2. '
        f'Time below 95 mm/s is {mm["Joint"]["corner_total_below95_s"]:.3f}, {mm["Fixed"]["corner_total_below95_s"]:.3f}, and {mm["S2"]["corner_total_below95_s"]:.3f} s, respectively. These measures need not give the same ordering.')
    commands['CruiseComparisonText']=(f'High-speed distance coverage is {100*cm["distance_fraction"]:.3f}\\% for Joint, {100*cf["distance_fraction"]:.3f}\\% for the selected fixed control, and {100*cs["distance_fraction"]:.3f}\\% for S2. The longest corresponding intervals are {cm["longest_time_s"]:.3f}, {cf["longest_time_s"]:.3f}, and {cs["longest_time_s"]:.3f} s.')
    table=[]
    for cid in geometry['mechanism_columns']:
        for name in sources:
            c=next(x for x in mm[name]['corners'] if x['id']==cid)
            table.append(f"{c['included_angle_deg']:.0f} & {name} & {c['minimum_speed_mm_s']:.2f} & {c['mean_speed_mm_s']:.2f} & {c['reverse_heading_deg']:.2f} & {c['outgoing_overshoot_mm']:.3f} " + chr(92)*2)
    commands['RepresentativeRows']='\n'.join(table)
    (paper/'natural_results.tex').write_text('% Generated exclusively from paper_result_binding.json.\n'+
        '\n'.join('\\newcommand{\\'+k.replace('S2','STwo')+'}{'+v+'}' for k,v in commands.items())+'\n',encoding='utf-8')
    freeze=dict(created_utc=datetime.now(timezone.utc).isoformat(),method_frozen=True,
        checkpoint='canonical_gate_band_worst_guard_joint_learning_20260920/shadow_training_policy.pt',
        checkpoint_sha256='25abeb3b88e45efac1d8311669f45a3fcaa26dc785d9bfbd216734b7f5a9c173',
        support_transform=mainr['spec'],decision_ms=20,execution_ms=1,resolution_check_ms=.5,
        paper_result_binding_sha256=sha(case/'paper_result_binding.json'),no_more_soft_metric_optimization=True,
        independent_training_repetition_unestablished=True,goal_complete=False,remaining='manuscript,figure and PDF QA')
    write_json(case/'method_freeze.json',freeze)
    print('FROZEN',best['label'],tf,'main',t,'gain_percent',gain,'S2',ts)
    print(commands['CornerComparisonText']);print(commands['CruiseComparisonText'])


if __name__=='__main__':main()
