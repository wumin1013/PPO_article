"""Analytic geometric examples only; never imported by a controller or trainer."""
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.integrate import cumulative_simpson

from study_io import register, receipt, write_json


def segment_distance(points, a, b):
    direction = b - a
    t = np.clip((points-a) @ direction / (direction @ direction), 0., 1.)
    return np.linalg.norm(points-a-t[..., None]*direction, axis=-1)


def make_curve(degrees, maximum_deviation=1.10):
    theta = np.deg2rad(degrees)
    u = np.linspace(0., 1., 20001)
    heading = theta * (u - np.sin(2*np.pi*u)/(2*np.pi))
    integral = cumulative_simpson(np.column_stack((np.cos(heading), np.sin(heading))),
                                  x=u, axis=0, initial=0.)
    target = np.array([1+np.cos(theta), np.sin(theta)])
    length = (target @ integral[-1]) / (integral[-1] @ integral[-1])
    xy = np.array([-1., 0.]) + length*integral
    normal_out = np.array([-np.sin(theta), np.cos(theta)])
    deviation = np.minimum(xy[:, 1], xy @ normal_out)
    d = maximum_deviation/deviation.max()
    length *= d
    xy *= d
    curvature = theta/length*(1-np.cos(2*np.pi*u))
    entry = np.array([-d, 0.])
    outgoing = np.array([np.cos(theta), np.sin(theta)])
    exit_point = d*outgoing
    extension = max(3., d*.65)
    reference = np.array([[-d-extension, 0.], [0., 0.], (d+extension)*outgoing])
    full_xy = np.vstack((reference[0], xy, reference[-1]))
    euclidean = np.minimum(segment_distance(full_xy, reference[0], reference[1]),
                           segment_distance(full_xy, reference[1], reference[2]))
    sides = np.column_stack((full_xy[:, 1], full_xy @ normal_out))
    checks = dict(turn_deg=degrees, entry_distance_to_vertex_mm=float(d),
                  exit_distance_to_vertex_mm=float(d), curve_length_mm=float(length),
                  max_euclidean_deviation_mm=float(euclidean.max()),
                  min_distance_to_vertex_mm=float(np.linalg.norm(xy, axis=1).min()),
                  max_curvature_mm_inv=float(curvature.max()),
                  endpoint_position_error_mm=float(np.linalg.norm(xy[-1]-exit_point)),
                  endpoint_heading_error_deg=[float(np.rad2deg(heading[0])),
                                               float(np.rad2deg(heading[-1]-theta))],
                  endpoint_curvature_mm_inv=curvature[[0,-1]].tolist(),
                  min_heading_increment_rad=float(np.diff(heading).min()),
                  outward_excursion_mm=float(max(0., -sides.min())),
                  heading_overshoot_deg=float(max(0., np.rad2deg(heading.max()-theta))),
                  dynamic_feasibility_tested=False, learned_policy=False)
    assert checks['endpoint_position_error_mm'] < 1e-9
    assert checks['max_euclidean_deviation_mm'] <= 1.25
    assert checks['outward_excursion_mm'] < 1e-9
    assert np.all(np.diff(heading) >= 0.) and np.all(curvature >= 0.)
    return dict(u=u, xy=xy, full_xy=full_xy, reference=reference,
                heading=heading, curvature=curvature, length=length,
                entry=entry, exit=exit_point, checks=checks)


def plot_examples(curves, stage):
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'font.size':10, 'axes.unicode_minus':False,
        'axes.spines.top':False, 'axes.spines.right':False,
        'svg.fonttype':'none', 'pdf.fonttype':42, 'axes.linewidth':.7})
    fig = plt.figure(figsize=(13.2,8.2), facecolor='white')
    grid = fig.add_gridspec(2, 6, height_ratios=(2.5,1),
                           left=.065,right=.98,top=.87,bottom=.17,hspace=.53,wspace=.72)
    colors = ['#286693','#C2703B','#6D6093']
    for i,curve in enumerate(curves):
        ax = fig.add_subplot(grid[0, 2*i:2*i+2])
        angle = curve['heading'][-1]
        rotation = np.array([[np.cos(-angle/2), -np.sin(-angle/2)],
                             [np.sin(-angle/2), np.cos(-angle/2)]])
        xy, reference = curve['full_xy']@rotation.T, curve['reference']@rotation.T
        lower = np.minimum(xy.min(axis=0),reference.min(axis=0))-1.8
        upper = np.maximum(xy.max(axis=0),reference.max(axis=0))+1.8
        gx,gy = np.meshgrid(np.linspace(lower[0],upper[0],400),
                            np.linspace(lower[1],upper[1],400))
        points = np.stack((gx,gy),axis=-1)
        dist = np.minimum(segment_distance(points,reference[0],reference[1]),
                          segment_distance(points,reference[1],reference[2]))
        ax.contourf(gx,gy,dist,levels=[0.,1.25],colors=['#E9EEF2'])
        ax.contour(gx,gy,dist,levels=[1.25],colors=['#ADB9C3'],linewidths=.9,linestyles='--')
        ax.plot(*reference.T, '--', color='#77818A',lw=1.3)
        ax.plot(*xy.T, color=colors[i],lw=2.6)
        ax.scatter(0,0,s=20,color='#54616C',zorder=5)
        ax.annotate('尖顶', (0,0), xytext=(0,-14),textcoords='offset points',ha='center',color='#54616C')
        for letter, point, shift in [('A',curve['entry'],(-9,9)),('B',curve['exit'],(8,9))]:
            point = point@rotation.T
            ax.scatter(*point,s=30,color=colors[i],zorder=6)
            ax.annotate(letter,point,xytext=shift,textcoords='offset points',weight='bold',color=colors[i])
        for fraction in (.25,.75):
            j = int(fraction*(len(curve['xy'])-1))
            px = curve['xy']@rotation.T
            ax.annotate('',px[j+220],px[j-220],arrowprops=dict(arrowstyle='-|>',color=colors[i],lw=1.7))
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlim(lower[0],upper[0]);ax.set_ylim(lower[1],upper[1])
        ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        ax.set_title(f"转向 {curve['checks']['turn_deg']}°\n距尖顶最近 {curve['checks']['min_distance_to_vertex_mm']:.2f} mm",
                     fontsize=11,pad=13)
        ax.tick_params(labelsize=9)
    ah = fig.add_subplot(grid[1,:3]);ak = fig.add_subplot(grid[1,3:])
    for color,curve in zip(colors,curves):
        label=f"{curve['checks']['turn_deg']}°"
        ah.plot(curve['u'],np.rad2deg(curve['heading']),color=color,lw=2,label=label)
        ak.plot(curve['u'],curve['curvature'],color=color,lw=2,label=label)
    ah.set(xlabel='过渡段归一化弧长 u',ylabel='相对入边航向 (°)',xlim=(0,1),ylim=(-4,157))
    ak.set(xlabel='过渡段归一化弧长 u',ylabel=r'曲率 ($\mathrm{mm}^{-1}$)',xlim=(0,1),ylim=(-.025,.88))
    ah.set_title('航向单调到达出边方向，无反向、无超调',fontsize=11,loc='left')
    ak.set_title('曲率由零平滑升起，再平滑回到零',fontsize=11,loc='left')
    ah.legend(frameon=False,ncol=3,fontsize=9,loc='upper left')
    for ax in (ah,ak):ax.grid(axis='y',color='#DDE3E7',lw=.6);ax.set_axisbelow(True)
    fig.suptitle('理想过弯的几何示例：两端相切，连续内转，平顺接出',fontsize=17,y=.98)
    fig.text(.5,.925,'A：从入边相切进入  ·  B：与出边相切接出  ·  三条曲线最大欧氏偏差均为 1.10 mm',ha='center',fontsize=11)
    fig.legend(handles=[Line2D([0],[0],color='#286693',lw=2.5,label='构造的几何过渡'),
                        Line2D([0],[0],color='#77818A',lw=1.4,ls='--',label='原始折线'),
                        Patch(facecolor='#E9EEF2',edgecolor='#ADB9C3',label='1.25 mm 欧氏允差带')],
               loc='lower center',bbox_to_anchor=(.5,.046),ncol=3,frameon=False)
    fig.text(.5,.018,'各 XY 图等比例显示；仅为几何示例，不代表学得策略、动态可行性或最短时间解。',ha='center',color='#66717B',fontsize=9)
    for ext in ('png','svg','pdf'):
        fig.savefig(stage/f'ideal_tangent_turns.{ext}',dpi=180 if ext=='png' else 300)
    plt.close(fig)


def main():
    started=time.monotonic()
    stage=register('tangent_turn_geometry_example_20260916',[__file__],
        dict(training_decisions=0,physical_ticks=0,wall_s=120),
        dict(question='Geometric interpretation of tangent entry/exit without reverse steering or exit overshoot.',
             construction='psi(u)=theta*(u-sin(2*pi*u)/(2*pi)); integrate unit tangent, scale to max distance1.10mm.',
             boundary='No controller, reward, training or physical constraint changes.',
             figure_contract='Exploratory schematic and analytic profiles, Python only; supplied geometric intent is the visual reference, not a fitted template or submission figure.'))
    curves=[make_curve(deg) for deg in (60,90,150)]
    plot_examples(curves,stage)
    arrays={f'{key}_{i}':curve[key] for i,curve in enumerate(curves)
            for key in ('u','xy','full_xy','reference','heading','curvature')}
    np.savez_compressed(stage/'curves.npz',**arrays)
    write_json(stage/'geometry_checks.json',[curve['checks'] for curve in curves])
    receipt(stage,dict(status='completed',geometry_checks_passed=True,
        wall_s=time.monotonic()-started,training_decisions=0,physical_ticks=0,
        learned_policy=False,dynamic_feasibility_tested=False,
        unique_curve_claim=False,visual_review='pending'))
    print(json.dumps(dict(stage=str(stage),checks=[c['checks'] for c in curves]),ensure_ascii=False))


if __name__=='__main__':main()
