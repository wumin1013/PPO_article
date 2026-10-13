"""Publish the reviewed local artifact and update the existing project's state."""
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(r'D:\PPO_article_work')
BASE = ROOT / '论文项目/joint_policy_20260915_v1'
OUT = Path(r'D:\PPO_article_joint_runs\natural_closed_resume')
R = OUT / 'all34_dominance_research'
D = BASE / 'paper_evidence_review/verified_results/all34_original_revision'
SRC = ROOT / '论文项目/submission_package_EAAI/06_latex_source_files/latex_source_anonymous'
BUILD = OUT / 'all34_original_revision_build'
J = R / 'joint_transit_retained_seed922222_full'
F = R / 'fixed_actual_failure_recovery_full'
now = datetime.now(timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%S%fZ')

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()

def item(p):
    return {'path':str(p),'sha256':sha(p)}

def write(p, value):
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def link(p, label=None):
    return f'[{label or Path(p).name}](<{Path(p).as_posix()}>)'

audit = read(D/'final_static_audit.json')
qa = read(D/'page_visual_qa.json')
binding = read(D/'paper_result_binding.json')
research = read(D/'research_acceptance.json')
paired = read(binding['paired_assessment'])
assert audit['passed'] and research['passed'] and binding['method_frozen']
assert qa['all_pages_visually_reviewed'] and qa['pages'] == 24
assert subprocess.run(['git','branch','--show-current'],cwd=BASE,capture_output=True,text=True).stdout.strip() == 'codex/complex-path-adaptive-preview'
for p,digest in audit['verified_file_sha256'].items():
    assert sha(p) == digest, f'Artifact changed after final audit: {p}'
for name in ['manuscript_without_author_details.pdf','交付说明.md','delivery_index.json','final_acceptance.json']:
    assert not (D/name).exists(), f'Refusing to overwrite final delivery: {name}'

history = R/'state_history'/stamp
history.mkdir(parents=True,exist_ok=False)
states = [BASE/'当前执行状态.json',OUT/'live_research_summary.json',R/'active_research_checkpoint.json']
notes = [BASE/'paper_evidence_review/72维观测与逐角验收说明.md',BASE/'paper_evidence_review/18页基稿与当前实现核对清单.md']
for p in states+notes:
    shutil.copy2(p,history/p.name)
pdf = D/'manuscript_without_author_details.pdf'
shutil.copy2(BUILD/'manuscript.pdf',pdf)
assert sha(pdf) == qa['pdf']['sha256']

status_note = f'''当前有效状态（{now.isoformat()}）：最终选定 seed922222 条件候选已完成双分辨率研究验收，方法已冻结；24页英文稿已完成助手技术、数据与逐页版面核验。此状态不表示作者已经批准新版。

完整时间：Joint 444.302/444.3005 s；S2 445.185/445.1850 s；强Fixed 444.422/444.4205 s（1/0.5 ms）。两分辨率均为全部34角通过；相对S2为33角降低、1角并列，相对Fixed为14角降低、20角并列。固定前瞻收益均为120ms，严格超过100ms门槛。最终方法、轨迹、引用、原图和PDF由 {link(D/'final_acceptance.json')} 绑定。

最终PDF：{link(pdf)}。72维仍无最优性结论；三个条件搜索候选共享一个继承模型，不是独立PPO重复。以下早期阶段结论按历史时点保留，以本段和最终验收记录为当前依据。
'''
for p in notes:
    old = p.read_text(encoding='utf-8-sig')
    heading,sep,body = old.partition('\n')
    p.write_text(heading+'\n\n'+status_note+'\n---\n\n历史过程记录（保留原文）：\n'+body,encoding='utf-8')

readme = f'''# PPO_article 24页英文稿交付说明

{link(pdf,'打开新PDF')} · {link(SRC/'manuscript.tex','LaTeX正文')} · {link(SRC/'references.bib','参考文献')} · {link(D/'all34_corner_statistics.csv','全部34角数据')}

本稿已通过助手的数据、来源和版面核验；未把作者对新PDF的批准写成既成事实。当前版本为24页、九幅主图，Figure 4含续图；已逐页实际查看24页渲染图。

本稿从用户指定的18页原始匿名稿对应LaTeX正文建立修订，未以历史23/24页修订稿替代基稿。原标题、elsarticle preprint 12pt、A4和1英寸边距、章节与实验三个小节均保留。原题仍准确概括Adaptive Preview与受约束执行，因此未改标题。原前三幅图在原稿、交付图目录与当前LaTeX目录逐字节相同，坚果云18页PDF未覆盖。{link(D/'original_to_revised.diff','原稿到新稿的完整差异')}和{link(notes[1],'必要方法改动及前三图作者修改建议')}可直接核对。方法改动限于真实实现要求：72维观测及因果记忆、20ms决策与物理步长区分、三动作解码、向量KCM与共同执行器、训练和优化来源。

| 方法 | 1ms完整时间（s） | 0.5ms完整时间（s） |
|---|---:|---:|
| J-NNC | 444.302 | 444.3005 |
| S2 | 445.185 | 445.1850 |
| 强固定前瞻 | 444.422 | 444.4205 |

J-NNC相对S2节省0.883/0.8845s，相对强固定前瞻均节省0.120s。所有收益严格超过登记的max(0.1s, 5倍双方分辨率时长变化绝对值之和)门槛。0.120s为可分辨的小幅收益，不宣称数量级提升。没有缩短44.420514m任务、删角、削弱S2或提高物理上限。所有方法均完成2362有序门、单侧1.25mm欧氏允差、向量V/A/J=100/2000/62500、全制动和至少20ms物理停稳。

两种分辨率下，34个注册±10mm角点窗口的速度亏损均不高于两基线：相对S2为33角更低、1角并列；相对Fixed为14角更低、20角并列。并列使用用户确认的1e-7mm数值容差。该结论针对速度亏损，不声称每个角的最小速度、曲率或所有形状指标同时最优。

≥99mm/s完整巡航为441.615s，S2为441.238s，Fixed为440.087s。全部34角首次持续转向均向内，出角20/50mm的最大航向误差为1.957254°/0.030946°，支持提前向内及近似相切恢复。取消转向在C31越界，强制最大进给在C02越界；均为真实输出干预，未将失败前缀当作完整时间，也未把它们写成重新训练的双动作控制器。

实验仍按Experimental Settings、Main Comparative Results、Ablation Results展开，讨论沿用实验与结论中的原有结构，不另设Discussion，不讨论泛化。Figure 4标注最长连续直线2262.742mm、最短80mm、最大/最小内角及典型圆弧，并保留10mm网格。Figures 5–6采用规定的135°、90°、45°局部放大和六行机制。Figure 7保留全34角及≥99巡航，已用累计低速时间和统计表替代密集蓝色色带。Figure 8显示实际失败局部；Figure 9显示双分辨率收益及候选完整验证。

{link(notes[0],'72维观测说明')}：维数多可能增加学习难度，维数少也可能损失运动状态；目前没有同预算缩维重训练证明72维最优。本轮新增PPO更新为0，三个条件搜索候选共享继承模型与上下文，每个学习方法最终只有一个选定复合模型。论文中如实保留这些来源和历史预算差异，不制造独立种子重复或误差条。当前最终候选在完整验证后按相同时间下的巡航和航向恢复作选择，选择记录保留，不声称预注册。

24个当前引用均完成解析及已保存来源身份核验，核验范围不等于24篇文献逐篇全文重读。原Fig3保留产生的PDF版本/tag提示和文本提取CMap提示已记录，实际渲染完整。唯一underfull hbox已目视确认不影响表格，未发现裁切、重叠或未解析引用。

{link(D/'page_visual_qa.json','逐页目视记录')} · {link(D/'citation_audit.json','本轮引用核验')} · {link(D/'final_static_audit.json','数据与来源核验')} · {link(D/'final_acceptance.json','最终验收与文件哈希')}

保留原项目与codex/complex-path-adaptive-preview分支，未派子代理、未新增项目编号、未删除、提交或推送。源码修改与记录使用UTF-8无BOM，训练和中间文件仍在非同步D盘。状态历史备份：{link(history)}。旧冻结结论保留为历史，由新验收状态取代；冻结研究快照内部的旧时点goal_complete字段不回写篡改。

工作采用单一冻结数据绑定与复用绘图函数（KISS/DRY），将研究验收、论文核验及状态更新各自集中（SOLID），没有为交付新增无关训练或维度排名实验（YAGNI）。主要挑战是严格100ms门槛：旧候选精细分辨率恰好100ms而未通过；最终候选在两分辨率均取得120ms，门槛未放宽。当前要求的研究与技术交付已完成；前三图若需更改，按建议由作者执行。

工具调用简报：本地conda PPO、XeLaTeX/TeX Live、Poppler与SHA256核验；本次最终核验通过287项检查。Codex额度实时查询为已用70%、剩余30%，已有重置0，未购买或重置。Serena当前不可用，使用本地工具完成。
'''
(D/'交付说明.md').write_text(readme,encoding='utf-8')

index = {'created_utc':now.isoformat(),'project':'PPO_article','branch':'codex/complex-path-adaptive-preview',
         'pdf':item(pdf),'source':item(SRC/'manuscript.tex'),'bibliography':item(SRC/'references.bib'),
         'readme':item(D/'交付说明.md'),'figures_directory':str(D/'figures_publication'),
         'all34_data':item(D/'all34_corner_statistics.csv'),'research_binding':item(D/'paper_result_binding.json'),
         'final_acceptance':str(D/'final_acceptance.json'),'pages':24,'main_figures':9,
         'original_figures_preserved':True,'author_has_approved_this_revision':False}
write(D/'delivery_index.json',index)
bound_files = [pdf,SRC/'manuscript.tex',SRC/'references.bib',SRC/'manuscript.bbl',SRC/'elsarticle.cls',SRC/'unsrt.bst',
               D/'delivery_index.json',D/'交付说明.md',D/'all34_corner_statistics.csv',D/'paper_result_binding.json',
               D/'research_acceptance.json',D/'final_static_audit.json',D/'page_visual_qa.json',D/'citation_audit.json',
               D/'revision_origin.json',D/'original_to_revised.diff',BUILD/'compile_receipt.json',
               OUT/'audit_all34_final_manuscript.py',Path(__file__)] + notes
bound_files += sorted((D/'figures_publication').glob('*'))
bound_files += sorted((SRC/'figures_all34_revision').glob('*.pdf'))
acceptance = {'created_utc':now.isoformat(),'passed':True,'goal_complete':True,'method_frozen':True,
              'research_acceptance_passed':True,'final_manuscript_accepted':True,
              'acceptance_scope':'Assistant technical, data, citation, provenance and page-layout verification of the delivered revision; user approval is not asserted.',
              'author_has_approved_this_revision':False,'pdf':item(pdf),'pages':24,'main_figures':9,
              'first_three_figures_unchanged':True,'original_title_and_format_preserved':True,
              'baseline_pdf':binding['baseline_pdf'],'research_binding':item(D/'paper_result_binding.json'),
              'complete_times_s':audit['complete_times_s'],'time_comparisons':paired['time_comparisons'],
              'all34_comparisons':audit['all34_comparisons'],
              'page_visual_qa':item(D/'page_visual_qa.json'),'citation_audit':item(D/'citation_audit.json'),
              'static_check_count':audit['check_count'],'dimension_optimality_established':False,
              'independent_full_pipeline_ppo_replications':0,'new_ppo_updates_in_final_refinement':0,
              'frozen_snapshot_state_note':'Research snapshots retain creation-time state fields. This final acceptance and the three effective project states define current completion.',
              'source_contract_note':audit['contract_self_hash_note'],
              'history_directory':str(history),'file_sha256':{str(p):sha(p) for p in bound_files if p.is_file()},
              'remaining_required_work':[]}
write(D/'final_acceptance.json',acceptance)

mechanism = read(J/'core_mechanism_review.json')
for p in states:
    s = read(p)
    s.update({'updated_utc':now.isoformat(),'project':'PPO_article','working_directory':str(BASE),
              'branch':'codex/complex-path-adaptive-preview',
              'current':'全部34角、完整成功、双分辨率时间收益与核心机制已通过。方法已冻结，原18页稿对应源码上的24页英文论文及九主图已完成技术/数据/逐页核验并交付。',
              'goal_complete':True,'method_frozen':True,'research_acceptance_passed':True,'final_manuscript_accepted':True,
              'author_has_approved_this_revision':False,
              'final_manuscript_acceptance_scope':acceptance['acceptance_scope'],
              'authoritative_acceptance':item(D/'final_acceptance.json'),
              'final_pdf':item(pdf),'delivery_index':item(D/'delivery_index.json'),
              'complete_times_s':audit['complete_times_s'],'current_candidate_complete_times_s':audit['complete_times_s'],
              'complete_times_scope':'Final selected policies; complete physical execution at both1ms and0.5ms, including all gates, braking and physical rest.',
              'current_candidate_time_scope':'Final accepted selected policies; paired strict time and all34 tests passed.',
              'candidate':str(J),'current_candidate':str(J),'last_verified_candidate':str(J),
              'fixed':str(F),'current_fixed':str(F),'current_paper_binding':str(D/'paper_result_binding.json'),
              'current_candidate_bindings':binding['primary_methods'],
              'current_candidate_all34_passed':True,'current_candidate_all34_nominal_passed':True,
              'current_candidate_resolution_verified':True,'current_quantitative_core_passed':True,
              'current_time_comparisons':paired['time_comparisons'],
              'current_candidate_nominal_mechanism':{'status':'Passed for the selected frozen candidate, including actual visual inspection and two completed interventions.',
                  'review':item(J/'core_mechanism_review.json'),'geometry_audit':item(J/'actual_turn_geometry/audit.json'),
                  'intervention_audit':item(J/'channel_interventions/mechanism_audit.json')},
              'current_acceptance':{'all34':audit['all34_comparisons'],'complete_success':True,'strict_time_margin':True,'mechanism_passed':True,'final_pdf_verified':True},
              'current_search_repeatability':'Three conditional-search candidates share inherited model and contexts, all with complete two-resolution executions. Not independent PPO training replications.',
              'new_ppo_updates':0,'independent_full_pipeline_ppo_seeds':0,'conditional_search_seeds':3,
              'observation_dimension':72,'dimension_optimality_established':False,
              'active_sessions':{},'active_validation_sessions':{},'running_candidate':None,
              'next_steps':[],'next_active_research':[],'next_research_if_requested':[],
              'revision_origin':item(D/'revision_origin.json'),'latest_revision_history':str(history),'state_history':str(history)})
    s.setdefault('historical_records',{})['before_final_all34_delivery'] = item(history/p.name)
    if p == OUT/'live_research_summary.json':
        s['status_source'] = str(BASE/'当前执行状态.json')
    write(p,s)

# Check final copies and written active state; no historical artifact is deleted.
for p,digest in acceptance['file_sha256'].items():
    assert sha(p) == digest, p
for p in states:
    s = read(p)
    assert s['goal_complete'] and s['final_manuscript_accepted'] and s['complete_times_s'] == audit['complete_times_s']
    assert s['authoritative_acceptance']['sha256'] == sha(D/'final_acceptance.json')
assert sha(Path(binding['baseline_pdf']['file'])) == binding['baseline_pdf']['file_sha256']
print(json.dumps({'passed':True,'pdf':item(pdf),'acceptance':item(D/'final_acceptance.json'),
                  'history':str(history),'pages':24,'all34_passed':True,'updated_states':[str(p) for p in states]},ensure_ascii=False,indent=2))
