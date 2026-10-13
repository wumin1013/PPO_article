const fs = require('fs');
const path = require('path');
const {Document,Packer,Paragraph,TextRun,HeadingLevel,Footer,PageNumber,AlignmentType,PageBreak}=require('C:/Users/wumin/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/docx');
const stage=path.join(__dirname,'staged');
const pkg=path.join(stage,'新EWSA发布包');
const teacher=path.join(stage,'论文','给辅导老师');
fs.mkdirSync(teacher,{recursive:true});
const title='Reinforcement Learning-Based Computer Numerical Control Motion Planning with Adaptive Preview and Jerk-Constrained Action Projection';
function para(text, opts={}) {return new Paragraph({spacing:{after:130,line:300},...opts,children:[new TextRun({text,...(opts.run||{})})]});}
function heading(text) {return para(text,{heading:HeadingLevel.HEADING_1,spacing:{before:210,after:140},keepNext:true});}
function item(head,text) {return new Paragraph({spacing:{after:145,line:300},children:[new TextRun({text:head+'　',bold:true}),new TextRun(text)]});}
function doc(children,cn=false) {return new Document({creator:'',title:cn?'新旧稿改进清单':'',styles:{default:{document:{run:{font:cn?{ascii:'Arial',hAnsi:'Arial',eastAsia:'SimSun'}:'Arial',size:cn?22:22},paragraph:{spacing:{after:130,line:300}}}},paragraphStyles:[{id:'Heading1',name:'Heading 1',basedOn:'Normal',next:'Normal',quickFormat:true,run:{bold:true,size:28,color:'17374C'},paragraph:{keepNext:true}}]},sections:[{properties:{page:{size:{width:11906,height:16838},margin:{top:1134,bottom:1134,left:1276,right:1276}}},footers:{default:new Footer({children:[new Paragraph({alignment:AlignmentType.RIGHT,children:[new TextRun({children:[PageNumber.CURRENT],size:18,color:'777777'})]})]})},children}]});}
async function save(p,children,cn=false) {fs.writeFileSync(p,await Packer.toBuffer(doc(children,cn)));}
const cover=[
'September 24, 2026',
'Expert Systems with Applications',
'Dear Editor-in-Chief,',
`Please consider our manuscript, “${title}”, for publication in Expert Systems with Applications. The study addresses a coupled decision problem in computer numerical control: using the contour-tolerance band to select executable trajectories and feedrates together, instead of fixing the path geometry before scheduling motion.`,
'We propose the Jerk-constrained Neural Numerical Controller (J-NNC). A policy with 14 current inputs coordinates steering, feedrate and adaptive preview. A kinematic constraint module projects the combined request into planar commands subject to velocity, acceleration and jerk limits. PPO learning and subsequent offline refinement produce the controller evaluated in the manuscript.',
'The revised simulation study evaluates complete ordered traversal, braking and physical rest on a 44.420514 m closed route and an 84-corner dense path. J-NNC completes the long route in 444.177 s, saving 1.008 s over the two-step planner, with no greater speed deficit at any of the 34 corners. On the dense path, complete time is reduced by 25.65% relative to two-step planning and by 6.07% relative to the fastest valid developed two-action controller. Local trajectory comparisons, kinematic-limit utilization and all-corner statistics explain the observed efficiency gains.',
'The work connects learned decision-making with explicit physical constraints in an intelligent manufacturing application. The manuscript has not been submitted elsewhere for publication and represents our original work. Thank you for considering it.',
'Sincerely,','Zifang Hu','Corresponding Author','School of Mechanical Science and Engineering','Huazhong University of Science and Technology','Wuhan, China','Email: huzifang@hust.edu.cn'];
const highlights=[
'Joint steering, feedrate and preview shape motion within a tolerance band.',
'Planar action projection enforces velocity, acceleration and jerk limits.',
'Long-route speed loss is no greater at any of the 34 corners.',
'Dense-path time falls by 25.65% versus two-step planning.',
'Dense-path time is 6.07% shorter than the fastest valid two-action result.'
];
if(highlights.some(s=>s.length>85))throw new Error('Highlight exceeds 85 characters');
const pages=[
[
['title','新旧稿改进清单'],
['text','供论文辅导老师审阅｜2026年9月24日'],
['text','对照基础：原EAAI匿名投稿稿18页；本次英文修订稿23页，中文对照稿21页。研究主题与英文标题沿用原稿。以下按论文内容说明改进，不包含开发过程和逐轮记录。'],
['h','一、此次修订主要解决什么问题'],
['text','本次修订沿用原稿关于“轨迹几何与进给应协同规划”的研究问题，重点加强实验对这一主张的支撑。原稿主要在方形、圆形和蝶形路径上展示可行性；新稿增加完整长路径和连续密集拐角，以完整执行时间、全角点速度亏损、巡航和运动学约束利用率说明方法的实际收益。'],
['item','1. 保留论文主线','保留标题、引言的核心论证，以及“问题建模—方法—实验—结论”的章节顺序。实验仍按“实验设置—主要对比结果—消融结果”组织，没有改成技术报告。'],
['item','2. 增强实验难度和覆盖','采用44.420514 m闭合长路径，统计全部34个角点；另设1438.009 mm、84角的密集案例，考察连续过弯时的收益。原长路径完整保留，未截取有利路段代替全程。'],
['item','3. 统一完整成功标准','在相同1.25 mm单侧欧氏允差与平面向量V/A/J上限100/2000/62500下，要求有序覆盖、完整制动和至少20 ms物理停稳。时间包含终段和停稳。'],
['item','4. 对比从“能否运行”延伸到“如何改善”','同时考察NNC的约束可行性、两步法的完整效率，以及去掉前瞻动作后的两动作控制器。局部轨迹与全程统计相互印证。'],
['item','5. 改善图表表达','正文保留9幅主图和3张结果表；算法细节、学习参数、动作对应和步长核验移入附录。图中统一方法名称、颜色和量纲，J-NNC在表中置于首行。']
],
[
['h','二、实验增加了哪些有用证据'],
['item','6. 完整长路径效率','J-NNC用时444.177 s，两步法445.185 s，选定两动作模型444.714 s；分别节省1.008 s和0.537 s。正文明确这一完整时间口径，不将局部过弯时间代替全程时间。'],
['item','7. 密集案例的主要效率优势','J-NNC用时14.839 s，两步法19.957 s，选定两动作模型15.875 s；相应缩短25.65%和6.53%。以已评估的最快有效两动作结果15.798 s为更强比较对象，仍缩短6.07%。两个两动作分母分开说明，避免混用。'],
['item','8. 全部角点及持续巡航','统一以每个角点±10 mm窗口的积分速度亏损评价。长路径全部34角均不劣于两步法及所比较的两动作控制器，允许零亏损并列。≥99 mm/s的累计巡航达到442.552 s，最长连续区间为111.753 s；两步法分别为441.238 s和71.569 s。'],
['item','9. 解释时间收益来自哪里','将完整时间分解为实际行程与累计速度亏损。相对两步法节省的1.008 s中，约0.991 s来自速度亏损减少，约0.017 s来自行程缩短。这样可以说明收益主要由减少不必要降速得到。'],
['item','10. 补回NNC结果图','单列NNC在两案例开头相同80 ms内的平面加速度和jerk越限情况，初始20 ms单独标注。该图支持执行约束层的必要性，不为未完成的NNC推算完整时间。'],
['item','11. 补充运动学约束利用率','正文增加V/A/J利用率结果表，并在同一90°角按参考进度对齐速度、加速度和jerk。与局部轨迹图结合，说明更少的降速与可执行约束之间的关系。'],
['item','12. 扩充模型结果与数值核验','三个不同初始化的选定学生模型均通过完整成功、物理约束和全部34角标准，密集案例相对最快有效两动作结果的收益为5.94%—6.07%。1 ms为正文主结果，0.5 ms为附录核验；如实区分“选定模型均成功”与“同一完整流程独立重复”。']
],
[
['h','三、九幅主图和正文结果表'],
['item','图1｜方法架构对比','保留原框架，校正14维输入、三动作及平面KCM等标签，使图与实际方法一致。'],
['item','图2｜参考路径与允差带','保留原图内容，用于说明几何可行域。'],
['item','图3｜闭环执行框架','保留原流程逻辑，校正固定几何输入、当前状态、动作请求和执行约束之间的关系。'],
['item','图4｜长路径与角度覆盖','标注最长/最短连续直线、最大/最小夹角和典型圆弧；短直线放大图使用10 mm网格；另列全部34角分布。'],
['item','图5｜NNC约束失效','并列两案例的加速度与jerk越限记录，以实际执行数据说明可行性问题。'],
['item','图6｜代表角V/A/J与速度亏损','在90°/C02角直接对齐三类运动学曲线，并给出同一窗口内的速度亏损与最低速度，突出过弯进给差异。'],
['item','图7｜典型局部轨迹','对135°、90°和45°角分别展示进入、通过、恢复及进一步放大视图；保持等比例坐标与允差带，避免全路径曲线重叠遮挡。'],
['item','图8｜全部角点与全程巡航','展示34角速度亏损、低于99 mm/s的累计时间，以及巡航总时长、占比和最长连续区间。'],
['item','图9｜密集案例与两动作对比','给出密集路径、累计速度亏损及90°和45°局部轨迹，支持连续拐角场景中的完整效率收益。'],
['item','正文表1—表3','依次呈现完整可行性/时间/最大轮廓偏差、V/A/J利用率、三个选定学生模型的完整时间。“最大轮廓偏差（mm）”替代含义不清的“最大距离”。'],
['item','附图B.1和C.1','B.1精简对应三动作与实际航向、曲率、速度；C.1保留完整长路径V/A/J及最终制动。用于核对正文结论，不占用主文叙述。']
],
[
['h','四、实验之外的必要修改'],
['item','13. 摘要、贡献与结论同步','换成新案例、完整时间和约束口径；强调在允差带内协调几何与进给，不保留已经被新实验替代的旧数值和过强结论。'],
['item','14. 状态与动作据实修订','明确14维当前观测、无策略历史记忆，固定前向几何采样位置与可调前瞻动作分开表述。保留转向、进给、前瞻三动作主线，附录交代静态特征和模型表示。'],
['item','15. KCM与真实执行链一致','将原线/角运动标量表述校正为真实平面向量约束和可行加速度处理，区分20 ms策略请求与1 ms物理执行。正文保留必要原理，KCM伪代码与完整制动说明移到附录A。'],
['item','16. 学习目标与训练说明校正','按真实实现简述物理时间、进度和失败惩罚。取消正文独立PPO参数小节，将参数、PPO与离线精化的区别置于附录B。此项属于必要的方法一致性修改，不能概括为纯排版调整。'],
['item','17. 正文与附录分工','正文回答可行性、完整效率、过弯形状与三动作整体收益；附录保留支撑复现和核验所需细节。探索、失败修复和逐轮日志不作为论文内容。'],
['item','18. 图件、引用与双语交付','核验24条引用，统一图表中的方法名称、量纲与模型版本；交付可编译LaTeX、PDF/SVG图件及中文对照稿。原18页稿与旧EAAI发布包保留。新投稿包面向Expert Systems with Applications，标题未变。'],
['h','五、建议向老师概括时保留的边界'],
['text','新稿能够以完整仿真证据说明所开发J-NNC控制器在统一约束下优于两步法和已比较的两动作控制器。三个选定模型共享教师数据且后续精化不同，不称为同一完整训练流程的三次独立成功；等预算配对试验也不足以把全部收益单独归因于前瞻动作。'],
['text','论文不提出未见路径泛化或实机加工验证结论，也不声称所有形状指标同时最优。改进的重点是：将原稿较为有限的实验展示，补成“完整可行—效率改善—局部机制—全程统计”的证据链。']
]
];
const children=[];
pages.forEach((page,i)=>{if(i)children.push(new Paragraph({children:[new PageBreak()]}));for(const [kind,a,b] of page){children.push(kind==='title'?para(a,{run:{bold:true,size:36},spacing:{after:180}}):kind==='h'?heading(a):kind==='item'?item(a,b):para(a));}});
const md=pages.map(p=>p.map(([k,a,b])=>k==='title'?'# '+a:k==='h'?'## '+a:k==='item'?'**'+a+'**　'+b:a).join('\n\n')).join('\n\n');
(async()=>{
await save(path.join(pkg,'03_cover_letter','cover_letter_ESWA.docx'),cover.map(t=>para(t,{spacing:{after:95,line:270}})));
await save(path.join(pkg,'07_highlights','Highlights.docx'),[heading('Highlights'),...highlights.map(t=>para(t,{bullet:{level:0}}))]);
fs.writeFileSync(path.join(pkg,'07_highlights','highlights.txt'),highlights.join('\n')+'\n','utf8');
await save(path.join(teacher,'新旧稿改进清单.docx'),children,true);
fs.writeFileSync(path.join(teacher,'新旧稿改进清单.md'),md+'\n','utf8');
console.log('Created ESWA cover letter, 5 highlights and four-page teacher comparison.');
})();
