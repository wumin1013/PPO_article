# 优化指令 08：可视化修复与训练稳定性强化 (Visualization Repair & Training Stabilization)

## 1. 背景与目标 (Context)

当前项目完成了环境解耦，但在运行时暴露了两个严重的工程问题：

1. **Dashboard 假死：** 在 Windows 环境下，`main.py` 写入日志与 `app.py` 读取日志发生文件锁冲突，导致 Streamlit 界面永久挂起（Pending）。
2. **训练数值不稳定：** 由于移除了奖励归一化并引入了随机复位（RSI），Critic 网络面临目标值方差过大的问题，导致 Loss 震荡甚至梯度爆炸。

**本次目标：** 通过“安全读取机制”修复可视化假死，通过“梯度裁剪”与“归一化确认”稳定训练收敛。

------

## 2. 第一阶段：修复 Dashboard 挂起问题 (Fix Visualization)

**核心任务：** 修改 `PPO_project/app.py`，消除文件读写冲突并防止内存泄漏。

### 步骤 1.1：实现“安全重试”读取机制

**执行动作：** 在 `app.py` 中，找到所有直接调用 `pd.read_csv` 的地方。你需要编写一个**封装函数**来替换原本的直接读取操作。 该封装函数必须具备以下逻辑：

1. **循环重试：** 设定一个 3-5 次的尝试循环。
2. **异常捕获：** 在循环中捕获 `PermissionError`（权限错误）和 `OSError`。
3. **退避策略：** 一旦捕获错误（意味着文件正被 `main.py` 占用写入），让程序休眠一小段时间（如 0.1秒）再重试。
4. **失败兜底：** 如果重试耗尽仍无法读取，应返回一个空的 DataFrame，而不是让程序崩溃或无限等待。
5. **引擎指定：** 在读取时显式指定 `engine='c'` 或使用 copy-on-read 策略，减少文件句柄占用的时间。

### 步骤 1.2：强制释放绘图内存

**执行动作：** 检查 `app.py` 中所有使用 `matplotlib.pyplot` (即 `plt`) 生成图表的地方。

- **强制关闭：** 在每一次调用 `st.pyplot(fig)` 将图表渲染到网页后，**必须**立即显式调用 `plt.close(fig)`。
- **原因：** Streamlit 在刷新时不会自动清理旧的 Matplotlib 对象，如果不手动关闭，内存会迅速溢出，导致页面卡顿最终挂起。

------

## 3. 第二阶段：算法数值稳定化 (Stabilize Algorithm)

**核心任务：** 修改 `src/algorithms/ppo.py` 和 `main.py`，限制梯度更新幅度，确保输入数据标准化。

### 步骤 2.1：引入梯度裁剪 (Gradient Clipping)

**执行动作：** 打开 `src/algorithms/ppo.py`，定位到 `update` 方法中的反向传播部分（即 `loss.backward()` 之后，`optimizer.step()` 之前）。

- **插入逻辑：** 对 Actor 网络和 Critic 网络的参数应用 **梯度裁剪 (Gradient Norm Clipping)**。
- **参数设定：** 将最大梯度范数（Max Norm）设定为 **0.5** 或 **1.0**。
- **目的：** 这将强制限制每一次参数更新的步长，防止因为移除了奖励归一化而产生的巨大 Loss 值导致神经网络权重瞬间“崩塌”。

### 步骤 2.2：移除残留的奖励归一化代码

**执行动作：** 再次检查 `src/algorithms/ppo.py` 的 `update` 方法。

- **确认删除：** 确保代码中**不再包含**任何对 `rewards` 变量进行 `(r - mean) / std` 操作的代码行。
- **保留 Advantage 归一化：** 注意，我们要保留对 `advantage`（优势函数）的归一化，这两个不要混淆。

### 步骤 2.3：严查状态归一化注入

**执行动作：** 打开 `PPO_project/main.py`，仔细审查训练主循环（While loop 或 For loop）。

- **逻辑核对：** 必须确保 `env.reset()` 产生的初始状态，以及 `env.step()` 产生的下一状态，**在传入 Agent 之前**都经过了 `normalizer(...)` 的处理。
- **常见错误修正：** 如果发现代码是 `state = env.reset()` 直接接着 `action = agent.take_action(state)`，这是错误的。必须在中间插入 `state = normalizer(state)`。

------

## 4. 自验证清单 (Verification Checklist)

完成上述修改后，请按以下流程自检：

1. **启动测试：** 同时运行 `main.py`（开始训练）和 `run_dashboard.py`（启动面板）。
2. **交互测试：** 在 Dashboard 页面连续多次点击“手动刷新日志”按钮。
   - *合格标准：* 页面应在 1 秒内响应并刷新数据，**不再出现**一直转圈的 "Running" 或 "Pending" 状态。
3. **观察 Loss 曲线：**
   - *合格标准：* 查看终端或日志中的 Critic Loss。虽然它在 RSI 阶段（前 30%）仍会有波动，但数值应当被限制在合理范围内（例如不会突然跳到几千或 NaN），且 Actor Loss 应该呈现缓慢下降或震荡收敛的趋势。