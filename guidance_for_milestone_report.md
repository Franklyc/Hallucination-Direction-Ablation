最可行的做法，不是把题目改掉，而是把**验证顺序**改掉：先做一个 **HDA-lite**，先证明“hallucination direction 这件事在你的模型上有信号、而且能因果地影响 TruthfulQA 分数”，再把它永久化成 weight patch。这样和你原提案的主线是一致的：你提案本来就承诺了 TruthfulQA 多项选择评测、对比式 direction 提取、rank-one orthogonalization、能力漂移控制，以及 DoLa 作为计划中的强 baseline；现在只是把“先 patch 再验证”改成“先验证再 patch”。

我建议你立刻收缩成这个版本：

**1. 评测先改成 TruthfulQA 的新 binary-choice multiple-choice。**
这一步和提案并不冲突，因为它本质上仍然是 multiple-choice、仍然是 deterministic log-prob scoring。更关键的是，TruthfulQA 官方仓库在 2025 年更新后明确推荐新 binary setting，并说明旧 MC1/MC2 和新版本的模型表现“非常相似”，所以这是一个更干净、但仍能和原提案对得上的替代。官方仓库还直接把 `Best Incorrect Answer` 放进了 `TruthfulQA.csv`，实现会比你自己处理 MC1/MC2 更省事。([GitHub][1])

**2. 先不要把时间花在完整复现 DoLa 上。**
这不是说不做，而是**这两天先不优先**。官方 DoLa 仓库当前 README 里写得很清楚：它用的是单独的 `transformers-4.28.1`，并且“currently we only support LLaMA-v1”。这意味着如果你现在主实验模型不是 LLaMA-v1，硬啃 DoLa 官方复现很可能把你两天全吃掉。最稳的说法是：DoLa 仍是 final comparison 的 planned baseline，但 progress report 先把核心 HDA mechanism 跑通。([GitHub][2])

**3. 你现在真正该做的是“activation probe → limited patch”这条最短闭环。**
具体说，就是：

* 用一个你已经能稳定跑通 hidden states / forward hooks 的 7B/8B instruct 模型；
* 跑 TruthfulQA binary-choice baseline；
* 用提案里的对比 prompt 思路提取每层 direction；
* 在**推理时最后一个 prompt token**上做 activation-level projection removal，先看 A/B 选择的 logit 是否朝正确方向移动；
* 只有在这个 probe 有效果后，再把同一方向写成真正的 weight patch。
  这条路最像你提案，而且最容易在两天内拿到“能写进 progress report 的图和表”。

这里最关键的技术收缩是：**activation probe 不要做常量减法，做 projection removal。**
也就是不要直接用 `r' = r - beta * v`，而用
[
r' = r - \beta (r^\top \hat v)\hat v
]
因为这更接近你提案里 weight orthogonalization 的本意：不是给所有样本统一平移，而是把当前样本在该方向上的分量削掉。这样如果 probe 有效，你后面再做
[
W' = W - \alpha \hat v (\hat v^\top W)
]
会非常顺。这个“先 activation、后 weight”的叙事，也更容易在 progress report 里讲成“先做因果验证，再做永久化实现”。

我会建议你这 48 小时按这个顺序推进：

1. **今天先把 binary-choice evaluator 跑通。**
   Prompt 固定成：问题 + A/B 两个选项 + `Answer:`，只比较下一 token 取 `A` 还是 `B` 的 logprob。先拿到 base accuracy，这样你至少已经有一个真正的 benchmark number 了。TruthfulQA 官方现在也明确给了这种 binary-choice 方向。([TruthfulAI][3])

2. **然后做 direction extraction。**
   先别全 817 题，抽一个 calibration subset。还是沿用提案思路：

   * `P_h`: encourage guessing / answer confidently even if unsure
   * `P_g`: be careful / avoid unsupported answers / abstain if uncertain
     抓每层最后一个 prompt token 或第一生成位点附近的 residual hidden state，做 mean difference。
     这一步的产物至少要有：每层 direction norm、层间 cosine similarity、以及少量 sanity check。

3. **今晚做 activation probe。**
   在 binary-choice prompt 上，对若干层做 projection removal，扫一小组 `beta`。你不用一开始就全层全参数网格，先做：

   * middle layers
   * late layers
   * middle+late 各 2–3 层
     只要你能看到某些层段让正确选项 logit 上升、或者 binary accuracy 有初步提升，就已经足够写进 progress report。

4. **明天再做最小版 weight patch。**
   只 patch 一种模块，优先 **attention out projection**。先别同时碰 MLP。
   只 patch activation probe 里最有效的 1–3 层。
   只做 2–3 个强度。
   这样你很有机会拿到一个“永久 patch 版本”的初步数值，而不是陷在大规模搜索里。

5. **能力漂移先做便宜版。**
   你提案写了 KL drift / benign prompt set。现在别搞复杂 benchmark，先手工准备 30–50 个 benign prompts，比较 base 和 patched model 的 next-token KL 或简单输出变化统计。progress report 里写“preliminary capability-drift proxy”就够了。

你这次 progress report 最好写成下面这个姿态，而不是 pretending everything is done：

> We preserved the original project objective and evaluation setting, but reordered implementation to first validate the existence and causal utility of a hallucination-related direction before committing to a permanent weight patch. Concretely, we implemented the TruthfulQA multiple-choice evaluation pipeline, extracted candidate layer-wise directions from contrastive prompts, and are testing activation-level projection removal as a causal probe prior to rank-one weight orthogonalization.

这段是完全能和原提案对上的，因为它没有改问题、没有改核心方法，只是改了工程顺序。

如果你问我“什么情况算这两天交代得过去”，门槛其实不高。你只要拿到下面三样东西，progress report 就已经站得住：

* 一个 **base binary-choice TruthfulQA** 分数；
* 一张 **per-layer direction / activation probe** 的图；
* 一段诚实但正面的结论：某些层上存在初步可干预信号，weight patch implementation is in progress。

最不该做的反而是：现在去追完整 DoLa 复现、全量 KL、三大消融、再加第二个模型。DoLa 官方实现当前对 LLaMA-v1 和老版 transformers 绑定很重，这件事非常容易把你卡死在环境里，而不是帮你在两天内交出 HDA 的实质进展。([GitHub][2])

你现在就可以开始，按最小代码结构建 4 个文件就够了：

* `truthfulqa_binary_eval.py`
* `extract_direction.py`
* `activation_probe.py`
* `weight_patch_eval.py`

先把第一个和第三个跑通，今天你就已经不是“还在想”，而是已经进入“有初步结果”的状态了。

[1]: https://github.com/sylinrl/TruthfulQA "GitHub - sylinrl/TruthfulQA: TruthfulQA: Measuring How Models Imitate Human Falsehoods · GitHub"
[2]: https://github.com/voidism/DoLa "GitHub - voidism/DoLa: Official implementation for the paper \"DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models\" · GitHub"
[3]: https://truthful.ai/blog/truthfulqa-binary-choice/ "New, improved multiple-choice TruthfulQA | TruthfulAI"
