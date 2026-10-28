export const DEMO_ITEM = {
  id: 1001,
  filename: '数据结构课程 · 二叉树遍历.mp4',
  status: 'COMPLETED',
  uploadTime: '2026-07-10T14:30:00',
  transcriptText: '本节课介绍二叉树前序、中序和后序遍历，并对比递归与迭代实现。'
}

export const DEMO_PLAN = {
  understoodGoal: '概括视频主要内容和核心观点，并引用有代表性的时间戳证据',
  intent: 'STRUCTURED_SUMMARY',
  tasks: ['读取视频元数据并确定总结范围', '检索主题、观点和示例相关的时间轴证据', '展开关键证据窗口并生成结构化报告', '校验结论、引用和报告结构']
}

export const DEMO_TRACE = {
  agentMode: 'MODEL_TOOL_CALLING',
  toolCallCount: 5,
  toolCalls: [
    { index: 1, tool: 'get_video_metadata', durationMs: 3, success: true },
    { index: 2, tool: 'search_timeline', durationMs: 18, success: true },
    { index: 3, tool: 'get_evidence_window', durationMs: 2, success: true },
    { index: 4, tool: 'verify_citations', durationMs: 4, success: true },
    { index: 5, tool: 'generate_report', durationMs: 5, success: true }
  ],
  stageDurationMs: {
    VIDEO_CONTEXT: 12840,
    RETRIEVAL: 860,
    PLANNER: 1150,
    EXECUTOR: 3420,
    CRITIC: 1260
  },
  spans: [
    { name: 'analysis', parent: null, startOffsetMs: 0, durationMs: 17410, attributes: { agentMode: 'MODEL_TOOL_CALLING', attempt: 1, evidenceSegmentCount: 42, toolCallCount: 5, providerRequestCount: 4 } },
    { name: 'video_context', parent: 'analysis', startOffsetMs: 0, durationMs: 12840, attributes: {} },
    { name: 'planner', parent: 'analysis', startOffsetMs: 12840, durationMs: 1150, attributes: {} },
    { name: 'executor', parent: 'analysis', startOffsetMs: 13990, durationMs: 3420, attributes: { toolCallCount: 5 } },
    { name: 'critic', parent: 'executor', startOffsetMs: 16150, durationMs: 1260, attributes: { approximated: true } },
    { name: 'tool.get_video_metadata#1', parent: 'executor', startOffsetMs: 13990, durationMs: 3, attributes: { tool: 'get_video_metadata', success: true } },
    { name: 'tool.search_timeline#2', parent: 'executor', startOffsetMs: 13993, durationMs: 18, attributes: { tool: 'search_timeline', success: true } },
    { name: 'tool.get_evidence_window#3', parent: 'executor', startOffsetMs: 14011, durationMs: 2, attributes: { tool: 'get_evidence_window', success: true } },
    { name: 'tool.verify_citations#4', parent: 'executor', startOffsetMs: 14013, durationMs: 4, attributes: { tool: 'verify_citations', success: true } },
    { name: 'tool.generate_report#5', parent: 'executor', startOffsetMs: 14017, durationMs: 5, attributes: { tool: 'generate_report', success: true } },
    { name: 'provider', parent: 'analysis', startOffsetMs: 0, durationMs: 4200, attributes: { requestCount: 4 } },
    { name: 'provider.PLANNER#1', parent: 'provider', startOffsetMs: 0, durationMs: 1150, attributes: { phase: 'PLANNER', totalTokens: 820, success: true } },
    { name: 'provider.VERIFIER#2', parent: 'provider', startOffsetMs: 1150, durationMs: 860, attributes: { phase: 'VERIFIER', totalTokens: 980, success: true } },
    { name: 'provider.WRITER#3', parent: 'provider', startOffsetMs: 2010, durationMs: 1420, attributes: { phase: 'WRITER', totalTokens: 1450, success: true } },
    { name: 'provider.CRITIC#4', parent: 'provider', startOffsetMs: 3430, durationMs: 770, attributes: { phase: 'CRITIC', totalTokens: 910, success: true } }
  ],
  budget: { rounds: 4, tokens: 4160, costUsd: 0.0032, elapsedMs: 17410, limits: { maxRounds: 0, maxTokens: 0, maxDurationMs: 0, maxCostUsd: 0 } },
  checkpoint: { enabled: true, restoredStages: ['planner'], savedStages: ['retriever'], promptVersion: 'video-evidence-agent-v5.2-grounded-synthesis' },
  providerUsage: {
    requestCount: 4,
    totalTokens: 4160,
    estimatedCostUsd: 0.0032,
    latencyMsP50: 1050,
    latencyMsP95: 1420,
    byPhase: {
      PLANNER: { requestCount: 1, totalTokens: 820 },
      VERIFIER: { requestCount: 1, totalTokens: 980 },
      WRITER: { requestCount: 1, totalTokens: 1450 },
      CRITIC: { requestCount: 1, totalTokens: 910 }
    }
  },
  budgetExceeded: false
}

export const DEMO_EVALUATION = {
  structuredValid: true,
  evidenceSupportRate: 0.92,
  criticPassed: true
}

export const DEMO_RESULT = `## 二叉树遍历课程分析

## 核心结论
- 前序遍历遵循“根节点、左子树、右子树”，适合构建树结构副本。
- 中序遍历二叉搜索树时可得到有序序列。
- 递归实现更直观，迭代实现通过栈显式保存访问状态。

## 视频证据
- [02:05] ASR：老师开始讲解前序遍历的访问顺序。
- [02:08] OCR：课件显示“根节点 → 左子树 → 右子树”。
- [08:42] ASR + OCR：对比递归调用栈与显式栈实现。

## 学习建议
- 用同一棵树手动推演三种遍历顺序。
- 分别实现递归版与迭代版，重点理解栈中保存的状态。`
