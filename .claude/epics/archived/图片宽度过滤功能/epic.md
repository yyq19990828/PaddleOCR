---
status: completed
progress: 100%
updated: 2025-09-03T08:59:29Z
completed: 2025-09-03T08:59:29Z
---

# Epic: 图片宽度过滤功能

## Overview

为PaddleOCR的transforms系统添加FilterByImageWidth操作符，通过在现有数据预处理管道中集成图片宽度过滤功能，使model evaluation能够聚焦于特定尺寸范围的图片。技术方案利用现有transforms架构和None返回机制，实现零侵入式的过滤功能。

## Architecture Decisions

### 核心架构选择
- **集成策略**: 新增transforms操作符而非修改SimpleDataSet核心逻辑
- **过滤机制**: 利用现有None返回触发样本跳过的架构模式
- **配置方式**: 通过YAML transforms配置，保持与现有系统一致性
- **统计实现**: 使用PaddleOCR内置logger系统输出过滤统计

### 技术选型理由
1. **transforms操作符方案优势**:
   - 完美融入现有数据预处理管道
   - 支持配置化，无需代码修改
   - 可复用于训练和评估场景
   - 遵循现有的模块化设计原则

2. **None返回机制**:
   - 复用现有的错误处理和样本跳过逻辑
   - 无需修改SimpleDataSet的__getitem__方法
   - 保持架构一致性

## Technical Approach

### Backend Services
**核心组件**: FilterByImageWidth操作符类
- **位置**: `ppocr/data/imaug/operators.py`
- **输入**: 配置参数`width_range`和图像数据字典
- **处理逻辑**:
  1. 检查图像已解码（依赖DecodeImage完成）
  2. 获取图像宽度信息
  3. 根据width_range配置判断是否保留
  4. 符合条件返回原数据，否则返回None
- **配置格式支持**:
  - `[min, max]`: 闭区间过滤
  - `[min, ]`: 开区间过滤（大于等于min）
  - `None`: 禁用过滤

**统计功能**:
- **数据收集**: 在操作符中记录过滤统计
- **输出时机**: 数据加载完成后统一输出
- **信息内容**: 总数量、过滤数量、保留数量、过滤率

### Infrastructure

**集成要求**:
- 在`__init__.py`中注册FilterByImageWidth操作符
- 确保在DecodeImage之后执行
- 兼容现有的transforms执行顺序

**性能考虑**:
- 图片宽度检查开销极小（O(1)操作）
- 无额外内存分配
- 预期性能影响<1%

## Implementation Strategy

### 开发阶段划分
1. **Phase 1**: 实现FilterByImageWidth操作符类（1天）
2. **Phase 2**: 集成到transforms系统并添加统计功能（0.5天）
3. **Phase 3**: 测试验证和文档更新（0.5天）

### 风险缓解
- **兼容性风险**: 充分利用现有架构模式，最小化变更
- **性能风险**: 简单的宽度检查，性能影响可忽略
- **配置错误**: 添加参数验证和清晰的错误信息

### 测试策略
- **单元测试**: 验证各种width_range配置的正确性
- **集成测试**: 在真实数据集上验证过滤效果
- **性能测试**: 确保数据加载时间增加<5%

## Task Breakdown Preview

高层次任务分类：

- [ ] **核心实现**: 创建FilterByImageWidth操作符类
- [ ] **系统集成**: 注册操作符到transforms系统
- [ ] **统计功能**: 实现过滤统计信息收集和输出
- [ ] **配置验证**: 添加参数校验和错误处理
- [ ] **单元测试**: 编写操作符功能测试
- [ ] **集成测试**: 在eval.py中验证端到端功能
- [ ] **文档更新**: 更新使用示例和配置说明

## Dependencies

### 外部服务依赖
- **PaddlePaddle框架**: 确保兼容现有版本
- **DecodeImage操作符**: 必须在FilterByImageWidth之前执行
- **Logger系统**: 依赖ppocr内置日志功能

### 内部团队依赖
- **无阻塞依赖**: 该功能为独立模块，无需等待其他团队工作
- **测试数据**: 使用现有测试数据集，包含不同尺寸图片

### 关键路径依赖
1. DecodeImage → FilterByImageWidth → 后续transforms
2. 操作符实现 → 系统注册 → 配置测试

## Success Criteria (Technical)

### 性能基准
- **处理速度**: 数据加载时间增加 < 5%
- **内存使用**: 无显著额外内存占用
- **CPU开销**: 单图片过滤检查 < 1ms

### 质量门禁
- **功能完整性**: 所有width_range格式100%正确工作
- **兼容性**: 与现有transforms零冲突
- **统计准确性**: 过滤统计信息100%准确
- **错误处理**: 配置错误时提供清晰反馈

### 验收标准
- [ ] 支持`[min, max]`、`[min, ]`、`None`三种配置格式
- [ ] eval.py输出仅包含过滤后的图片结果
- [ ] logger正确输出过滤统计信息
- [ ] 不影响现有任何功能
- [ ] 通过所有自动化测试

## Estimated Effort

### 总体时间估算
- **开发时间**: 2天（16工时）
  - 核心实现: 8工时
  - 集成测试: 4工时
  - 文档和验证: 4工时

### 资源需求
- **开发人员**: 1名Python开发者（熟悉PaddleOCR架构）
- **测试环境**: 包含多种尺寸图片的验证数据集
- **无特殊硬件要求**: 标准开发环境即可

### 关键路径项
1. **FilterByImageWidth类实现** (关键路径，1天)
2. **transforms系统集成** (依赖1，0.5天)
3. **端到端验证测试** (依赖2，0.5天)

## Stats

Total tasks: 7
Parallel tasks: 4 (can be worked on simultaneously)
Sequential tasks: 3 (have dependencies)
Estimated total effort: 24小时

---

*该技术实施计划基于PRD需求分析，专注于最小化代码变更的同时最大化功能价值。通过充分利用现有架构，确保实施的简洁性和可靠性。*