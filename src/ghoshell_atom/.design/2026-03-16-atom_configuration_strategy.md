# Atom 配置策略设计

## 背景
Atom 项目需要为 AI 自迭代场景设计配置管理方案。核心需求：
1. **AI 可理解**：配置应对 AI 透明，支持 Code as Prompt 原则
2. **运行时可修改**：AI 能在运行时安全地修改配置并立即生效
3. **端侧安全**：配置修改不能导致进程崩溃
4. **开发友好**：人类工程师也能轻松理解和修改

## 方案对比分析

### 方案1：基于文件约定配置
- **优点**：可序列化（yaml/json），易于迁移到数据库/配置中心；支持 watchdog 热重载；权限分离
- **缺点**：配置与实现分离（重复劳动）；同步风险；解释性不足

### 方案2：代码即配置
- **优点**：极致自解释（代码即文档）；零抽象成本；类型安全；符合 Code as Prompt
- **缺点**：运行时修改危险；热更新复杂；序列化困难

## 核心决策
采用 **混合策略**：基于现有 `ghoshell_ghost.contracts.configs` 抽象，增强为 **缓存+watchdog** 模式。

### 设计原则
1. **无状态获取**：业务代码每次都调用 `get_or_create`，不持有配置引用
2. **透明缓存**：ConfigStore 内部管理缓存，业务代码无感知
3. **文件监听**：文件变化时自动失效缓存
4. **懒加载**：首次访问时加载，后续从缓存获取
5. **UNIX 哲学**：通过文件系统协调，组件独立

## 技术实现方案

### 1. 增强的 ConfigStore
```python
class CachedYamlConfigStore(YamlConfigStore):
    """带缓存和 watchdog 的 ConfigStore"""

    def __init__(self, configs_dir: str):
        super().__init__(configs_dir)
        self._cache: Dict[str, Tuple[ConfigType, float]] = {}  # 配置缓存
        self._cache_lock = threading.RLock()  # 线程安全
        self._stop_watchdog = threading.Event()  # 停止信号
        self._watchdog_thread = None  # 监听线程

    def get_or_create(self, conf_type: Type[CONF_TYPE]) -> CONF_TYPE:
        """带缓存的获取或创建"""
        cache_key = conf_type.conf_name()

        with self._cache_lock:
            # 检查缓存
            if cache_key in self._cache:
                cached_conf, timestamp = self._cache[cache_key]
                if self._is_file_unchanged(cache_key, timestamp):
                    return cached_conf

            # 从父类获取（检查文件是否存在）
            conf = super().get_or_create(conf_type)
            self._cache[cache_key] = (conf, time.time())
            return conf

    def start_watchdog(self):
        """启动文件监听"""
        def watch_files():
            for changes in watch(self._configs_dir, stop_event=self._stop_watchdog):
                for change_type, file_path in changes:
                    # 提取相对路径作为缓存 key
                    rel_path = os.path.relpath(file_path, self._configs_dir)
                    cache_key = rel_path.replace('.yml', '')

                    # 失效缓存
                    with self._cache_lock:
                        self._cache.pop(cache_key, None)
```

### 2. 原子性和一致性保证
```python
class AtomicYamlConfigStore(CachedYamlConfigStore):
    """支持原子写入的 ConfigStore"""

    def save(self, conf: ConfigType, relative_path: Optional[str] = None) -> None:
        """
        原子写入：先写入临时文件，然后重命名
        避免写入过程中读取到部分数据
        """
        # 创建临时文件
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.yml',
            dir=os.path.dirname(file_path)
        )

        # 写入临时文件
        with os.fdopen(temp_fd, 'wb') as f:
            content = self._marshal(conf.model_dump(), type(conf))
            f.write(content)

        # 原子重命名（Unix 保证）
        os.replace(temp_path, file_path)

        # 更新缓存
        with self._cache_lock:
            self._cache[cache_key] = (conf, time.time())
```

### 3. 业务代码使用模式
```python
# 任何需要配置的地方
def some_business_function():
    # 获取当前 Ghost 实例
    ghost = Atom.get_env_instance()
    store = ghost.container.get(ConfigStore)

    # 每次调用都获取最新配置
    channel_config = store.get_or_create(ChannelConfig)

    # 使用配置
    if channel_config.enabled:
        timeout = channel_config.timeout_seconds

    # AI 修改配置
    new_config = channel_config.copy(update={"timeout_seconds": 60.0})
    store.save(new_config)  # 自动更新缓存
```

## 架构优势

### 1. 简单性
- **没有复杂抽象**：无观察者模式、无依赖图管理
- **职责清晰**：ConfigStore 负责缓存，业务负责获取
- **易于调试**：配置在文件中，可直接查看和修改

### 2. 可靠性
- **故障隔离**：一个配置读取失败不影响其他
- **自动恢复**：文件损坏时使用缓存或创建默认值
- **线程安全**：锁保护缓存访问

### 3. 性能
- **缓存透明**：业务代码无感知
- **懒加载**：按需加载配置
- **分级缓存**：可选内存缓存 + 进程缓存优化

### 4. AI 友好
- **Code as Prompt**：配置类型是 Python 类，AI 可阅读源码
- **自解释文档**：Pydantic Field 的 description 字段提供 AI 可读说明
- **安全更新**：原子写入 + 缓存失效保证一致性

## 初始化顺序解决方案

### 方案1：懒加载 + 依赖检查
```python
class ConfigDependencyChecker:
    """配置依赖检查器（轻量级）"""

    @classmethod
    def ensure_config_exists(cls, conf_type: Type[ConfigType]) -> None:
        """确保配置存在，如果不存在则创建"""
        ghost = Atom.get_env_instance()
        store = ghost.container.get(ConfigStore)
        store.get_or_create(conf_type)  # 创建默认配置如果不存在
```

### 方案2：启动时预加载
```python
class ConfigPreloader:
    """启动时预加载关键配置"""

    ESSENTIAL_CONFIGS = [ChannelConfig, ModelConfig, LoggingConfig]

    @classmethod
    def preload(cls):
        """预加载所有关键配置"""
        ghost = Atom.get_env_instance()
        store = ghost.container.get(ConfigStore)

        for conf_type in cls.ESSENTIAL_CONFIGS:
            store.get_or_create(conf_type)
```

## 性能优化策略

### 1. 分级缓存（可选）
```python
class TieredCacheConfigStore(CachedYamlConfigStore):
    """分级缓存：内存缓存 + 进程缓存"""

    def get_or_create(self, conf_type: Type[CONF_TYPE]) -> CONF_TYPE:
        cache_key = conf_type.conf_name()

        # 检查进程缓存（TTL 1秒）
        now = time.time()
        if cache_key in self._process_cache:
            if now - self._process_cache_ttl[cache_key] < 1.0:
                return self._process_cache[cache_key]

        # 从父类获取（包含文件缓存）
        conf = super().get_or_create(conf_type)
        self._process_cache[cache_key] = conf
        self._process_cache_ttl[cache_key] = now
        return conf
```

### 2. 批量读取优化（可选）
```python
class BatchConfigStore(CachedYamlConfigStore):
    """支持批量读取的 ConfigStore"""

    def batch_get(self, conf_types: List[Type[ConfigType]]) -> Dict[Type[ConfigType], ConfigType]:
        """批量获取配置，减少锁竞争"""
        result = {}

        with self._cache_lock:
            for conf_type in conf_types:
                # 批量处理缓存逻辑...
                result[conf_type] = conf

        return result
```

## 实施优先级

### 简单版本（MVP）
1. ✅ **继承 YamlConfigStore**：添加内存缓存
2. ⬜ **覆盖 get_or_create**：实现缓存逻辑
3. ⬜ **简单 watchdog**：监听文件变化，失效缓存
4. ⬜ **业务代码适配**：总是调用 `store.get_or_create()`

### 增强版本
5. ⬜ **原子写入**：避免写入过程中读取到损坏数据
6. ⬜ **一致性保证**：文件损坏时自动恢复
7. ⬜ **性能优化**：分级缓存、批量读取
8. ⬜ **监控日志**：记录配置变更历史

## 相关文件
- `src/ghoshell_ghost/contracts/configs.py`：现有 ConfigType/ConfigStore 抽象
- `src/ghoshell_ghost/concepts/ghost.py`：Ghost 单例和 config_models() 接口
- `src/ghoshell_atom/framework/configs.py`：Atom 配置管理实现
- `src/ghoshell_atom/templates/src/Atom/configs.py`：配置类型定义

## 设计验证

### 符合 AI 自迭代场景
1. **AI 可理解**：配置是 Python 类，AI 可阅读源码和文档字符串
2. **运行时可修改**：AI 通过 Ghost 单例获取 ConfigStore 并调用 save()
3. **安全更新**：原子写入 + 缓存失效保证不会读取到部分数据
4. **即时生效**：业务代码下次调用 get_or_create() 时获取新配置

### 符合端侧运行约束
1. **文件优先**：配置存储在 YAML 文件中，符合 UNIX 哲学
2. **进程单例**：通过 Ghost 单例访问，保证多进程一致性
3. **资源友好**：懒加载 + 缓存减少文件 IO

---
*设计记录创建于 2026-03-16，基于与人类工程师关于 AI 时代配置策略的讨论总结*