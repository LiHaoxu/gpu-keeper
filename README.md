# gpu-keeper

在多人共用的 GPU 服务器上，自动占住空闲的卡，留给自己的实验用。卡一空出来就占上，占到的卡一直保持，直到你手动释放。

用 bash + docker 实现，不需要改实验代码。

## 原理

```
                  每 POLL 秒检查一次
gpu_keeper.sh ─────────────────────────────────────────────┐
   │ 1. 已占的卡：占位容器不在运行就重启                      │
   │ 2. 占的卡不足 TARGET 张：找显存占用 < IDLE_MIB 的卡       │
   │    在上面启动一个占位容器                                │
   ▼                                                          │
占位容器 gpu-hold-g<N>（docker，--restart unless-stopped）      │
   分配（当前空闲显存 − RESERVE_GB）然后一直 sleep              │
   ⇒ 别人看到这张卡被占用；你自己仍有 RESERVE_GB 可用 ◄─────────┘
```

- 占位容器只占**显存**，不占算力，基本不耗电、不占 CPU。
- 已占的 GPU 编号记在 `$STATE_DIR/held`，你的调度脚本可以读这个文件来决定实验跑在哪些卡上。
- 占位容器带 `--restart unless-stopped`：docker 或机器重启后会自动恢复；keeper 也会把意外退出的容器重新拉起来。

## 快速开始

```bash
git clone <this repo> && cd gpu-keeper
chmod +x *.sh

# 占 4 张卡，每张留 24 GiB 给自己；后台运行，退出终端也不受影响
TARGET=4 RESERVE_GB=24 setsid nohup ./gpu_keeper.sh >> keeper.log 2>&1 < /dev/null &

./gpu_status.sh          # 查看 keeper 是否在运行、占了哪些卡、每张卡的显存
cat ~/.gpu_keeper/held   # 已占的 GPU 编号，一行一个

./gpu_release.sh         # 用完后全部释放
```

在占住的卡上跑实验，例如：

```bash
for g in $(cat ~/.gpu_keeper/held); do
    docker run --rm --gpus "\"device=$g\"" ... your_job &
done
```

## 参数（环境变量）

| 变量 | 默认值 | 含义 |
|---|---|---|
| `TARGET` | 2 | 要占多少张卡 |
| `RESERVE_GB` | 16 | 每张卡留给自己的显存（GiB）。占位容器分配的是“启动时的空闲显存 − RESERVE_GB” |
| `IDLE_MIB` | 100 | 显存占用低于这个值（MiB）才算空闲 |
| `CANDIDATES` | 空 | 只在这些卡里挑，例如 `0,1,2,3`；空表示所有卡 |
| `IMAGE` | `nvcr.io/nvidia/pytorch:25.06-py3` | 占位容器用的镜像，需要有 python 和 torch |
| `PREFIX` | `gpu-hold-g` | 占位容器名前缀，容器名为 `<PREFIX><GPU 编号>` |
| `STATE_DIR` | `~/.gpu_keeper` | 存放 `held` 和 `keeper.pid` |
| `POLL` | 30 | 检查间隔（秒） |

`gpu_release.sh` 和 `gpu_status.sh` 读取同样的 `STATE_DIR` 和 `PREFIX`。如果启动 keeper 时改过这两个变量，运行另外两个脚本时也要设成一样的值。

想中途改 `TARGET` 或 `RESERVE_GB`：先 `kill $(cat ~/.gpu_keeper/keeper.pid)`，再用新参数启动 keeper。已有的占位容器和 `held` 列表都会保留。要让某张卡的占位容器按新的 `RESERVE_GB` 重新分配，执行 `docker stop gpu-hold-g<N>`，keeper 下一轮会自动按新参数重建它。

## 使用经验与注意事项

1. **占位容器只保证“看起来被占”，不保证别人进不来。** 留给自己的 `RESERVE_GB` 在你的作业还没用上时是空着的，别人的程序照样能申请。我们遇到过一次：别人 16 GB 的作业占进了这部分显存，导致我们在这张卡上 OOM。应对方法：
   - 每卡并发按“单作业显存 × 并发数 ≤ RESERVE_GB − 余量”来估算，不要压满。
   - 发现某张卡被别人挤占，就把这张卡从你的调度列表中拿掉，**不要去动别人的进程**。
2. **占位容器按“启动那一刻的空闲显存”来分配。** 如果它在你的作业还在跑的时候被重启，占到的显存会少。要重新占满，等作业结束后 `docker stop gpu-hold-g<N>`，让 keeper 重建。
3. **`IDLE_MIB` 不要设太大。** 设成 1024 会把别人只用了几百 MB 的卡（比如一些小 benchmark）也当成空闲，从而占到别人正在用的卡上。
4. **通过 ssh 远程操作时，不要用 `pkill -f <模式>`，也不要用 `ps | grep <模式> | wc -l` 计数。** 如果你自己的命令行里也包含这个模式，前者会把你的 ssh shell 一起杀掉，后者的计数永远不会归零。本仓库的脚本都用 `keeper.pid` 和容器名来停止进程，不按命令行匹配。
5. **用完及时释放。** 共享机器上长期占着不用会影响别人，实验跑完就运行 `./gpu_release.sh`。

## 文件

| 文件 | 作用 |
|---|---|
| `gpu_keeper.sh` | 后台循环：维持并补足占位容器 |
| `gpu_release.sh` | 停止 keeper，停止并删除所有占位容器，清空 `held` |
| `gpu_status.sh` | 查看 keeper 状态、已占的卡和各卡显存 |
