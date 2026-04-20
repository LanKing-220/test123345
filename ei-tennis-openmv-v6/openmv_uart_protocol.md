# OpenMV 与下位机串口通信说明

本文档对应当前文件 [main.py](D:/副桌面文件夹/论文/fomo/ei-tennis-openmv-v6/main.py)，用于和下位机开发者统一串口协议、状态约束和联调方式。

## 1. 基本约定

- 串口参数：`115200, 8N1`
- 行结束符：OpenMV 每条报文后发送 `\n`
- 握手阶段：
  - OpenMV 主动周期发送 `hello`
  - 下位机收到后返回 `HI`
  - OpenMV 收到 `HI` 后回发 `OK`
- OpenMV 发出 `OK` 后，双方进入主控命令交互状态
- 控制命令统一采用二阶段确认：
  - 第 1 阶段：`REQ`
  - 第 2 阶段：`CONFIRM`
- `REQ` 与 `CONFIRM` 必须使用同一个 `token`
- `CONFIRM` 必须在 OpenMV 收到对应 `REQ` 后 3 秒内发送

## 2. 报文总表

| 报文示例 | 方向 | 作用说明 |
| --- | --- | --- |
| `hello` | OpenMV -> 下位机 | 视觉端主动发起握手 |
| `HI` | 下位机 -> OpenMV | 表示串口链路已建立 |
| `OK` | OpenMV -> 下位机 | 表示视觉端已完成握手，可进入主控命令交互状态 |
| `SEEK,92,118,128.4` | OpenMV -> 下位机 | 捡球模式跟踪到网球时，上报当前水平/俯仰舵机角度与距离 |
| `PLAY,-1,-1,0.0` | OpenMV -> 下位机 | 刚进入击球模式或暂未锁到人时的占位包 |
| `PLAY,92,118,0.0` | OpenMV -> 下位机 | 击球模式锁到人后，上报当前水平/俯仰舵机角度，距离位固定为 `0.0` |
| `PICKED:1` / `COLLECT:1` | 下位机 -> OpenMV | 返回捡球执行完成反馈，OpenMV 收到后解除当前捡球跟踪 |
| `CTRL,MODE_PLAY,REQ,T01` | 下位机 -> OpenMV | 请求切换到击球模式 |
| `ACK,MODE_PLAY,REQ,T01` | OpenMV -> 下位机 | 已收到切换到击球模式的请求 |
| `CTRL,MODE_PLAY,CONFIRM,T01` | 下位机 -> OpenMV | 二次确认切换到击球模式 |
| `ACK,MODE_PLAY,CONFIRM,T01` | OpenMV -> 下位机 | 已接受切换到击球模式命令 |
| `CTRL,MODE_PICK,REQ,T02` | 下位机 -> OpenMV | 请求切换回捡球模式 |
| `ACK,MODE_PICK,REQ,T02` | OpenMV -> 下位机 | 已收到切换回捡球模式请求 |
| `CTRL,MODE_PICK,CONFIRM,T02` | 下位机 -> OpenMV | 二次确认切换回捡球模式 |
| `ACK,MODE_PICK,CONFIRM,T02` | OpenMV -> 下位机 | 已接受切换回捡球模式命令 |
| `CTRL,UNLOCK_TRACK,REQ,T03` | 下位机 -> OpenMV | 请求解除当前捡球跟踪目标 |
| `ACK,UNLOCK_TRACK,REQ,T03` | OpenMV -> 下位机 | 已收到解除跟踪请求 |
| `CTRL,UNLOCK_TRACK,CONFIRM,T03` | 下位机 -> OpenMV | 二次确认解除当前捡球跟踪目标 |
| `ACK,UNLOCK_TRACK,CONFIRM,T03` | OpenMV -> 下位机 | 已接受解除跟踪命令 |
| `NACK,ACTION,PHASE,TOKEN` | OpenMV -> 下位机 | 当前链路状态、模式状态或 token 不满足要求，拒绝执行 |
| `HIT,1\|92\|118,PLAY,TRACK_P` | OpenMV -> 下位机 | 补充事件报文：击球触发，参数分别为 `hit_flag|pan|tilt` |

## 3. 命令接受条件

### `MODE_PLAY`

- 当前链路状态必须已经到 `OK`
- 当前 OpenMV 必须处于 `MODE_PICK`

### `MODE_PICK`

- 当前链路状态必须已经到 `OK`
- 当前 OpenMV 必须处于 `MODE_PLAY`

### `UNLOCK_TRACK`

- 当前链路状态必须已经到 `OK`
- 当前 OpenMV 必须处于捡球模式的跟踪状态，也就是 `MODE_PICK + PICK_TRACK`
- 如果当前还在扫描、回位、确认阶段，OpenMV 会返回 `NACK`

## 4. OpenMV 发包行为

### 4.1 握手

- 上电后 OpenMV 立即发送一次 `hello`
- 如果还未收到 `HI`，OpenMV 每 5 秒继续发送一次 `hello`
- 收到 `HI` 后链路状态进入 `linked`
- OpenMV 随后发出 `OK`，链路状态进入 `ok`，此时控制命令生效

### 4.2 运行态上报

- 捡球模式下：
  - 仅在 `SEEK + TRACK` 阶段发送 `SEEK,pan,tilt,distance`
  - 当前实现按帧节流，每 5 帧发一次
- 击球模式下：
  - 切入 `MODE_PLAY` 后，立即发送一次占位包 `PLAY,-1,-1,0.0`
  - 在 `PLAY + SEARCH_P`，或者人目标未锁定/当前未实时可见时，每 5 帧发送一次 `PLAY,-1,-1,0.0`
  - 在 `PLAY + TRACK_P` 且人目标实时可见时，每 5 帧发送一次 `PLAY,pan,tilt,0.0`
  - 其中 `pan,tilt` 为当前水平/俯仰舵机角度整数值；击球模式当前不输出距离，固定填 `0.0`

### 4.3 事件上报

- 击球触发时，OpenMV 额外发送：

```text
HIT,1|pan|tilt,PLAY,TRACK_P
```

其中：

- `1` 表示本次触发有效
- `pan` 为当前水平舵机角度整数值
- `tilt` 为当前俯仰舵机角度整数值

## 5. 全过程流程图

```mermaid
flowchart TD
    A[OpenMV 上电启动] --> B[OpenMV 发送 hello]
    B --> C{下位机是否回复 HI}
    C -- 否 --> D[OpenMV 每 5 秒重发 hello]
    D --> C
    C -- 是 --> E[OpenMV 进入 linked]
    E --> F[OpenMV 回复 OK]
    F --> G[OpenMV 进入 ok]
    G --> H[默认进入 MODE_PICK]

    subgraph PICK[MODE_PICK 内部状态]
        H --> P1[SEEK/SCAN<br/>全局扫描网球]
        P1 --> P2{扫描完成且发现候选球}
        P2 -- 否 --> P1
        P2 -- 是 --> P3[SEEK/RETURN<br/>云台回到候选球方位]
        P3 --> P4[SEEK/CONFIRM<br/>局部确认候选球]
        P4 --> P5{确认锁定成功}
        P5 -- 否 --> P6{是否还有下一候选球}
        P6 -- 是 --> P3
        P6 -- 否 --> P1
        P5 -- 是 --> P7[SEEK/TRACK<br/>每 5 帧发送 SEEK,pan,tilt,distance]
        P7 --> P8{收到 PICKED/COLLECT}
        P8 -- 是 --> P1
        P8 -- 否 --> P9{收到已确认的 UNLOCK_TRACK}
        P9 -- 是 --> P1
        P9 -- 否 --> P10{目标连续丢失达到阈值}
        P10 -- 是 --> P1
        P10 -- 否 --> P7
    end

    subgraph CTRL[控制命令二阶段确认]
        C1[下位机发送 CTRL,ACTION,REQ,TOKEN] --> C2{链路状态是否为 ok}
        C2 -- 否 --> C9[OpenMV 回复 NACK,ACTION,PHASE,TOKEN]
        C2 -- 是 --> C3{当前状态是否允许该动作}
        C3 -- 否 --> C9
        C3 -- 是 --> C4[OpenMV 回复 ACK,ACTION,REQ,TOKEN]
        C4 --> C5[下位机发送 CTRL,ACTION,CONFIRM,TOKEN]
        C5 --> C6{token 匹配且未超时}
        C6 -- 否 --> C9
        C6 -- 是 --> C7{当前状态仍允许该动作}
        C7 -- 否 --> C9
        C7 -- 是 --> C8[OpenMV 回复 ACK,ACTION,CONFIRM,TOKEN]
    end

    P1 -. 可请求 MODE_PLAY .-> C1
    P3 -. 可请求 MODE_PLAY .-> C1
    P4 -. 可请求 MODE_PLAY .-> C1
    P7 -. 可请求 MODE_PLAY / UNLOCK_TRACK .-> C1

    C8 -- MODE_PLAY --> Q1
    C8 -- UNLOCK_TRACK --> P1

    subgraph PLAY[MODE_PLAY 内部状态]
        Q1[进入 MODE_PLAY] --> Q2[PLAY/SEARCH_P 或 PLAY/TRACK_P]
        Q2 --> Q3{是否实时锁到人}
        Q3 -- 否 --> Q4[立即或按节流发送 PLAY,-1,-1,0.0]
        Q3 -- 是 --> Q5[按节流发送 PLAY,pan,tilt,0.0]
        Q4 --> Q6{满足击球触发条件}
        Q5 --> Q6
        Q6 -- 是 --> Q7[额外发送 HIT,1|pan|tilt,PLAY,TRACK_P]
        Q6 -- 否 --> Q2
        Q7 --> Q2
    end

    Q2 -. 可请求 MODE_PICK .-> C1
    C8 -- MODE_PICK --> H
```

## 6. 完整通信时序示例

```text
OpenMV  -> hello
下位机   -> HI
OpenMV  -> OK

OpenMV  -> SEEK,92,118,128.4
OpenMV  -> SEEK,91,117,127.9

下位机   -> CTRL,MODE_PLAY,REQ,T01
OpenMV  -> ACK,MODE_PLAY,REQ,T01
下位机   -> CTRL,MODE_PLAY,CONFIRM,T01
OpenMV  -> ACK,MODE_PLAY,CONFIRM,T01

OpenMV  -> PLAY,-1,-1,0.0
OpenMV  -> PLAY,92,118,0.0
OpenMV  -> HIT,1|92|118,PLAY,TRACK_P

下位机   -> CTRL,MODE_PICK,REQ,T02
OpenMV  -> ACK,MODE_PICK,REQ,T02
下位机   -> CTRL,MODE_PICK,CONFIRM,T02
OpenMV  -> ACK,MODE_PICK,CONFIRM,T02

OpenMV  -> SEEK,88,121,96.7
下位机   -> CTRL,UNLOCK_TRACK,REQ,T03
OpenMV  -> ACK,UNLOCK_TRACK,REQ,T03
下位机   -> CTRL,UNLOCK_TRACK,CONFIRM,T03
OpenMV  -> ACK,UNLOCK_TRACK,CONFIRM,T03

OpenMV  -> SEEK,84,123,121.3
下位机   -> PICKED:1
OpenMV  -> SEEK,4,6,110.8
```

## 7. 下位机联调参考代码

下面给一个 Python 串口版参考程序，便于先把协议跑通。实际移植到 MCU 时，只需要保留同样的状态机和报文格式即可。

```python
import time
import serial


PORT = "COM6"
BAUD = 115200


class OpenMVLink:
    def __init__(self, port, baud):
        self.ser = serial.Serial(port, baud, timeout=0.1)
        self.linked = False
        self.ok = False
        self.token_id = 1

    def next_token(self):
        token = f"T{self.token_id:02d}"
        self.token_id += 1
        return token

    def send_line(self, line):
        print(f"MCU -> {line}")
        self.ser.write((line + "\n").encode("utf-8"))

    def read_line(self, timeout_s=3.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="ignore").strip()
            if text:
                print(f"OpenMV -> {text}")
                return text
        return None

    def wait_handshake(self):
        while True:
            line = self.read_line(timeout_s=10.0)
            if line == "hello":
                self.send_line("HI")
                self.linked = True
                ok_line = self.read_line(timeout_s=3.0)
                if ok_line == "OK":
                    self.ok = True
                    return
                raise RuntimeError(f"握手失败，期望收到 OK，实际收到: {ok_line}")

    def send_ctrl_action(self, action):
        token = self.next_token()

        self.send_line(f"CTRL,{action},REQ,{token}")
        ack_req = self.read_line()
        if ack_req != f"ACK,{action},REQ,{token}":
            raise RuntimeError(f"REQ not accepted: {ack_req}")

        self.send_line(f"CTRL,{action},CONFIRM,{token}")
        ack_confirm = self.read_line()
        if ack_confirm != f"ACK,{action},CONFIRM,{token}":
            raise RuntimeError(f"CONFIRM not accepted: {ack_confirm}")

    def run(self):
        self.wait_handshake()

        while True:
            line = self.read_line(timeout_s=0.2)
            if not line:
                continue

            if line.startswith("SEEK,"):
                print("收到捡球目标角度，上位控制器可据此执行指向。")
                pan, tilt, dist = line.split(",")[1:]
                print("pan =", pan, "tilt =", tilt, "dist =", dist)

                # 示例：先切换到击球模式
                self.send_ctrl_action("MODE_PLAY")

            elif line.startswith("PLAY,"):
                pan, tilt, dist = line.split(",")[1:]
                if pan == "-1" and tilt == "-1":
                    print("当前为击球模式占位包。")
                else:
                    print("收到击球模式人物角度: pan =", pan, "tilt =", tilt, "dist =", dist)

                # 示例：击球模式跑一段时间后切回捡球
                time.sleep(0.5)
                self.send_ctrl_action("MODE_PICK")

            elif line.startswith("HIT,"):
                print("收到击球事件:", line)

            elif line.startswith("ACK,") or line.startswith("NACK,"):
                print("控制命令响应:", line)


if __name__ == "__main__":
    link = OpenMVLink(PORT, BAUD)
    link.run()
```

## 8. MCU 侧实现建议

- 接收必须按“按行解析”实现，不要按固定长度截包
- `REQ` 和 `CONFIRM` 要使用同一个 token
- `token` 建议自增，例如 `T01`、`T02`、`T03`
- 如果收到 `NACK`，不要直接重发 `CONFIRM`，应先检查当前模式和状态是否满足
- `UNLOCK_TRACK` 只在捡球跟踪阶段使用，不要在扫描阶段发送
- `PICKED:1` / `COLLECT:1` 建议在执行机构动作真正完成后再发送
