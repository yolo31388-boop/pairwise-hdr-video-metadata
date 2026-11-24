# pairwise-hdr-video-metadata

## 领域

游戏、图形与多媒体

## 主题

HDR 视频逐帧元数据映射

## 说明

按帧元数据、传输函数分支与参数平滑。

初始状态下 `media/hdr_video_metadata.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
