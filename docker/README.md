# Docker

构建、启动及镜像推送见[主 README 的 Docker 章节](../README.md#docker)。本页保留直接调用 Docker 的示例和镜像内容说明。

### Running a single command

```bash
docker run --rm \
    --gpus all \
    --network=host \
    --entrypoint /workspace/isaaclab/_isaac_sim/python.sh \
    -e ACCEPT_EULA=Y \
    robolab:<tag> \
    <script.py> [args...]
```

### Running with display (for GUI/rendering)

```bash
docker run --rm -it \
    --gpus all \
    --network=host \
    --entrypoint /bin/bash \
    -e ACCEPT_EULA=Y \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    robolab:<tag>
```

## What's in the image

- **Base**: `nvcr.io/nvidia/isaac-lab:3.0.0-beta2` (Isaac Sim 6.0)
- **Code**: `robolab/`, `scripts/`, `examples/`, `tests/`
- **Assets**: `assets/` (~6.5GB)
- **Python packages**: Core dependencies from `pyproject.toml`, installed via `pip install -e .`; the base image supplies the simulator stack
- **System tools**: `htop`, `nvtop`, `tmux`, `vim`, `git-lfs`, `zip`
