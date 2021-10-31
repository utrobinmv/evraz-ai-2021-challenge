#docker run --gpus all  --device /dev/nvidia0:/dev/nvidia0  --device /dev/nvidiactl:/dev/nvidiactl  --device /dev/nvidia-caps:/dev/nvidia-caps  --device /dev/nvidia-uvm:/dev/nvidia-uvm  --device /dev/nvidia-modeset:/dev/nvidia-modeset  --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools -t evraz-rover
nvidia-docker run --rm \
  --gpus=all \
  --ipc=host \
    evraz-rover

