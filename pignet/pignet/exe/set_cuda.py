import os
import shutil
import subprocess


def cuda_visible_devices(
    num_gpus: int,
    max_num_gpus: int = 16,
) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    idle_gpus = []

    if num_gpus and shutil.which("nvidia-smi") is not None:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]
            proc = subprocess.run(cmd, capture_output=True, text=True)

            if "No devices were found" in proc.stdout:
                break

            if "No running" in proc.stdout:
                idle_gpus.append(i)

            if len(idle_gpus) >= num_gpus:
                break

        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)

        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

    else:
        idle_gpus = ""

    return idle_gpus


try:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        idx = cuda_visible_devices(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
except RuntimeError:
    pass
