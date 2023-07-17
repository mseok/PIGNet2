# CUDA devices should be recognized first.
# isort: off
import set_cuda

# isort: on
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# isort: off
import path
import utils
from data import ComplexDataModule


def run(
    model: torch.nn.Module,
    data: ComplexDataModule,
    device: torch.device,
):
    model.eval()

    for batch in data.test_dataloader():
        with torch.no_grad():
            model.test_step(batch.to(device))


@hydra.main(config_path="../config", config_name="config_test")
def main(config: DictConfig):
    assert config.run.checkpoint_file

    # `log_file` can have an additional subdir prepended.
    os.makedirs(os.path.dirname(os.path.realpath(config.run.log_file)), exist_ok=True)

    logger = utils.initialize_logger(config.run.log_file)
    logger.info(f"Current working directory: {os.getcwd()}")

    # Set GPUs.
    gpu_idx = utils.cuda_visible_devices(config.run.ngpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint.
    checkpoint = torch.load(config.run.checkpoint_file, map_location=device)
    config = utils.merge_configs(checkpoint["config"], config)
    logger.info(f"Load from: {os.path.realpath(config.run.checkpoint_file)}")

    logger.info(OmegaConf.to_yaml(config, resolve=True))
    logger.info(f"device: {repr(device)}, gpu_idx: {gpu_idx}")

    # Datamodule containing datasets & loaders
    data = ComplexDataModule(config)
    # Only one dataset is assumed per test run.
    (task,) = data.tasks

    # `test_result_path` can have an additional subdir prepended.
    os.makedirs(
        os.path.dirname(os.path.realpath(config.data[task].test_result_path)),
        exist_ok=True,
    )

    # Initialize the model.
    model, epoch = utils.initialize_state(device, checkpoint, config)

    # Tell if processed data are used.
    if dir_path := config.data[task].processed_data_dir:
        logger.info(
            f"Using processed data for '{task}' from: {os.path.realpath(dir_path)}"
        )

    # Tell data sizes.
    logger.info(f"Number of test data: {data.size[task][1]}")

    # Tell model size.
    logger.info(f"Number of parameters: {model.size[0]}")
    logger.info("")

    # Test starts.
    start_time = time.time()
    model.reset_log()
    run(model, data, device)

    test_losses = utils.get_losses(model)
    test_r, test_r2, test_tau = utils.get_stats(model, task)
    utils.write_predictions(model, config, False)

    end_time = time.time()

    # Print the header line.
    log_elements = [
        "epoch",
        "test_l",
        "test_l_dvdw",
        "test_r2",
        "test_r",
        "test_tau",
        "time",
    ]
    logger.info("\t".join(log_elements))
    # Print the loss values.
    log_elements = [
        str(epoch),
        utils.get_log_line([task], test_losses),
        "{:.3f}".format(test_r2),
        "{:.3f}".format(test_r),
        "{:.3f}".format(test_tau),
        "{:.3f}".format(end_time - start_time),
    ]
    logger.info("\t".join(log_elements))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    main()
