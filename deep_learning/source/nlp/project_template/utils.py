from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_logger(args):
    '''
    Get logger for PyTorch Lightning.
    args:
        logger_name: str,
        project_name: str
        run_name: str
        save_dir: str
    '''
    logger = None
    if args.logger_name == 'wandb':
        logger = WandbLogger(
            project=args.project_name,
            name=args.run_name,
            save_dir=args.save_dir,
        )
    return logger


def get_callbacks(
    monitor: str,
    min_delta: float,
    patience: int,
    mode: str,
    checkpoint_path: str,
    checkpoint_name: str,
):
    '''
    Get callbacks for PyTorch Lightning.
    '''
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        mode=mode,)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=checkpoint_name,
        verbose=True,
        monitor=monitor,
        mode=mode
    )
    return [early_stop_callback, checkpoint_callback]
