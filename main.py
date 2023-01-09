from dataset import IMDBBertDataset
from model import BERT
from trainer import BertTrainer


if __name__=="__main__":
    # Parameters
    ## Data path
    ibdb_data_path = "./data/IMDB-Dataset.csv"
    ## Model parameters
    dim_inp = 128
    dim_out = 512
    n_attention_heads = 2
    ## Training parameters
    log_dir = './checkpoints/tensorboard_logdir'
    checkpoint_dir = './checkpoints'
    print_progress_every = 500
    batch_size = 24
    learning_rate = 0.005
    n_epochs = 10
    device = 'cuda'
    debug = False

    # Prepare dataset
    if debug:
        ds_from, ds_to = 0, 500
    else:
        ds_from, ds_to = None, None
    dataset = IMDBBertDataset(ibdb_data_path,
                                ds_from=ds_from, ds_to=ds_to)

    # Prepare the model
    model = BERT(
                vocab_size=len(dataset.vocab),
                dim_inp=dim_inp,
                dim_out=dim_out,
                attention_heads=n_attention_heads
            )
    model.to(device)

    # Bert Trainer
    trainer = BertTrainer(
                model, dataset,
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
                print_progress_every=print_progress_every,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=n_epochs,
                device=device,
            )

    # Train the model
    trainer()
