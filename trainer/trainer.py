from utils import inf_loop


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config, lr_scheduler)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        progress = tqdm(enumerate(self.data_loader), desc="Loss: ", total=len(train_dataloader))
        total_acc = 0
        for batch_idx, (data, target) in progress:
            X = X.to(self.device); y= y.to(self.device)
            pred = model(X)
            loss = self.criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses += loss.detach().item()
            size += len(X)

            total_acc += self.metric_ftns(pred, y)
            train_loss = losses/(batch+1)
            train_acc = total_acc/(batch+1)
            progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(train_loss, train_acc))
        result = {
            "train_loss": train_loss,
            "train_acc": train_acc
        }
        return result

    def valid_epochs(self, epoch):
        size = 0
        num_batches = len(self.valid_data_loader)
        valid_loss, total_acc = 0, 0

        self.model.eval()
        for X, y in self.valid_data_loader:
            X = X.to(self.device); y= y.to(self.device)
            pred = self.model(X)

            loss = self.criterion(pred, y)
            test_loss += loss.detach().item()
            total_acc += self.metric_ftns(pred, y)

        valid_loss /= num_batches
        valid_acc = total_acc / num_batches
        result = {
            "valid_loss": valid_loss,
            "valid_acc":valid_acc
        }
        return result
            