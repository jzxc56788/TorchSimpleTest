import time

import torch.utils.data as Data


class Trainer():
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def train(self, x, y, epochs = 2, batch_size = 1):
        train_dataset = Data.TensorDataset(x, y)
        loader = Data.DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
        )
        
        start_time = time.time()
        step_size = len(loader)
        for epoch in range(epochs):
            for step, (batch_x, batch_y) in enumerate(loader):
                step_time = time.time()
                self.model.zero_grad()
                loss = self.model.neg_log_likelihood(batch_x.cuda(), batch_y.cuda())
                print('Epoch: %i | Step: %i/%i | Loss: %.2f | time: %.2f s' % (epoch, step, step_size, loss, time.time() - step_time))
                loss.backward()
                self.optimizer.step()
        print('all time : ', time.time() - start_time,'s')
