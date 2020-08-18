import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import process_data_th
import BilstmCrfStringBatch
import Trainer

START_TAG = "<START>"
STOP_TAG = "<STOP>"

EPOCHS = 20
BATCH_SIZE = 256
EMBEDDING_DIM = 200
HIDDEN_DIM = 200

(train_x, train_y), (test_x, test_y), (vocab, chunk_tags) = process_data_th.load_data()

tag2id = {}
for tag in chunk_tags:
    tag2id.setdefault(tag, len(tag2id))
tag2id.setdefault(START_TAG, len(tag2id))
tag2id.setdefault(STOP_TAG, len(tag2id))
print("tag2id: " + str(len(tag2id)))
print("data max len: " + str(len(train_x[0])))

train_x = torch.tensor(train_x).to(torch.int64)
train_y = torch.tensor(train_y).to(torch.int64)

model = BilstmCrfStringBatch.BilstmCrfStringBatch(len(vocab), tag2id, EMBEDDING_DIM, HIDDEN_DIM, vocab, device="cuda").cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

trainer = Trainer.Trainer(model, optimizer)
trainer.train(train_x, train_y, EPOCHS, BATCH_SIZE)
model = model.cpu()
model.device = "cpu"
print(model("100臺北市中正區中山南路21號"))
torch.save(model, "model/BilstmCrfStringBatch_v1.pt")
