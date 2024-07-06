from predictor import BASEPREDICTOR, UINT64, OpType
from utils import *
import numpy as np
import os
import matplotlib.pyplot as plt

DEBUG = True
LOOPPREDICTOR = True


if LOOPPREDICTOR:
    LOGL = 5
    WIDTH_NBITER_LOOP = 10  # we predict only loops with less than 1K iterations
    LOOP_TAG = 10  # tag width in the loop predictor
    CONFLOOP = 15

SEED = 10
np.random.seed(SEED) 
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(SEED)

class Attention(nn.Module) :
    def __init__(self, hidden_dim, bidirectional):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        # The attention linear layer which transforms the input data to the hidden space
        self.attn = nn.Linear(self.hidden_dim * (4 if self.bidirectional else 2), self.hidden_dim * (2 if self.bidirectional else 1))
        # The linear layer that calculates the attention scores
        self.v = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), 1, bias=True)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        # Concatenate the last two hidden states in case of a bidirectional LSTM
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        else:
            hidden = hidden[-1]
        # Repeat the hidden state across the sequence length
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # Calculate attention weights
        attn_weights = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        # Compute attention scores
        attn_weights = self.v(attn_weights).squeeze(2)
        # Apply softmax to get valid probabilities
        return nn.functional.softmax(attn_weights, dim=1)

class AI_Predictor(nn.Module) :
    def __init__(self, system_address_bits, sub_pc_embedding_dim, hidden_dim, stacked_num = 1, attn = False, pretrained_weights = None, bidirectional = False, dropout=0.1) :
        super().__init__()
        self.system_address_bits = system_address_bits # what is the underlying system, 64-bit or 32-bit, used to determine the possible vocab size (address space) (how many distinct vocab this embedding layer needs to handle)
        self.sub_pc_embedding_dim = sub_pc_embedding_dim # how much dimension each PC number is embedded into ? ==> ex. PC = 0x1234 ==> [0.45, 0.73, -0.12, 0.3]
        self.hidden_dim = hidden_dim # how much dimension of each LSTM cells' hidden state 
        self.stacked_num = stacked_num # how many LSTM layers are stacked together
        self.attn = attn # whether to use attension mechanism or not
        # self.pretrained_weights = pretrained_weights # the pretrained weight of the embedding layer
        self.bidirectional = bidirectional # whether to use bidirectional LSTM or not
        self.dropout = dropout # dropout percentage to prevent from overfitting

        if self.system_address_bits == 32 : 
            self.embeddings_1 = nn.Embedding(2**16, self.sub_pc_embedding_dim)
            self.embeddings_1.weight.requires_grad = True
            self.embeddings_2 = nn.Embedding(2**16, self.sub_pc_embedding_dim)
            self.embeddings_2.weight.requires_grad = True
        if self.system_address_bits == 64 : 
            self.embeddings_1 = nn.Embedding(2**16, self.sub_pc_embedding_dim)
            self.embeddings_1.weight.requires_grad = True
            self.embeddings_2 = nn.Embedding(2**16, self.sub_pc_embedding_dim)
            self.embeddings_2.weight.requires_grad = True
            self.embeddings_3 = nn.Embedding(2**16, self.sub_pc_embedding_dim)
            self.embeddings_3.weight.requires_grad = True
            self.embeddings_4 = nn.Embedding(2**16, self.sub_pc_embedding_dim)
            self.embeddings_4.weight.requires_grad = True

        self.lstm = nn.LSTM(input_size=self.sub_pc_embedding_dim*int(self.system_address_bits/16), hidden_size=self.hidden_dim, num_layers=self.stacked_num, batch_first=True, bidirectional = self.bidirectional)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), self.hidden_dim * (2 if self.bidirectional else 1))
        self.dropout = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), 1)
        self.sigmoid = nn.Sigmoid()
        if self.attn:
            self.attention_layer = Attention(self.hidden_dim, self.bidirectional)
        
    def forward(self, pure_PCs_chunks):
        embedded_PC_vectors = None
        if self.system_address_bits == 32 : 
            embedded_PC_part1 = self.embeddings_1(pure_PCs_chunks[:,:,0])
            embedded_PC_part2 = self.embeddings_2(pure_PCs_chunks[:,:,1])
            embedded_PC_vectors = torch.cat((embedded_PC_part1, embedded_PC_part2), dim=-1)

        if self.system_address_bits == 64 : 
            embedded_PC_part1 = self.embeddings_1(pure_PCs_chunks[:,:,0])
            embedded_PC_part2 = self.embeddings_2(pure_PCs_chunks[:,:,1])
            embedded_PC_part3 = self.embeddings_3(pure_PCs_chunks[:,:,2])
            embedded_PC_part4 = self.embeddings_4(pure_PCs_chunks[:,:,3])
            embedded_PC_vectors = torch.cat((embedded_PC_part1, embedded_PC_part2, embedded_PC_part3, embedded_PC_part4), dim=-1)

        lstm_out, (ht, ct) = self.lstm(embedded_PC_vectors)
        context_vector = None
        attn_weights = None
        if self.attn == False:
            context_vector = self.dropout(ht[-1])
        else:
            # pad_out, _ = pad_packed_sequence(pack_out, batch_first=True)
            # desired_zero = self.step - pad_out.shape[1]
            # desired_pad = torch.zeros(pad_out.shape[0], desired_zero, pad_out.shape[2]).to(device)
            # pad_out = torch.cat((pad_out, desired_pad), dim=1)
            attn_weights = self.attention_layer(ht, lstm_out)
            context_vector = attn_weights.unsqueeze(1).bmm(lstm_out).squeeze(1)

        fc1_out = self.fc1(context_vector)
        relu_out = self.relu(fc1_out)
        dropout_out = self.dropout(relu_out)
        fc2_out = self.fc2(dropout_out)
        branch_taken_probability = self.sigmoid(fc2_out)
        return branch_taken_probability

class LEntry() :
    NbIter = 0         # 10 bits
    confid = 0         # 4 bits
    CurrentIter = 0    # 10 bits
    tag = 0            # 10 bits
    age = 0            # 4 bits
    dir = False        # 1 bit
    def __init__(self):
        self.NbIter = 0         # 10 bits
        self.confid = 0         # 4 bits
        self.CurrentIter = 0    # 10 bits
        self.TAG = 0            # 10 bits
        self.age = 0            # 4 bits
        self.dir = False        # 1 bit

class LOOP_Predictor():
    Ltable = []  # loop predictor table
    predloop = False  # loop predictor prediction
    Lib = 0
    Li = 0
    Lhit = 0  # hitting way in the loop predictor
    Ltag = 0  # tag on the loop predictor
    Lvalid = False  # validity of the loop predictor prediction
    withloop = -1  # counter to monitor whether or not loop prediction is beneficial

    # Loop predictor flag, used to identify whether the last prediction is based on loop predictor so we need to call ctrupdate()
    predbyloop = False
    predloop = False
    prednet = False

    def __init__(self):
        # Loop predictor variables
        self.Ltable = [LEntry() for _ in range(1 << LOGL)]  # loop predictor table
        self.predloop = False  # loop predictor prediction
        self.Lib = 0
        self.Li = 0
        self.Lhit = 0  # hitting way in the loop predictor
        self.Ltag = 0  # tag on the loop predictor
        self.Lvalid = False  # validity of the loop predictor prediction
        self.withloop = -1  # counter to monitor whether or not loop prediction is beneficial

    # Function to calculate index for loop predictor
    def lindex(self, PC):
        return ((PC & ((1 << (LOGL - 2)) - 1)) << 2)

    def MYRANDOM(self) :
        return np.random.randint(low=0, high=9223372036854775807)
    
    def ctrupdate (self, ctr, taken, nbits) :
        if taken:
            if ctr < ((1 << (nbits - 1)) - 1):
                ctr+=1
            else:
                pass
        else:
            if ctr > -(1 << (nbits - 1)):
                ctr-=1
            else:
                pass

    # bool GetPrediction(UINT64 PC)
    def GetPrediction(self,
                      PC: UINT64) -> bool:
        self.Lhit = -1
        self.Li = self.lindex(PC)
        self.Lib = ((PC >> (LOGL - 2)) & ((1 << (LOGL - 2)) - 1))
        self.Ltag = (PC >> (LOGL - 2)) & ((1 << (2 * LOOP_TAG)) - 1)
        self.Ltag ^= (self.Ltag >> LOOP_TAG)
        self.Ltag = (self.Ltag & ((1 << LOOP_TAG) - 1))

        for i in range(4):
            index = (self.Li ^ ((self.Lib >> i) << 2)) + i

            if self.Ltable[index].tag == self.Ltag:
                self.Lhit = i
                self.Lvalid = (self.Ltable[index].confid == CONFLOOP) or (self.Ltable[index].confid * self.Ltable[index].NbIter > 128)
                if self.Ltable[index].CurrentIter + 1 == self.Ltable[index].NbIter:
                    return not self.Ltable[index].dir
                else:
                    return self.Ltable[index].dir

        self.Lvalid = False
        return False

    # void UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir,
    #                      bool predDir, UINT64 branchTarget)
    def UpdatePredictor(self,
                        PC: UINT64,
                        opType: OpType,
                        resolveDir: bool,       # correct answer 
                        predDir: bool,          # our  prediction
                        branchTarget: UINT64):
        
        # code in C is in line 982
        # Update loop predictor
        # if LOOP_PREDICTOR and LVALID:
        #     if pred_taken != predloop:
        #         ctrupdate(WITHLOOP, predloop == resolveDir, 7)
        if self.Lvalid and predDir != self.predloop:
            self.ctrupdate(self.withloop, predDir == resolveDir, 7)

        if self.Lhit >= 0:
            index = (self.Li  ^ ((self.Lib >> self.Lhit) << 2)) + self.Lhit
            if self.Lvalid:
                if resolveDir != self.predloop:
                    # Free the entry
                    self.Ltable[index].NbIter = 0
                    self.Ltable[index].age = 0
                    self.Ltable[index].confid = 0
                    self.Ltable[index].CurrentIter = 0
                    return

                # need to modify!
                elif (self.predloop != self.prednet) or ((self.MYRANDOM() & 7) == 0):
                    if self.Ltable[index].age < CONFLOOP:
                        self.Ltable[index].age += 1

            self.Ltable[index].CurrentIter += 1
            self.Ltable[index].CurrentIter &= ((1 << WIDTH_NBITER_LOOP) - 1)
            if self.Ltable[index].CurrentIter > self.Ltable[index].NbIter:
                self.Ltable[index].confid = 0
                self.Ltable[index].NbIter = 0

            if resolveDir != self.Ltable[index].dir:
                if self.Ltable[index].CurrentIter == self.Ltable[index].NbIter:
                    if self.Ltable[index].confid < CONFLOOP:
                        self.Ltable[index].confid += 1
                    if self.Ltable[index].NbIter < 3:
                        # Just do not predict when the loop count is 1 or 2
                        self.Ltable[index].dir = resolveDir
                        self.Ltable[index].NbIter = 0
                        self.Ltable[index].age = 0
                        self.Ltable[index].confid = 0
                else:
                    if self.Ltable[index].NbIter == 0:
                        # First complete nest
                        self.Ltable[index].confid = 0
                        self.Ltable[index].NbIter = self.Ltable[index].CurrentIter
                    else:
                        # Not the same number of iterations as last time: free the entry
                        self.Ltable[index].NbIter = 0
                        self.Ltable[index].confid = 0
                self.Ltable[index].CurrentIter = 0

        elif resolveDir != predDir:
            X = self.MYRANDOM() & 3
            for i in range(4):
                self.Lhit = (X + i) & 3
                index = (self.Li ^ ((self.Lib >> self.Lhit) << 2)) + self.Lhit
                if self.Ltable[index].age == 0:
                    self.Ltable[index].dir = not resolveDir
                    self.Ltable[index].tag = self.Ltag
                    self.Ltable[index].NbIter = 0
                    self.Ltable[index].age = 7
                    self.Ltable[index].confid = 0
                    self.Ltable[index].CurrentIter = 0
                    break
                else:
                    self.Ltable[index].age -= 1
                    break
    
class EarlyStopper() :
    def __init__(self, min_tolerance_degrade = 1, PC_workingset_size = 500):
        self.min_tolerance_degrade = min_tolerance_degrade
        self.previous_val_loss = 100 ## just initialize it as a big loss value
        self.degrad_step = 0 
        self.total_counter = 0
        self.correct_counter = 0 
        self.PC_workingset_size = PC_workingset_size
    
    def to_stop_training(self, cur_val_loss):
        if cur_val_loss >= self.previous_val_loss:
            self.degrad_step += 1
            if self.degrad_step == self.min_tolerance_degrade :
                return True
        else:
            self.degrad_step = 0
        self.previous_val_loss = cur_val_loss
        return False
    
    def simple_to_stop_training(self, is_correct):
        self.total_counter += 1
        self.correct_counter += int(is_correct == True)
        if self.total_counter == self.PC_workingset_size:
            if self.correct_counter >= int(self.PC_workingset_size*0.9):
                return True
            else:
                self.total_counter = 0
                self.correct_counter = 0
        return False

class PREDICTOR(BASEPREDICTOR):
    # system configuration
    system_address_bits = 64

    # network configuration
    sub_pc_embedding_dim = 100
    hidden_dim = 128
    stacked_num = 1
    attn = True
    bidirectional = True
    PC_workingset_size = 500
    batch_size = 100

    # system history table
    history_table = Container(max_size = PC_workingset_size)
    for i in range(PC_workingset_size):
        history_table.push(0)

    # batch update container
    prediction_in_batch = []
    groundtruth_in_batch = []

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # network initialization
    network = AI_Predictor(
        system_address_bits = system_address_bits, \
        sub_pc_embedding_dim = sub_pc_embedding_dim, \
        hidden_dim = hidden_dim, \
        stacked_num = stacked_num, \
        attn = attn, \
        bidirectional = bidirectional
    )
    network.to(device)
    print(network)

    # network's loss function and optimizer configuration
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # network early-stopper
    early_stopper = EarlyStopper(PC_workingset_size = PC_workingset_size)
 
    # network early-stoping identifier
    is_early_stopped = False
    
    # records related
    record_prefix = "test"
    numMispred_container = []
    pattern_container = []
    rolling_accuracy = []

    # Loop predictor
    loop_predictor = LOOP_Predictor()

    # bool GetPrediction(UINT64 PC)
    def GetPrediction(self, 
                      PC: UINT64,
                      Speculative_PC: UINT64
                      ) -> bool:
        # if DEBUG :
        #     print('GetPrediction | PC =', PC
        #     )

        predloop = self.loop_predictor.GetPrediction(PC)
        
        PCs_chunks_tensor = create_PCs_chunks(PC, self.history_table.container, Speculative_PC, self.PC_workingset_size)
        PCs_chunks_tensor = PCs_chunks_tensor.to(self.device)
        if not self.is_early_stopped: # no matter this prediction is made by loop predictor or not, we train the model
            self.network.train()
            branch_taken_probability = self.network(PCs_chunks_tensor) # shape [1,1], 1 prediction made each time and 1 sigmoid prob value
            self.history_table.push(PC)
            self.prediction_in_batch.append(branch_taken_probability)
        else:
            self.network.eval()
            branch_taken_probability = self.network(PCs_chunks_tensor) # shape [1,1], 1 prediction and 1 sigmoid prob value
        
        prednet = bool(branch_taken_probability[0] > 0.5) # why [0] is needed : because we have shape [1,1], not [1]

        self.loop_predictor.predloop = predloop
        self.loop_predictor.prednet = prednet
        if LOOPPREDICTOR and (self.loop_predictor.withloop >= 0) and (self.loop_predictor.Lvalid) :
            self.loop_predictor.predbyloop = True
            return predloop
        else:
            self.loop_predictor.predbyloop = False
            return prednet

    # void UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir, bool predDir, UINT64 branchTarget)
    def UpdatePredictor(self,
                        PC: UINT64,
                        opType: OpType,
                        resolveDir: bool,
                        predDir: bool,
                        branchTarget: UINT64,
                        numIter: UINT64,
                        branch_instruction_counter: UINT64,
                        ):
        # if DEBUG :
            # print(
            #     'UpdatePredictor | PC = {} OpType = {} resolveDir = {} '
            #     'predDir = {} predbyloop = {} branchTarget = {} is_early_stopped = {} progress = {}/{} device = {}'.format(
            #         PC, opType, resolveDir, predDir, self.loop_predictor.predbyloop, branchTarget, self.is_early_stopped, numIter, branch_instruction_counter, self.device)
            # )

        self.loop_predictor.UpdatePredictor(PC, opType, resolveDir, predDir, branchTarget)

        if self.is_early_stopped:
            return
        
        self.groundtruth_in_batch.append(torch.tensor([[int(resolveDir)]], dtype = torch.float))

        if len(self.prediction_in_batch) == self.batch_size: 
            assert(len(self.prediction_in_batch) == len(self.groundtruth_in_batch))
            prediction_in_batch_tensor, groundtruth_in_batch_tensor = create_batch_tensor(self.prediction_in_batch, self.groundtruth_in_batch, self.device)
            loss = self.loss_fn(prediction_in_batch_tensor, groundtruth_in_batch_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.prediction_in_batch = [] # don't forget to empty the lists after each time we update
            self.groundtruth_in_batch = []

        # self.is_early_stopped = self.early_stopper.simple_to_stop_training(bool(resolveDir == predDir))
        # self.is_early_stopped = False

    def TrackOtherInst(self,
                       PC: UINT64,
                       previous_PC: UINT64,
                       opType: OpType,
                       taken: bool,
                       branchTarget: UINT64,
                       numIter: UINT64,
                       branch_instruction_counter: UINT64):
        # self.pattern_container.append(PC)
        # if numIter == branch_instruction_counter:
        #     np.save("./ckpt_pattern/{}.npy".format(self.record_prefix + "_ckpt_pattern"), self.pattern_container)
        #     assert np.array_equal(np.load("./ckpt_pattern/{}.npy".format(self.record_prefix + "_ckpt_pattern")), self.pattern_container)
        #     plt.title("{}_pattern".format(self.record_prefix))
        #     plt.scatter(range(len(self.pattern_container)), self.pattern_container, s=1)
        #     plt.savefig("./plot_pattern/{}.png".format(self.record_prefix + "_plot_pattern"))
        #     plt.clf() 
        pass
    
    def RecordResult(self, 
                    numIter: UINT64,
                    numMispred: UINT64, 
                    branch_instruction_counter: UINT64):
        self.numMispred_container.append(numMispred)
        self.rolling_accuracy.append(float("{:.4f}".format(numMispred/numIter)))
        if numIter == branch_instruction_counter:
            np.save("./ckpt_numMispred/{}.npy".format(self.record_prefix + "_ckpt_numMispred"), self.numMispred_container)
            assert np.array_equal(np.load("./ckpt_numMispred/{}.npy".format(self.record_prefix + "_ckpt_numMispred")), self.numMispred_container)
            plt.title("{}_numMispred".format(self.record_prefix))
            plt.plot(range(len(self.numMispred_container)), self.numMispred_container)
            plt.savefig("./plot_numMispred/{}.png".format(self.record_prefix + "_plot_numMispred"))
            plt.clf() 

            np.save("./ckpt_rollingAcc/{}.npy".format(self.record_prefix + "_ckpt_rollingAcc"), self.rolling_accuracy)
            assert np.array_equal(np.load("./ckpt_rollingAcc/{}.npy".format(self.record_prefix + "_ckpt_rollingAcc")), self.rolling_accuracy)
            plt.title("{}_rollingAcc".format(self.record_prefix))
            plt.plot(range(len(self.rolling_accuracy)), self.rolling_accuracy)
            plt.savefig("./plot_rollingAcc/{}.png".format(self.record_prefix + "_plot_rollingAcc"))
            plt.clf() 
            
