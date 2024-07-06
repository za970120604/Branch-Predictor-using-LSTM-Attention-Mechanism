from predictor import BASEPREDICTOR, UINT64, OpType
import numpy as np
import os
import matplotlib.pyplot as plt

DEBUG = True
STRONGLYTAKEN    = 3
WEAKLYTAKEN      = 2
WEAKLYNOTTAKEN   = 1
STRONGLYNOTTAKEN = 0
PHT_ENTRIES = 1024 

class PREDICTOR(BASEPREDICTOR):
    state = []
    # records related
    record_prefix = "2BITFSM_SHORT_MOBILE-39"
    numMispred_container = []
    rolling_accuracy = []

    def __init__(self):
        for _ in range(PHT_ENTRIES):
            self.state.append(STRONGLYNOTTAKEN)

    # bool GetPrediction(UINT64 PC)
    def GetPrediction(self,
                      PC: UINT64,
                      Speculative_PC: UINT64
                      ) -> bool:
        if DEBUG:
            print('GetPrediction | PC =', PC)
        # Always predict taken
        if self.state[(PC >> 2) % PHT_ENTRIES] == STRONGLYTAKEN or self.state[(PC >> 2) % PHT_ENTRIES] == WEAKLYTAKEN:
            return True
        else:
            return False

    # void UpdatePredictor(UINT64 PC, OpType opType, bool resolveDir,
    #                      bool predDir, UINT64 branchTarget)
    def UpdatePredictor(self,
                        PC: UINT64,
                        opType: OpType,
                        resolveDir: bool,
                        predDir: bool,
                        branchTarget: UINT64,
                        numIter: UINT64,
                        branch_instruction_counter: UINT64,
                        ):
        if DEBUG :
            print(
                'UpdatePredictor | PC = {} OpType = {} resolveDir = {} '
                'predDir = {} branchTarget = {} is_early_stopped = {} progress = {}/{} device = {}'.format(
                    PC, opType, resolveDir, predDir, branchTarget, 'false', numIter, branch_instruction_counter, 'cpu')
            )

        if resolveDir == True:
            if self.state[(PC >> 2) % PHT_ENTRIES] == WEAKLYTAKEN:
                self.state[(PC >> 2) % PHT_ENTRIES] = STRONGLYTAKEN
            elif self.state[(PC >> 2) % PHT_ENTRIES] == WEAKLYNOTTAKEN:
                self.state[(PC >> 2) % PHT_ENTRIES] = STRONGLYTAKEN
            elif self.state[(PC >> 2) % PHT_ENTRIES] == STRONGLYNOTTAKEN:
                self.state[(PC >> 2) % PHT_ENTRIES] = WEAKLYNOTTAKEN
        else:
            if self.state[(PC >> 2) % PHT_ENTRIES] == STRONGLYTAKEN:
                self.state[(PC >> 2) % PHT_ENTRIES] = WEAKLYTAKEN
            elif self.state[(PC >> 2) % PHT_ENTRIES] == WEAKLYTAKEN:
                self.state[(PC >> 2) % PHT_ENTRIES] = STRONGLYNOTTAKEN
            elif self.state[(PC >> 2) % PHT_ENTRIES] == WEAKLYNOTTAKEN:
                self.state[(PC >> 2) % PHT_ENTRIES] = STRONGLYNOTTAKEN
        
    def TrackOtherInst(self,
                       PC: UINT64,
                       previous_PC: UINT64,
                       opType: OpType,
                       taken: bool,
                       branchTarget: UINT64,
                       numIter: UINT64,
                       branch_instruction_counter: UINT64):
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
