# Branch-Predictor-using-LSTM-Attention-Mechanism

1. ./code : 

It contains 3 sub folders : common, simnlog and simpython.
common stores utility functions to read trace file in bt9 format.
simnlog stores how TAGE-SC-L is implemented, and it is from github repository provided by professor Chang.
simpython stores how my model and 2-bit FSM are implemented.

2. ./log : 

It contains logs generated by testing 3 different algorithm(TAGE-SC-L, 2bit FSM, our model) under chosen workload : SHORT_MOBILE-1, SHORT_MOBILE-3, SHORT_MOBILE-31, SHORT_MOBILE-37, SHORT_MOBILE-55, SHORT_MOBILE-61.

3. ./pattern_peek : 

It contains a sub folder storing all traces pattern plotted by matplotlib.

4. ./plot_numMispred : 

It contains numMispred v.s. time figures generated by testing 3 different algorithm(TAGE-SC-L, 2bit FSM, our model) under chosen workload : SHORT_MOBILE-1, SHORT_MOBILE-3, SHORT_MOBILE-31, SHORT_MOBILE-37, SHORT_MOBILE-55, SHORT_MOBILE-61.

5. ./plot_rollingAcc : 

It contains # branches currently executed v.s. # misprediction currently made divided by # branches currently executed figures generated by testing 3 different algorithm(TAGE-SC-L, 2bit FSM, our model) under chosen workload : SHORT_MOBILE-1, SHORT_MOBILE-3, SHORT_MOBILE-31, SHORT_MOBILE-37, SHORT_MOBILE-55, SHORT_MOBILE-61. 

The name “rollingAcc” might be confusing.

6. report PDF file
