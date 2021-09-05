# Investigation of HTM layer instability in HTM

### What is HTM layer instability?

Hierarchical Temporal Memory (HTM) is designed in such a way that it can learn the pattern and generate Sparse Distributed Representation (SDR) of the input quickly . 
Initially after the research it was found that SDRs can be forgotten during the training progress and this caused SP to learn same patterns again, which generated new set of SDRs. This instable learning behavior of SP was because of internal boosting algorithm by Homeostatic Plasticity Mechanism. ([See the link)](https://github.com/ddobric/neocortexapi/blob/htm-serialization/source/Documentation/Experiments/ICPRAM_2021_76_CR.pdf)

### What were the problems in layer L4 of SP + TM?

During the research it was found that after SP gets stable in New-born stage ([See the link](https://github.com/ddobric/neocortexapi/blob/htm-serialization/source/Documentation/Experiments/ICPRAM_2021_76_CR.pdf)), TM is not stable. This means that layer L4 gets stable in SP (by using Homeostatic Plasticity Controller), unfortunately the TM in same layer do not get stable. The below SDR visualization result shows that same image in different cycles change the SDR code.
![][Image1]

[Image1]: ./Desktop/Picture1.png

