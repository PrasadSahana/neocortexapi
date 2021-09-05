# Investigation of HTM layer instability in HTM

### What is HTM layer instability?

Hierarchical Temporal Memory (HTM) is designed in such a way that it can learn the pattern and generate Sparse Distributed Representation (SDR) of the input quickly . 
Initially after the research it was found that SDRs can be forgotten during the training progress and this caused SP to learn same patterns again, which generated new set of SDRs. This instable learning behavior of SP was because of internal boosting algorithm by Homeostatic Plasticity Mechanism. ([See the link)](https://github.com/ddobric/neocortexapi/blob/htm-serialization/source/Documentation/Experiments/ICPRAM_2021_76_CR.pdf)

### What were the problems in layer L4 of SP + TM?

During the research it was found that after SP gets stable in New-born stage ([See the link](https://github.com/ddobric/neocortexapi/blob/htm-serialization/source/Documentation/Experiments/ICPRAM_2021_76_CR.pdf)), TM is not stable. This means that layer L4 gets stable in SP (by using Homeostatic Plasticity Controller), unfortunately the TM in same layer do not get stable. The SDR visualization result in the above link ([See the link](https://github.com/PrasadSahana/neocortexapi/blob/master/source/Documentation/images/sdr-compare.png)) shows that same image in different cycles change the SDR Pattern.
It is observed that Cell SDRs of the same sequence key were changing over time even if SP was stable. SP being stable means that **Column SDRs do not change anymore**, but it stops changing after the TM (Cell SDRs) gets stable. For this to happen, all patterns must attain 100% accuracy without any change in Cell/Column SDRs. Hence, the stability check was implemented in the class ‘SequenceLearning.cs’.

### How stability was achieved in the class ‘SequenceLearning.cs’ of layer L4  SP + TM? 

A new version of HTM classifier was introduced which implemented the following method as follows:

```csharp
	public List<ClassifierResult> GetPredictedInputValues(Cell[] predictiveCells, short howMany)
```

Hence the SequenceLearning.cs has a new method:

```csharp
	var predictedInputValue = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);
```

This method retrieves top 3 predicted values from the list having Similarity >= 50%. Apart from that there was also drop in the accuracy during learning process, which started producing new set of cell SDRs again.

### 1.	SequenceLearning.cs learning Process:

Consider an example input sequence as follows:

```csharp
	List<double> inputValues = new List<double>(new double[] { 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 7.0, 1.0, 9.0, 12.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 14.0, 5.0, 7.0, 6.0, 9.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0 });
```

So, when this input sequence starts learning, then SP gets stable in New-born stage. After the New-born stage, TM starts learning cell SDRs in order to attain 100% accuracy. During initial stages of learning process, the Similarity % may not be 100 as shown below.

```
-------------- Cycle 259 ---------------						
-------------- 0 ---------------						
						
Active segments: 40, Matching segments: 40						
Col  SDR: 83, 84, 87, 99, 147, 148, 156, 158, 161, 176, 231, 237, 250, 253, 258, 262, 268, 280, 282, 283, 285, 293, 297, 302, 305, 313, 316, 317, 320, 322, 323, 324, 326, 335, 336, 338, 341, 345, 377, 383, 						
Cell SDR: 2091, 2110, 2188, 2488, 3682, 3700, 3907, 3967, 4042, 4406, 5789, 5928, 6269, 6342, 6457, 6558, 6702, 7009, 7069, 7080, 7149, 7330, 7438, 7561, 7634, 7825, 7916, 7926, 8007, 8053, 8088, 8100, 8165, 8391, 8418, 8469, 8548, 8640, 9433, 9582, 						
Match. Actual value: 1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0 - Predicted value: 1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0						
Item length: 40	 Items: 34					
Predictive cells: 40 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7971, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9315, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 					
>indx:0	inp/len: 0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1/40 ,Same Bits = 40	, Similarity% 100 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7971, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9315, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 			
>indx:1	inp/len: 9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6/40 ,Same Bits = 39	, Similarity% 97,5 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7997, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9315, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 			
>indx:2  inp/len: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3/40 ,Same Bits = 39	, Similarity% 97,5 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7971, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9275, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 				
Current Input: 0						
Predicted Input: 0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1						
Predicted Input: 9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6						
Predicted Input: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3
```

However, as learning continues, Similarity % also increases and eventually it attains 100%.	It is also interesting to note thatduring learning process one pattern leads to another pattern and so on.

```
-------------- Cycle 271 ---------------						
-------------- 0 ---------------						
						
Active segments: 40, Matching segments: 40						
Col  SDR: 83, 84, 87, 99, 147, 148, 156, 158, 161, 176, 231, 237, 250, 253, 258, 262, 268, 280, 282, 283, 285, 293, 297, 302, 305, 313, 316, 317, 320, 322, 323, 324, 326, 335, 336, 338, 341, 345, 377, 383, 						
Cell SDR: 2091, 2110, 2188, 2488, 3682, 3700, 3907, 3967, 4042, 4406, 5789, 5928, 6269, 6342, 6457, 6558, 6702, 7009, 7069, 7080, 7149, 7330, 7438, 7561, 7634, 7825, 7916, 7926, 8007, 8053, 8088, 8100, 8165, 8391, 8418, 8469, 8548, 8640, 9433, 9582, 						
Match. Actual value: 1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0 - Predicted value: 1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0						
Item length: 40	 Items: 34					
Predictive cells: 40 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7971, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9315, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 					
>indx:0	inp/len: 0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1/40 ,Same Bits = 40	, Similarity% 100 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7971, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9315, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 			
>indx:1	inp/len: 9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6/40 ,Same Bits = 40	, Similarity% 100 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7971, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9315, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 			
>indx:2	inp/len: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3/40 ,Same Bits = 40	, Similarity% 100 	 4449, 5898, 6021, 6570, 7052, 7129, 7443, 7625, 7835, 7898, 7928, 7971, 8022, 8143, 8159, 8342, 8388, 8415, 8471, 8545, 8645, 8888, 8907, 8929, 8967, 8983, 9228, 9315, 9405, 9426, 9598, 9620, 9648, 9993, 10034, 10085, 10543, 10744, 10880, 11567, 			
Current Input: 0						
Predicted Input: 0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1						
Predicted Input: 9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6						
Predicted Input: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3
```

Once the TM learns SDR patterns completely, then accuracy (30 times) is reached. A method is added in our code to generate ‘cell state trace’ so that statistical analysis can be done. This helps to compare column/cell activity. Consider the 5 SDRs generated for sequence shown above.

```
---- cell state trace ----	
0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4						
12235, 12326, 12362, 12392, 12459, 12672, 13228, 14345, 14497, 15106, 15379, 15431, 15486, 15585, 15608, 15671, 15679, 15758, 15883, 15904, 15936, 15996, 16037, 16145, 16233, 16303, 16370, 16490, 16605, 16887, 16984, 17099, 17140, 17209, 17299, 17499, 17528, 17572, 17786, 18386, 						
12235, 12326, 12362, 12392, 12459, 12672, 13228, 14345, 14497, 15106, 15379, 15431, 15486, 15585, 15608, 15671, 15679, 15758, 15883, 15904, 15936, 15996, 16037, 16145, 16233, 16303, 16370, 16490, 16605, 16887, 16984, 17099, 17140, 17209, 17299, 17499, 17528, 17572, 17786, 18386, 						
12235, 12326, 12362, 12392, 12459, 12672, 13228, 14345, 14497, 15106, 15379, 15431, 15486, 15585, 15608, 15671, 15679, 15758, 15883, 15904, 15936, 15996, 16037, 16145, 16233, 16303, 16370, 16490, 16605, 16887, 16984, 17099, 17140, 17209, 17299, 17499, 17528, 17572, 17786, 18386, 						
12235, 12326, 12362, 12392, 12459, 12672, 13228, 14345, 14497, 15106, 15379, 15431, 15486, 15585, 15608, 15671, 15679, 15758, 15883, 15904, 15936, 15996, 16037, 16145, 16233, 16303, 16370, 16490, 16605, 16887, 16984, 17099, 17140, 17209, 17299, 17499, 17528, 17572, 17786, 18386, 						
12235, 12326, 12362, 12392, 12459, 12672, 13228, 14345, 14497, 15106, 15379, 15431, 15486, 15585, 15608, 15671, 15679, 15758, 15883, 15904, 15936, 15996, 16037, 16145, 16233, 16303, 16370, 16490, 16605, 16887, 16984, 17099, 17140, 17209, 17299, 17499, 17528, 17572, 17786, 18386, 						
```

Now, we need to visualize this learnt SDRs to check for any instability. The below result shows visual representation of SDR1/2/3/4/5 of the above mentioned sequence.

![][img0.1]

[img0.1]: ./Visualized SDR Comparison/SDR_Comparison_Sequence_1_Stable.JPG


Hence, we can conclude that there are absolutely no instability caused during learning process of Cell SDR patterns.

### How did we generate Visualized SDRs?

The Column/Cell activity of SDRs can be generated using the python script draw_figure.py which is located in the below mentioned link.

[See the link](https://github.com/PrasadSahana/neocortexapi/blob/master/Python/ColumnActivityDiagram/draw_figure.py)