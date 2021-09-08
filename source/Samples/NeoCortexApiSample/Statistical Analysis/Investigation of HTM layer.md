# Investigation of HTM layer instability in HTM

### What is HTM layer instability?

Hierarchical Temporal Memory (HTM) is designed in such a way that it can learn the patterns and generate Sparse Distributed Representation (SDR) of the input quickly . 
Initially after the research it was found that SDRs can be forgotten during the training progress and this caused SP to learn same patterns again, which generated new set of SDRs. This instable learning behavior of SP was because of internal boosting algorithm by Homeostatic Plasticity Mechanism. ([See the link)](https://github.com/ddobric/neocortexapi/blob/htm-serialization/source/Documentation/Experiments/ICPRAM_2021_76_CR.pdf)

### How stability was achieved in the class ‘SequenceLearning.cs’ of layer L4  SP + TM? 

A new version of HTM classifier was introduced which implemented the following method as follows:

```csharp
	public List<ClassifierResult> GetPredictedInputValues(Cell[] predictiveCells, short howMany)
```

Hence the SequenceLearning.cs has a new method:

```csharp
	var predictedInputValue = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);
                foreach (var item in predictedInputValue)
                {
                    if (item.Similarity >= (double)50.00 && item.PredictedInput.Contains("-1.0") == false)
                    {
                        Debug.WriteLine($"Current Input: {input}, Predicted Input: {item.PredictedInput}, Similarity %: {item.Similarity}");
                    }
                }
                lastPredictedValue = predictedInputValue.First().PredictedInput;
```

This method retrieves top 3 predicted values from the list having Similarity >= 50%. Apart from that there was also drop in the accuracy during learning process, which started producing new set of cell SDRs again.

### 1.	SequenceLearning.cs learning Process:

Consider an example input sequence as follows:

```csharp
	List<double> inputValues = new List<double>(new double[] { 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 7.0, 1.0, 9.0, 12.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 14.0, 5.0, 7.0, 6.0, 9.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0 });
```

When this input sequence starts learning, then SP gets stable in New-born stage. After the New-born stage, TM starts learning cell SDRs in order to attain accuracy >90%. During initial stages of learning process, the Similarity % may not be high as shown below.


```
-------------- Cycle 161 ---------------						
-------------- 6 ---------------			
			
Active segments: 40, Matching segments: 40			
Col  SDR: 503, 543, 546, 570, 575, 615, 618, 638, 674, 681, 698, 717, 725, 730, 745, 746, 750, 772, 774, 777, 778, 788, 799, 801, 834, 846, 853, 856, 860, 863, 871, 874, 875, 882, 883, 891, 893, 894, 900, 915, 			
Cell SDR: 12583, 13598, 13656, 14256, 14384, 15389, 15469, 15966, 16858, 17036, 17461, 17931, 18144, 18262, 18625, 18651, 18768, 19323, 19367, 19434, 19461, 19712, 19982, 20029, 20853, 21174, 21345, 21413, 21521, 21585, 21795, 21867, 21879, 22061, 22093, 22299, 22342, 22357, 22510, 22876, 			
Match. Actual value: 5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6 - Predicted value: 5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6			
Item length: 40	 Items: 34		
Predictive cells: 40 	 10653, 10861, 10906, 11480, 12500, 12669, 13011, 13651, 14100, 14270, 14390, 15267, 15647, 15963, 16538, 16673, 16743, 17046, 17471, 17479, 17525, 17757, 17890, 18025, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19319, 19351, 19435, 19491, 19718, 19751, 19785, 19863, 		
>indx:0	inp/len: 4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5/40 ,Same Bits = 40	, Similarity% 100 	 10653, 10861, 10906, 11480, 12500, 12669, 13011, 13651, 14100, 14270, 14390, 15267, 15647, 15963, 16538, 16673, 16743, 17046, 17471, 17479, 17525, 17757, 17890, 18025, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19319, 19351, 19435, 19491, 19718, 19751, 19785, 19863, 
>indx:1	inp/len: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3/40 ,Same Bits = 36	, Similarity% 90 	 10653, 10861, 10906, 11480, 12499, 12669, 13011, 13651, 14100, 14270, 14390, 15265, 15647, 15963, 16538, 16673, 16743, 17043, 17471, 17479, 17525, 17757, 17890, 18021, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19319, 19351, 19435, 19491, 19718, 19751, 19785, 19863,
>indx:2	inp/len: 4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3/40 ,Same Bits = 36	, Similarity% 90 	 10653, 10861, 10906, 11476, 12500, 12669, 13011, 13651, 14100, 14270, 14390, 15264, 15647, 15963, 16538, 16673, 16743, 17046, 17471, 17479, 17525, 17757, 17888, 18025, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19318, 19351, 19435, 19491, 19718, 19751, 19785, 19863,
Current Input: 6			
Predicted Input: 4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5    Similarity %: 100	
Predicted Input: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3    Similarity %: 90
Predicted Input: 4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3    Similarity %: 90		
```

However, as learning continues, Similarity % also increases and eventually it attains the required accuracy. It is also interesting to note that during learning process one pattern leads to another pattern and so on.

```
-------------- Cycle 173 ---------------						
-------------- 6 ---------------			
			
Active segments: 40, Matching segments: 40			
Col  SDR: 503, 543, 546, 570, 575, 615, 618, 638, 674, 681, 698, 717, 725, 730, 745, 746, 750, 772, 774, 777, 778, 788, 799, 801, 834, 846, 853, 856, 860, 863, 871, 874, 875, 882, 883, 891, 893, 894, 900, 915, 			
Cell SDR: 12583, 13598, 13656, 14256, 14384, 15389, 15469, 15966, 16858, 17036, 17461, 17931, 18144, 18262, 18625, 18651, 18768, 19323, 19367, 19434, 19461, 19712, 19982, 20029, 20853, 21174, 21345, 21413, 21521, 21585, 21795, 21867, 21879, 22061, 22093, 22299, 22342, 22357, 22510, 22876, 			
Match. Actual value: 5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6 - Predicted value: 5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6			
Item length: 40	 Items: 34		
Predictive cells: 40 	 10653, 10861, 10906, 11480, 12500, 12669, 13011, 13651, 14100, 14270, 14390, 15267, 15647, 15963, 16538, 16673, 16743, 17046, 17471, 17479, 17525, 17757, 17890, 18025, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19319, 19351, 19435, 19491, 19718, 19751, 19785, 19863, 		
>indx:0	inp/len: 4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5/40 ,Same Bits = 40	, Similarity% 100 	 10653, 10861, 10906, 11480, 12500, 12669, 13011, 13651, 14100, 14270, 14390, 15267, 15647, 15963, 16538, 16673, 16743, 17046, 17471, 17479, 17525, 17757, 17890, 18025, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19319, 19351, 19435, 19491, 19718, 19751, 19785, 19863, 
>indx:1	inp/len: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3/40 ,Same Bits = 40	, Similarity% 100 	 10653, 10861, 10906, 11480, 12500, 12669, 13011, 13651, 14100, 14270, 14390, 15267, 15647, 15963, 16538, 16673, 16743, 17046, 17471, 17479, 17525, 17757, 17890, 18025, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19319, 19351, 19435, 19491, 19718, 19751, 19785, 19863,
>indx:2	inp/len: 4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3/40 ,Same Bits = 40	, Similarity% 100 	 10653, 10861, 10906, 11480, 12500, 12669, 13011, 13651, 14100, 14270, 14390, 15267, 15647, 15963, 16538, 16673, 16743, 17046, 17471, 17479, 17525, 17757, 17890, 18025, 18184, 18479, 18543, 18580, 18679, 18873, 18944, 19218, 19319, 19351, 19435, 19491, 19718, 19751, 19785, 19863,
Current Input: 6			
Predicted Input: 4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4-0-1-0-2-3-4-5-6-5    Similarity %: 100	
Predicted Input: 4-3-4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3    Similarity %: 100
Predicted Input: 4-3-4-0-1-0-2-3-4-5-6-5-4-3-7-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3    Similarity %: 100		
```

### 2.	Observation of visualized cell SDRs:

Once the TM learns SDR patterns completely, then accuracy (30 times) is reached. A method is added in our code to generate ‘cell state trace’ so that statistical analysis can be done. This helps to compare column/cell activity.

```csharp
    foreach (var input in activeColumnsLst)
                {
                    using (StreamWriter colSw = new StreamWriter($"ColumState_MinPctOverlDuty-{cfg.MinPctOverlapDutyCycles}_MaxBoost-{cfg.MaxBoost}_input-{input.Key}.csv"))
                    {
                        Debug.WriteLine($"------------ {input.Key} ------------");

                        foreach (var actCols in input.Value)
                        {
                            Debug.WriteLine(Helpers.StringifyVector(actCols.ToArray()));
                            colSw.WriteLine(Helpers.StringifyVector(actCols.ToArray()));
                        }
                    }
                }
```

Consider the following cell state trace generated for input sequence shown above.
```
---- cell state trace ----	
0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4
7594, 8438, 8532, 8638, 9209, 9367, 9631, 9688, 10242, 10262, 10854, 10914, 11303, 11485, 12086, 12519, 12660, 14266, 14377, 14457, 14580, 15102, 15551, 15630, 15738, 15767, 16009, 16052, 16124, 16155, 16474, 16548, 16714, 16725, 16829, 16934, 17044, 17199, 17247, 17767, 			
7594, 8438, 8532, 8638, 9209, 9367, 9631, 9688, 10242, 10262, 10854, 10914, 11303, 11485, 12086, 12519, 12660, 14266, 14377, 14457, 14580, 15102, 15551, 15630, 15738, 15767, 16009, 16052, 16124, 16155, 16474, 16548, 16714, 16725, 16829, 16934, 17044, 17199, 17247, 17767, 			
7594, 8438, 8532, 8638, 9209, 9367, 9631, 9688, 10242, 10262, 10854, 10914, 11303, 11485, 12086, 12519, 12660, 14266, 14377, 14457, 14580, 15102, 15551, 15630, 15738, 15767, 16009, 16052, 16124, 16155, 16474, 16548, 16714, 16725, 16829, 16934, 17044, 17199, 17247, 17767, 			
7594, 8438, 8532, 8638, 9209, 9367, 9631, 9688, 10242, 10262, 10854, 10914, 11303, 11485, 12086, 12519, 12660, 14266, 14377, 14457, 14580, 15102, 15551, 15630, 15738, 15767, 16009, 16052, 16124, 16155, 16474, 16548, 16714, 16725, 16829, 16934, 17044, 17199, 17247, 17767, 			
7594, 8438, 8532, 8638, 9209, 9367, 9631, 9688, 10242, 10262, 10854, 10914, 11303, 11485, 12086, 12519, 12660, 14266, 14377, 14457, 14580, 15102, 15551, 15630, 15738, 15767, 16009, 16052, 16124, 16155, 16474, 16548, 16714, 16725, 16829, 16934, 17044, 17199, 17247, 17767, 			
```

Cell state trace SDR1/2/3/4/5 is always the last 5 cycle's Cell SDR of input sequence. For example, In this case they represent Cell SDR from cycle 169 to cycle 173.
Once the experiment is complete, we need to visualize this learnt SDRs to check for any instability. The below result shows visual representation of SDR1/2/3/4/5 for the above mentioned input sequence.

|```Input Sequence```||```0-1-0-2-3-4-5-6-5-4-3-2-1-9-12-11-12-13-14-11-12-14-5-7-6-9-3-4-3-4-3-4```||```1-2-3-1-4-3-2-5-3-7-2-9-3-11-2```|

    ![][img0.1]|![][img0.2]|

[img0.1]: ./Visualized%20SDR%20Comparison/SDR_Comparison_Sequence_1_Stable.JPG
[img0.2]: ./Visualized%20SDR%20Comparison/SDR_Comparison_Sequence_2_Stable.JPG


We observe that these visualized Cell SDRs are same for all the 5 cycles and there is no difference in the pattern observed. 

### 3.	How did we generate Visualized SDRs?

The Column/Cell activity of SDRs can be generated using the python script draw_figure.py which is located in the below mentioned link.

[See the link](https://github.com/PrasadSahana/neocortexapi/blob/master/Python/ColumnActivityDiagram/draw_figure.py)

Following linux command can be run on VS terminal to generate the visualized SDRs:
```
py draw_figure.py -fn "CDS_FILE_WITH_SDRs" -gn "OUTPUT_FILENAME" -mc NUM-OF-ROWS-TO-PROCESS -ht THRESHOLDLINE-OPTIONAL -yt "Y_AXIS_TITLE" -xt "X_AXIS_TITLE" -st "SDR1/SDR2/SDR3/SDR4/SDR5" -fign NOTUSED
```
