﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using NeoCortexApi;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;


namespace NeoCortexApiSample
{
    /// <summary>
    /// Implements an experiment that demonstrates how to learn sequences.
    /// </summary>
    public class SequenceLearning
    {
        public void Run()
        {
            Console.WriteLine($"Hello NeocortexApi! Experiment {nameof(SequenceLearning)}");

            int inputBits = 100;
            // All the Experiments are performed with 2048 Columns
            int numColumns = 2048;

            //Spatial Pooler Parameter Configuration using HtmConfig
            HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
            {
                Random = new ThreadSafeRandom(42),

                CellsPerColumn = 25,
                GlobalInhibition = true,
                LocalAreaDensity = -1,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * inputBits),
                InhibitionRadius = 15,

                MaxBoost = 10.0,
                DutyCyclePeriod = 25,
                MinPctOverlapDutyCycles = 0.75,
                MaxSynapsesPerSegment = (int)(0.02 * numColumns),

                ActivationThreshold = 15,
                ConnectedPermanence = 0.5,

                // Learning is slower than forgetting in this case.
                PermanenceDecrement = 0.25,
                PermanenceIncrement = 0.15,

                // Used by punishing of segments.
                PredictedSegmentDecrement = 0.1
            };

            double max = 20;

            Dictionary<string, object> settings = new Dictionary<string, object>()
            {
                { "W", 15},
                { "N", inputBits},
                { "Radius", -1.0},
                { "MinVal", 0.0},
                { "Periodic", false},
                { "Name", "scalar"},
                { "ClipInput", false},
                { "MaxVal", max}
            };

            EncoderBase encoder = new ScalarEncoder(settings);

            //  Attains 100% Accuracy (30 times) at Cycle 262, without any instability. Which means that the learnt SDRs are never forgotten and cell SDRs remain same throughout.
            //  List<double> inputValues = new List<double>(new double[] { 2.0, 3.0, 2.0, 5.0, 2.0, 8.0, 2.0, 7.0, 2.0, 5.0, 2.0, 13.0, 2.0, 12.0, 2.0, 9.0, 2.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 271, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 9.0, 12.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 14.0, 5.0, 7.0, 6.0, 9.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 396, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 0.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 179, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 2.0, 3.0, 2.0, 4.0, 3.0, 4.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0, 3.0, 6.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 231, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 2.0, 3.0, 3.0, 4.0, 1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 3.0, 6.0, 7.0, 4.0, 3.0, 7.0, 1.0, 9.0, 3.0, 11.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 205, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 13.0, 12.0, 11.0, 12.0, 13.0, 14.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 231, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 2.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0, 3.0, 8.0, 2.0, 8.0, 9.0, 10.0, 2.0, 11 });

            //  Attains 100% Accuracy (30 times) at Cycle 171, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 2.0, 11.0, 9.0, 2.0, 8.0, 6.0, 2.0, 3.0, 2.0, 9.0, 3.0, 2.0, 1.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 196, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 7.0, 2.0, 8.0, 2.0, 3.0, 6.0, 2.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 204, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 3.0, 4.0, 2.0, 5.0, 6.0, 2.0, 7.0, 8.0, 2.0, 9.0, 10.0, 11.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 205, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 5.0, 6.0, 1.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 213, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 0.0, 2.0, 1.0, 2.0, 3.0, 2.0, 4.0, 2.0, 5.0, 2.0, 6.0, 2.0, 7.0, 2.0, 8.0, 2.0, 9.0, 2.0, 10.0 });

            //  Attains 100% Accuracy (30 times) at Cycle 296, without any instability.
            //  List<double> inputValues = new List<double>(new double[] { 1.0, 2.0, 3.0, 1.0, 4.0, 3.0, 2.0, 5.0, 3.0, 7.0, 2.0, 9.0, 3.0, 11.0, 2.0 });


            // Stable with PermanenceDecrement 0.25/PermanenceIncrement 0.15 and ActivationThreshold 25.
            List<double> inputValues = new List<double>(new double[] { 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 4.0, 3.0, 7.0, 1.0, 9.0, 12.0, 11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 14.0, 5.0, 7.0, 6.0, 9.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0 });

            RunExperiment(inputBits, cfg, encoder, inputValues);
        }

        /// <summary>
        ///
        /// </summary>
        private void RunExperiment(int inputBits, HtmConfig cfg, EncoderBase encoder, List<double> inputValues)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            int maxMatchCnt = 0;
            // learn is always set to true during stable state
            bool learn = true;

            var mem = new Connections(cfg);
            bool isInStableState = false;

            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();

            var numInputs = inputValues.Distinct<double>().ToList().Count;

            TemporalMemory tm = new TemporalMemory();

            HomeostaticPlasticityController hpa = new HomeostaticPlasticityController(mem, numInputs * 155, (isStable, numPatterns, actColAvg, seenInputs) =>
            {
                if (isStable)
                    // Event should be fired when entering the stable state.
                    Debug.WriteLine($"STABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");
                else
                    // Ideal Spatial Pooler should never enter unstable state after stable state.
                    Debug.WriteLine($"INSTABLE: Patterns: {numPatterns}, Inputs: {seenInputs}, iteration: {seenInputs / numPatterns}");

                // learning should be set to false during instable state
                learn = isInStableState = isStable;

                //if (isStable && layer1.HtmModules.ContainsKey("tm") == false)
                //    layer1.HtmModules.Add("tm", tm);

                // Clear all learned patterns in the classifier.
                cls.ClearState();

                // Clear active and predictive cells.
                tm.Reset(mem);

            }, numOfCyclesToWaitOnChange: 25);

            SpatialPoolerMT sp = new SpatialPoolerMT(hpa);
            sp.Init(mem);
            tm.Init(mem);

            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");

            layer1.HtmModules.Add("encoder", encoder);
            layer1.HtmModules.Add("sp", sp);

            double[] inputs = inputValues.ToArray();
            int[] prevActiveCols = new int[0];

            int cycle = 0;
            int matches = 0;

            string lastPredictedValue = "0";

            Dictionary<double, List<List<int>>> activeColumnsLst = new Dictionary<double, List<List<int>>>();

            foreach (var input in inputs)
            {
                if (activeColumnsLst.ContainsKey(input) == false)
                    activeColumnsLst.Add(input, new List<List<int>>());
            }

            int maxCycles = 3500;
            int maxPrevInputs = inputValues.Count - 1;
            List<string> previousInputs = new List<string>();
            previousInputs.Add("-1.0");

            // Training Spatial Pooler to get stable during New-born stage.

            for (int i = 0; i < maxCycles; i++)
            {
                matches = 0;

                cycle++;

                Debug.WriteLine($"-------------- Newborn Cycle {cycle} ---------------");

                foreach (var input in inputs)
                {
                    Debug.WriteLine($" -- {input} --");

                    var lyrOut = layer1.Compute(input, learn);

                    if (isInStableState)
                        break;
                }
                if (isInStableState)
                    break;
            }

            layer1.HtmModules.Add("tm", tm);

            //
            // Now training with SP+TM. SP is already pretrained on the given input pattern set.
            for (int i = 0; i < maxCycles; i++)
            {
                matches = 0;

                cycle++;

                Debug.WriteLine($"-------------- Cycle {cycle} ---------------");

                foreach (var input in inputs)
                {
                    Debug.WriteLine($"-------------- {input} ---------------");

                    var lyrOut = layer1.Compute(input, learn) as ComputeCycle;

                    // lyrOut is null when the TM is added to the layer inside of HPC callback by entering of the stable state.
                    //if (isInStableState && lyrOut != null)
                    {
                        var activeColumns = layer1.GetResult("sp") as int[];

                        //layer2.Compute(lyrOut.WinnerCells, true);
                        //activeColumnsLst[input].Add(activeColumns.ToList());

                        previousInputs.Add(input.ToString());
                        if (previousInputs.Count > (maxPrevInputs + 1))
                            previousInputs.RemoveAt(0);

                        // In the pretrained SP with HPC, the TM will quickly learn cells for patterns
                        // In that case the starting sequence 4-5-6 might have the same SDR as 1-2-3-4-5-6,
                        // Which will result in returning of 4-5-6 instead of 1-2-3-4-5-6.
                        // HtmClassifier allways return the first matching sequence. Because 4-5-6 will be as first
                        // memorized, it will match as the first one.
                        if (previousInputs.Count < maxPrevInputs)
                            continue;

                        string key = GetKey(previousInputs, input);

                        List<Cell> actCells;

                        if (lyrOut.ActiveCells.Count == lyrOut.WinnerCells.Count)
                        {
                            actCells = lyrOut.ActiveCells;
                        }
                        else
                        {
                            actCells = lyrOut.WinnerCells;
                        }

                        cls.Learn(key, actCells.ToArray());

                        if (learn == false)
                            Debug.WriteLine($"Inference mode");

                        Debug.WriteLine($"Col  SDR: {Helpers.StringifyVector(lyrOut.ActivColumnIndicies)}");
                        Debug.WriteLine($"Cell SDR: {Helpers.StringifyVector(actCells.Select(c => c.Index).ToArray())}");

                        if (key == lastPredictedValue)
                        {
                            matches++;
                            Debug.WriteLine($"Match. Actual value: {key} - Predicted value: {lastPredictedValue}");
                        }
                        else
                            Debug.WriteLine($"Missmatch! Actual value: {key} - Predicted value: {lastPredictedValue}");

                        if (lyrOut.PredictiveCells.Count > 0)
                        {
                            // The below line of code picks top 3 Predictions from Index list based on Similarity Percentage
                            var predictedInputValue = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);
                            Debug.WriteLine($"Current Input: {input}");

                            foreach (var item in predictedInputValue)
                            {
                                // We are filtering the values to pick top 3 predictions with Similarity >= 50%
                                if (item.Similarity >= (double)50.00 && item.PredictedInput.Contains("-1.0") == false)
                                {
                                    Debug.WriteLine($"Predicted Input: {item.PredictedInput}");
                                }
                            }

                            lastPredictedValue = predictedInputValue.First().PredictedInput;
                        }
                        else
                        {
                            Debug.WriteLine($"NO CELLS PREDICTED for next cycle.");
                            lastPredictedValue = String.Empty;
                        }
                    }
                }

                // The brain does not do that this way, so we don't use it.
                //tm1.Reset(mem);

                double accuracy = (double)matches / (double)inputs.Length * 100.0;

                Debug.WriteLine($"Cycle: {cycle}\tMatches={matches} of {inputs.Length}\t {accuracy}%");

                if (accuracy == 100.0)
                {
                    maxMatchCnt++;
                    Debug.WriteLine($"100% accuracy reached {maxMatchCnt} times.");
                    //
                    // Experiment is completed if we are 30 cycles long at the 100% accuracy.
                    if (maxMatchCnt >= 30)
                    {
                        sw.Stop();
                        Debug.WriteLine($"Exit experiment in the stable state after 30 repeats with 100% of accuracy. Elapsed time: {sw.ElapsedMilliseconds / 1000 / 60} min.");
                        learn = false;
                        break;
                    }
                }
                else if (maxMatchCnt > 0)
                {
                    // If there is drop in accuracy then TM has forgotten learnt patterns and it starts learning new pattern of SDRs again
                    // Drop in accuracy should not happen usually.
                    Debug.WriteLine($"At 100% accuracy after {maxMatchCnt} repeats we get a drop of accuracy with {accuracy}. This indicates instable state. Learning will be continued.");
                    maxMatchCnt = 0;
                }
            }

            Debug.WriteLine("---- cell state trace ----");

            cls.TraceState($"cellState_MinPctOverlDuty-{cfg.MinPctOverlapDutyCycles}_MaxBoost-{cfg.MaxBoost}.csv");

            Debug.WriteLine("---- Spatial Pooler column state  ----");

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

            Debug.WriteLine("------------------------------- END -------------------------------");
        }


        private static string GetKey(List<string> prevInputs, double input)
        {
            string key = String.Empty;

            for (int i = 0; i < prevInputs.Count; i++)
            {
                if (i > 0)
                    key += "-";

                key += (prevInputs[i]);
            }

            return key;
        }

    }
}