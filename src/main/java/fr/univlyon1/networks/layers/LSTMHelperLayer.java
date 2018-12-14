package fr.univlyon1.networks.layers;

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.AbstractLSTM;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.GravesBidirectionalLSTM;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.layers.recurrent.FwdPassReturn;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelper;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelpers;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.blas.Level1;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.TimesOneMinus;
import org.nd4j.linalg.api.ops.impl.transforms.arithmetic.OldMulOp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class LSTMHelperLayer{

    public static FwdPassReturn activateHelper(BaseLayer layer, NeuralNetConfiguration conf, IActivation gateActivationFn, INDArray input, INDArray recurrentWeights, INDArray originalInputWeights, INDArray biases, boolean training, INDArray originalPrevOutputActivations, INDArray originalPrevMemCellState, boolean forBackprop, boolean forwards, String inputWeightKey, INDArray maskArray, boolean hasPeepholeConnections, LSTMHelper helper, CacheMode cacheMode, LayerWorkspaceMgr workspaceMgr) {
        if (input != null && input.length() != 0L) {
            INDArray inputWeights = originalInputWeights;
            INDArray prevOutputActivations = originalPrevOutputActivations;
            boolean is2dInput = input.rank() < 3;
            int timeSeriesLength = (int)(is2dInput ? 1L : input.size(2));
            int hiddenLayerSize = (int)recurrentWeights.size(0);
            int miniBatchSize = (int)input.size(0);
            INDArray prevMemCellState;
            if (originalPrevMemCellState == null) {
                prevMemCellState = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize}, 'f');
            } else {
                prevMemCellState = originalPrevMemCellState.dup('f');
            }

            INDArray recurrentWeightsIFOG = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, 4 * hiddenLayerSize)}).dup('f');
            INDArray wFFTranspose = null;
            INDArray wOOTranspose = null;
            INDArray wGGTranspose = null;
            if (hasPeepholeConnections) {
                wFFTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(4 * hiddenLayerSize, 4 * hiddenLayerSize + 1)}).transpose();
                wOOTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(4 * hiddenLayerSize + 1, 4 * hiddenLayerSize + 2)}).transpose();
                wGGTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(4 * hiddenLayerSize + 2, 4 * hiddenLayerSize + 3)}).transpose();
                if (timeSeriesLength > 1 || forBackprop) {
                    wFFTranspose = Shape.toMmulCompatible(wFFTranspose);
                    wOOTranspose = Shape.toMmulCompatible(wOOTranspose);
                    wGGTranspose = Shape.toMmulCompatible(wGGTranspose);
                }
            }

            boolean sigmoidGates = gateActivationFn instanceof ActivationSigmoid;
            IActivation afn = layer.layerConf().getActivationFn();
            INDArray outputActivations = null;
            FwdPassReturn toReturn = new FwdPassReturn();
            if (forBackprop) {
                toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];
                toReturn.memCellState = new INDArray[timeSeriesLength];
                toReturn.memCellActivations = new INDArray[timeSeriesLength];
                toReturn.iz = new INDArray[timeSeriesLength];
                toReturn.ia = new INDArray[timeSeriesLength];
                toReturn.fa = new INDArray[timeSeriesLength];
                toReturn.oa = new INDArray[timeSeriesLength];
                toReturn.ga = new INDArray[timeSeriesLength];
                if (!sigmoidGates) {
                    toReturn.fz = new INDArray[timeSeriesLength];
                    toReturn.oz = new INDArray[timeSeriesLength];
                    toReturn.gz = new INDArray[timeSeriesLength];
                }

                if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE) && workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
                    MemoryWorkspace wsB = workspaceMgr.notifyScopeBorrowed(ArrayType.FF_CACHE);
                    Throwable var34 = null;

                    try {
                        outputActivations = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength}, 'f');
                        toReturn.fwdPassOutput = outputActivations;
                    } catch (Throwable var68) {
                        var34 = var68;
                        throw var68;
                    } finally {
                        if (wsB != null) {
                            if (var34 != null) {
                                try {
                                    wsB.close();
                                } catch (Throwable var66) {
                                    var34.addSuppressed(var66);
                                }
                            } else {
                                wsB.close();
                            }
                        }

                    }
                } else {
                    outputActivations = workspaceMgr.create(ArrayType.ACTIVATIONS, new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength}, 'f');
                    toReturn.fwdPassOutput = outputActivations;
                }
            } else {
                outputActivations = workspaceMgr.create(ArrayType.ACTIVATIONS, new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength}, 'f');
                toReturn.fwdPassOutput = outputActivations;
            }

            Level1 l1BLAS = Nd4j.getBlasWrapper().level1();
            if (input.size(1) != originalInputWeights.size(0)) {
                throw new DL4JInvalidInputException("Received input with size(1) = " + input.size(1) + " (input array shape = " + Arrays.toString(input.shape()) + "); input.size(1) must match layer nIn size (nIn = " + originalInputWeights.size(0) + ")");
            } else if (originalPrevOutputActivations != null && originalPrevOutputActivations.size(0) != input.size(0)) {
                throw new DL4JInvalidInputException("Previous activations (stored state) number of examples = " + originalPrevOutputActivations.size(0) + " but input array number of examples = " + input.size(0) + ". Possible cause: using rnnTimeStep() without calling rnnClearPreviousState() between different sequences?");
            } else {
                if (originalPrevOutputActivations == null) {
                    prevOutputActivations = Nd4j.zeros(new int[]{miniBatchSize, hiddenLayerSize});
                }

                if (helper != null) {
                    FwdPassReturn ret = helper.activate(layer, conf, gateActivationFn, input, recurrentWeights, originalInputWeights, biases, training, prevOutputActivations, prevMemCellState, forBackprop, forwards, inputWeightKey, maskArray, hasPeepholeConnections, workspaceMgr);
                    if (ret != null) {
                        return ret;
                    }
                }

                for(int iTimeIndex = 0; iTimeIndex < timeSeriesLength; ++iTimeIndex) {
                    MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.RNN_FF_LOOP_WORKING_MEM);
                    Throwable var36 = null;

                    try {
                        int time = iTimeIndex;
                        if (!forwards) {
                            time = timeSeriesLength - iTimeIndex - 1;
                        }

                        INDArray miniBatchData = is2dInput ? input : input.tensorAlongDimension(time, new int[]{1, 0});
                        miniBatchData = Shape.toMmulCompatible(miniBatchData);
                        cacheEnter(training, cacheMode, workspaceMgr);
                        INDArray ifogActivations = miniBatchData.mmul(inputWeights);
                        cacheExit(training, cacheMode, workspaceMgr);
                        Nd4j.gemm(prevOutputActivations, recurrentWeightsIFOG, ifogActivations, false, false, 1.0D, 1.0D);
                        ifogActivations.addiRowVector(biases);
                        INDArray inputActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize)});
                        if (forBackprop) {
                            if (shouldCache(training, cacheMode, workspaceMgr)) {
                                cacheEnter(training, cacheMode, workspaceMgr);
                                toReturn.iz[time] = inputActivations.dup('f');
                                cacheExit(training, cacheMode, workspaceMgr);
                            } else {
                                toReturn.iz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, inputActivations, 'f');
                            }
                        }

                        layer.layerConf().getActivationFn().getActivation(inputActivations, training);
                        if (forBackprop) {
                            if (shouldCache(training, cacheMode, workspaceMgr)) {
                                cacheEnter(training, cacheMode, workspaceMgr);
                                toReturn.ia[time] = inputActivations.dup('f');
                                cacheExit(training, cacheMode, workspaceMgr);
                            } else {
                                toReturn.ia[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, inputActivations);
                            }
                        }

                        INDArray forgetGateActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize)});
                        INDArray inputModGateActivations;
                        if (hasPeepholeConnections) {
                            inputModGateActivations = prevMemCellState.dup('f').muliRowVector(wFFTranspose);
                            l1BLAS.axpy(inputModGateActivations.length(), 1.0D, inputModGateActivations, forgetGateActivations);
                        }

                        if (forBackprop && !sigmoidGates) {
                            if (shouldCache(training, cacheMode, workspaceMgr)) {
                                cacheEnter(training, cacheMode, workspaceMgr);
                                toReturn.fz[time] = forgetGateActivations.dup('f');
                                cacheExit(training, cacheMode, workspaceMgr);
                            } else {
                                toReturn.fz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, forgetGateActivations, 'f');
                            }
                        }

                        gateActivationFn.getActivation(forgetGateActivations, training);
                        if (forBackprop) {
                            if (shouldCache(training, cacheMode, workspaceMgr)) {
                                cacheEnter(training, cacheMode, workspaceMgr);
                                toReturn.fa[time] = forgetGateActivations.dup('f');
                                cacheExit(training, cacheMode, workspaceMgr);
                            } else {
                                toReturn.fa[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, forgetGateActivations);
                            }
                        }

                        inputModGateActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(3 * hiddenLayerSize, 4 * hiddenLayerSize)});
                        INDArray currentMemoryCellState;
                        if (hasPeepholeConnections) {
                            currentMemoryCellState = prevMemCellState.dup('f').muliRowVector(wGGTranspose);
                            l1BLAS.axpy(currentMemoryCellState.length(), 1.0D, currentMemoryCellState, inputModGateActivations);
                        }

                        if (forBackprop && !sigmoidGates) {
                            cacheEnter(training, cacheMode, workspaceMgr);
                            toReturn.gz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, inputModGateActivations, 'f');
                            cacheExit(training, cacheMode, workspaceMgr);
                        }

                        gateActivationFn.getActivation(inputModGateActivations, training);
                        if (forBackprop) {
                            if (shouldCache(training, cacheMode, workspaceMgr)) {
                                cacheEnter(training, cacheMode, workspaceMgr);
                                toReturn.ga[time] = inputModGateActivations.dup('f');
                                cacheExit(training, cacheMode, workspaceMgr);
                            } else {
                                toReturn.ga[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, inputModGateActivations);
                            }
                        }

                        INDArray inputModMulInput;
                        if (forBackprop) {
                            cacheEnter(training, cacheMode, workspaceMgr);
                            currentMemoryCellState = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, prevMemCellState, 'f').muli(forgetGateActivations);
                            cacheExit(training, cacheMode, workspaceMgr);
                            inputModMulInput = inputModGateActivations.dup('f').muli(inputActivations);
                        } else {
                            currentMemoryCellState = workspaceMgr.leverageTo(ArrayType.FF_WORKING_MEM, forgetGateActivations.muli(prevMemCellState));
                            inputModMulInput = inputModGateActivations.muli(inputActivations);
                        }

                        l1BLAS.axpy(currentMemoryCellState.length(), 1.0D, inputModMulInput, currentMemoryCellState);
                        INDArray outputGateActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)});
                        INDArray currMemoryCellActivation;
                        if (hasPeepholeConnections) {
                            currMemoryCellActivation = currentMemoryCellState.dup('f').muliRowVector(wOOTranspose);
                            l1BLAS.axpy(currMemoryCellActivation.length(), 1.0D, currMemoryCellActivation, outputGateActivations);
                        }

                        if (forBackprop && !sigmoidGates) {
                            cacheEnter(training, cacheMode, workspaceMgr);
                            toReturn.oz[time] = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, outputGateActivations, 'f');
                            cacheExit(training, cacheMode, workspaceMgr);
                        }

                        gateActivationFn.getActivation(outputGateActivations, training);
                        if (forBackprop) {
                            if (shouldCache(training, cacheMode, workspaceMgr)) {
                                cacheEnter(training, cacheMode, workspaceMgr);
                                toReturn.oa[time] = outputGateActivations.dup('f');
                                cacheExit(training, cacheMode, workspaceMgr);
                            } else {
                                toReturn.oa[time] = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, outputGateActivations);
                            }
                        }

                        cacheEnter(training, cacheMode, workspaceMgr);
                        currMemoryCellActivation = workspaceMgr.dup(ArrayType.FF_WORKING_MEM, currentMemoryCellState, 'f');
                        currMemoryCellActivation = afn.getActivation(currMemoryCellActivation, training);
                        cacheExit(training, cacheMode, workspaceMgr);
                        INDArray currHiddenUnitActivations;
                        if (forBackprop) {
                            cacheEnter(training, cacheMode, workspaceMgr);
                            currHiddenUnitActivations = workspaceMgr.dup(ArrayType.BP_WORKING_MEM, currMemoryCellActivation, 'f').muli(outputGateActivations);
                            cacheExit(training, cacheMode, workspaceMgr);
                        } else {
                            currHiddenUnitActivations = currMemoryCellActivation.muli(outputGateActivations);
                        }

                        if (maskArray != null) {
                            INDArray timeStepMaskColumn = maskArray.getColumn((long)time);
                            currHiddenUnitActivations.muliColumnVector(timeStepMaskColumn);
                            currentMemoryCellState.muliColumnVector(timeStepMaskColumn);
                        }

                        currentMemoryCellState = workspaceMgr.leverageTo(ArrayType.FF_WORKING_MEM, currentMemoryCellState);
                        if (forBackprop) {
                            toReturn.fwdPassOutputAsArrays[time] = currHiddenUnitActivations;
                            toReturn.memCellState[time] = currentMemoryCellState;
                            toReturn.memCellActivations[time] = currMemoryCellActivation;
                            if (training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE) && workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE)) {
                                toReturn.memCellActivations[time] = workspaceMgr.leverageTo(ArrayType.FF_CACHE, toReturn.memCellActivations[time]);
                                toReturn.memCellState[time] = workspaceMgr.leverageTo(ArrayType.FF_CACHE, toReturn.memCellState[time]);
                            }

                            if (cacheMode != CacheMode.NONE) {
                                outputActivations.tensorAlongDimension(time, new int[]{1, 0}).assign(currHiddenUnitActivations);
                            }
                        } else {
                            outputActivations.tensorAlongDimension(time, new int[]{1, 0}).assign(currHiddenUnitActivations);
                        }

                        prevOutputActivations = currHiddenUnitActivations;
                        prevMemCellState = currentMemoryCellState;
                        toReturn.lastAct = currHiddenUnitActivations;
                        toReturn.lastMemCell = currentMemoryCellState;
                    } catch (Throwable var70) {
                        var36 = var70;
                        throw var70;
                    } finally {
                        if (ws != null) {
                            if (var36 != null) {
                                try {
                                    ws.close();
                                } catch (Throwable var67) {
                                    var36.addSuppressed(var67);
                                }
                            } else {
                                ws.close();
                            }
                        }

                    }
                }

                toReturn.prevAct = originalPrevOutputActivations;
                toReturn.prevMemCell = originalPrevMemCellState;
                return toReturn;
            }
        } else {
            throw new IllegalArgumentException("Invalid input: not set or 0 length");
        }
    }

    private static boolean shouldCache(boolean training, CacheMode cacheMode, LayerWorkspaceMgr workspaceMgr) {
        return training && cacheMode != CacheMode.NONE && workspaceMgr.hasConfiguration(ArrayType.FF_CACHE) && workspaceMgr.isWorkspaceOpen(ArrayType.FF_CACHE);
    }

    private static void cacheEnter(boolean training, CacheMode cacheMode, LayerWorkspaceMgr workspaceMgr) {
        if (shouldCache(training, cacheMode, workspaceMgr)) {
            workspaceMgr.notifyScopeBorrowed(ArrayType.FF_CACHE);
        }

    }

    private static void cacheExit(boolean training, CacheMode cacheMode, LayerWorkspaceMgr workspaceMgr) {
        if (shouldCache(training, cacheMode, workspaceMgr)) {
            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceMgr.getWorkspaceName(ArrayType.FF_CACHE)).notifyScopeLeft();
        }

    }

    public static Pair<Gradient, INDArray> backpropGradientHelper(NeuralNetConfiguration conf, IActivation gateActivationFn, INDArray input, INDArray recurrentWeights, INDArray inputWeights, INDArray epsilon, boolean truncatedBPTT, int tbpttBackwardLength, FwdPassReturn fwdPass, boolean forwards, String inputWeightKey, String recurrentWeightKey, String biasWeightKey, Map<String, INDArray> gradientViews, INDArray maskArray, boolean hasPeepholeConnections, LSTMHelper helper, LayerWorkspaceMgr workspaceMgr) {
        long hiddenLayerSize = recurrentWeights.size(0);
        long prevLayerSize = inputWeights.size(0);
        long miniBatchSize = epsilon.size(0);
        boolean is2dInput = epsilon.rank() < 3;
        long timeSeriesLength = is2dInput ? 1L : epsilon.size(2);
        INDArray wFFTranspose = null;
        INDArray wOOTranspose = null;
        INDArray wGGTranspose = null;
        if (hasPeepholeConnections) {
            wFFTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4L * hiddenLayerSize)}).transpose();
            wOOTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4L * hiddenLayerSize + 1L)}).transpose();
            wGGTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4L * hiddenLayerSize + 2L)}).transpose();
        }

        INDArray wIFOG = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0L, 4L * hiddenLayerSize)});
        INDArray epsilonNext = workspaceMgr.create(ArrayType.ACTIVATION_GRAD, new long[]{miniBatchSize, prevLayerSize, timeSeriesLength}, 'f');
        INDArray nablaCellStateNext = null;
        INDArray deltaifogNext = Nd4j.create(new long[]{miniBatchSize, 4L * hiddenLayerSize}, 'f');
        INDArray deltaiNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0L, hiddenLayerSize)});
        INDArray deltafNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize, 2L * hiddenLayerSize)});
        INDArray deltaoNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2L * hiddenLayerSize, 3L * hiddenLayerSize)});
        INDArray deltagNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(3L * hiddenLayerSize, 4L * hiddenLayerSize)});
        Level1 l1BLAS = Nd4j.getBlasWrapper().level1();
        long endIdx = 0L;
        if (truncatedBPTT) {
            endIdx = Math.max(0L, timeSeriesLength - (long)tbpttBackwardLength);
        }

        INDArray iwGradientsOut = (INDArray)gradientViews.get(inputWeightKey);
        INDArray rwGradientsOut = (INDArray)gradientViews.get(recurrentWeightKey);
        INDArray bGradientsOut = (INDArray)gradientViews.get(biasWeightKey);
        iwGradientsOut.assign(0);
        rwGradientsOut.assign(0);
        bGradientsOut.assign(0);
        INDArray rwGradientsIFOG = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0L, 4L * hiddenLayerSize)});
        INDArray rwGradientsFF = null;
        INDArray rwGradientsOO = null;
        INDArray rwGradientsGG = null;
        if (hasPeepholeConnections) {
            rwGradientsFF = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4L * hiddenLayerSize)});
            rwGradientsOO = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4L * hiddenLayerSize + 1L)});
            rwGradientsGG = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4L * hiddenLayerSize + 2L)});
        }

        if (helper != null) {
            Pair<Gradient, INDArray> ret = helper.backpropGradient(conf, gateActivationFn, input, recurrentWeights, inputWeights, epsilon, truncatedBPTT, tbpttBackwardLength, fwdPass, forwards, inputWeightKey, recurrentWeightKey, biasWeightKey, gradientViews, maskArray, hasPeepholeConnections, workspaceMgr);
            if (ret != null) {
                return ret;
            }
        }

        boolean sigmoidGates = gateActivationFn instanceof ActivationSigmoid;
        IActivation afn = ((org.deeplearning4j.nn.conf.layers.BaseLayer)conf.getLayer()).getActivationFn();
        INDArray timeStepMaskColumn = null;

        for(long iTimeIndex = timeSeriesLength - 1L; iTimeIndex >= endIdx; --iTimeIndex) {
            MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.RNN_BP_LOOP_WORKING_MEM);
            Throwable var54 = null;

            try {
                int time = (int)iTimeIndex;
                int inext = 1;
                if (!forwards) {
                    time = (int)(timeSeriesLength - iTimeIndex - 1L);
                    inext = -1;
                }

                INDArray nablaCellState;
                if (iTimeIndex != timeSeriesLength - 1L && hasPeepholeConnections) {
                    nablaCellState = deltafNext.dup('f').muliRowVector(wFFTranspose);
                    l1BLAS.axpy(nablaCellState.length(), 1.0D, deltagNext.dup('f').muliRowVector(wGGTranspose), nablaCellState);
                } else {
                    nablaCellState = Nd4j.create(new long[]{miniBatchSize, hiddenLayerSize}, 'f');
                }

                INDArray prevMemCellState = iTimeIndex == 0L ? fwdPass.prevMemCell : fwdPass.memCellState[time - inext];
                INDArray prevHiddenUnitActivation = iTimeIndex == 0L ? fwdPass.prevAct : fwdPass.fwdPassOutputAsArrays[time - inext];
                INDArray currMemCellState = fwdPass.memCellState[time];
                INDArray epsilonSlice = is2dInput ? epsilon : epsilon.tensorAlongDimension(time, new int[]{1, 0});
                INDArray nablaOut = Shape.toOffsetZeroCopy(epsilonSlice, 'f');
                if (iTimeIndex != timeSeriesLength - 1L) {
                    Nd4j.gemm(deltaifogNext, wIFOG, nablaOut, false, true, 1.0D, 1.0D);
                }

                INDArray sigmahOfS = fwdPass.memCellActivations[time];
                INDArray ao = fwdPass.oa[time];
                Nd4j.getExecutioner().exec(new OldMulOp(nablaOut, sigmahOfS, deltaoNext));
                INDArray temp;
                if (sigmoidGates) {
                    temp = Nd4j.getExecutioner().execAndReturn(new TimesOneMinus(ao.dup('f')));
                    deltaoNext.muli(temp);
                } else {
                    deltaoNext.assign((INDArray)gateActivationFn.backprop(fwdPass.oz[time], deltaoNext).getFirst());
                }

                temp = (INDArray)afn.backprop(currMemCellState.dup('f'), ao.muli(nablaOut)).getFirst();
                l1BLAS.axpy(nablaCellState.length(), 1.0D, temp, nablaCellState);
                INDArray af;
                if (hasPeepholeConnections) {
                    af = deltaoNext.dup('f').muliRowVector(wOOTranspose);
                    l1BLAS.axpy(nablaCellState.length(), 1.0D, af, nablaCellState);
                }

                if (iTimeIndex != timeSeriesLength - 1L) {
                    af = fwdPass.fa[time + inext];
                    long length = nablaCellState.length();
                    l1BLAS.axpy(length, 1.0D, af.muli(nablaCellStateNext), nablaCellState);
                }

                nablaCellStateNext = workspaceMgr.leverageTo(ArrayType.BP_WORKING_MEM, nablaCellState);
                af = fwdPass.fa[time];
                INDArray deltaf = null;
                INDArray ag;
                if (iTimeIndex > 0L || prevMemCellState != null) {
                    deltaf = deltafNext;
                    if (sigmoidGates) {
                        Nd4j.getExecutioner().exec(new TimesOneMinus(af, deltafNext));
                        deltafNext.muli(nablaCellState);
                        deltafNext.muli(prevMemCellState);
                    } else {
                        ag = nablaCellState.mul(prevMemCellState);
                        deltafNext.assign((INDArray)gateActivationFn.backprop(fwdPass.fz[time].dup('f'), ag).getFirst());
                    }
                }

                ag = fwdPass.ga[time];
                INDArray ai = fwdPass.ia[time];
                INDArray zi;
                if (sigmoidGates) {
                    Nd4j.getExecutioner().exec(new TimesOneMinus(ag, deltagNext));
                    deltagNext.muli(ai);
                    deltagNext.muli(nablaCellState);
                } else {
                    zi = Nd4j.getExecutioner().execAndReturn(new OldMulOp(ai, nablaCellState, Nd4j.createUninitialized(ai.shape(), 'f')));
                    deltagNext.assign((INDArray)gateActivationFn.backprop(fwdPass.gz[time], zi).getFirst());
                }

                zi = fwdPass.iz[time];
                temp = Nd4j.getExecutioner().execAndReturn(new OldMulOp(ag, nablaCellState, Nd4j.createUninitialized(deltaiNext.shape(), 'f')));
                deltaiNext.assign((INDArray)afn.backprop(zi, temp).getFirst());
                if (maskArray != null) {
                    timeStepMaskColumn = maskArray.getColumn((long)time);
                    deltaifogNext.muliColumnVector(timeStepMaskColumn);
                }

                INDArray prevLayerActivationSlice = Shape.toMmulCompatible(is2dInput ? input : input.tensorAlongDimension(time, new int[]{1, 0}));
                INDArray epsilonNextSlice;
                INDArray wi;
                INDArray deltaog;
                if (iTimeIndex <= 0L && prevHiddenUnitActivation == null) {
                    epsilonNextSlice = iwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0L, hiddenLayerSize)});
                    Nd4j.gemm(prevLayerActivationSlice, deltaiNext, epsilonNextSlice, true, false, 1.0D, 1.0D);
                    wi = iwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2L * hiddenLayerSize, 4L * hiddenLayerSize)});
                    deltaog = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2L * hiddenLayerSize, 4L * hiddenLayerSize)});
                    Nd4j.gemm(prevLayerActivationSlice, deltaog, wi, true, false, 1.0D, 1.0D);
                } else {
                    Nd4j.gemm(prevLayerActivationSlice, deltaifogNext, iwGradientsOut, true, false, 1.0D, 1.0D);
                }

                if (iTimeIndex > 0L || prevHiddenUnitActivation != null) {
                    Nd4j.gemm(prevHiddenUnitActivation, deltaifogNext, rwGradientsIFOG, true, false, 1.0D, 1.0D);
                    if (hasPeepholeConnections) {
                        epsilonNextSlice = deltaf.dup('f').muli(prevMemCellState).sum(new int[]{0});
                        l1BLAS.axpy(hiddenLayerSize, 1.0D, epsilonNextSlice, rwGradientsFF);
                        wi = deltagNext.dup('f').muli(prevMemCellState).sum(new int[]{0});
                        l1BLAS.axpy(hiddenLayerSize, 1.0D, wi, rwGradientsGG);
                    }
                }

                if (hasPeepholeConnections) {
                    epsilonNextSlice = deltaoNext.dup('f').muli(currMemCellState).sum(new int[]{0});
                    l1BLAS.axpy(hiddenLayerSize, 1.0D, epsilonNextSlice, rwGradientsOO);
                }

                if (iTimeIndex <= 0L && prevHiddenUnitActivation == null) {
                    l1BLAS.axpy(hiddenLayerSize, 1.0D, deltaiNext.sum(new int[]{0}), bGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(0L, hiddenLayerSize)}));
                    epsilonNextSlice = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2L * hiddenLayerSize, 4L * hiddenLayerSize)}).sum(new int[]{0});
                    wi = bGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.point(0L), NDArrayIndex.interval(2L * hiddenLayerSize, 4L * hiddenLayerSize)});
                    l1BLAS.axpy(2L * hiddenLayerSize, 1.0D, epsilonNextSlice, wi);
                } else {
                    l1BLAS.axpy(4L * hiddenLayerSize, 1.0D, deltaifogNext.sum(new int[]{0}), bGradientsOut);
                }

                epsilonNextSlice = epsilonNext.tensorAlongDimension(time, new int[]{1, 0});
                if (iTimeIndex <= 0L && prevHiddenUnitActivation == null) {
                    wi = inputWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0L, hiddenLayerSize)});
                    Nd4j.gemm(deltaiNext, wi, epsilonNextSlice, false, true, 1.0D, 1.0D);
                    deltaog = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2L * hiddenLayerSize, 4L * hiddenLayerSize)});
                    INDArray wog = inputWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2L * hiddenLayerSize, 4L * hiddenLayerSize)});
                    Nd4j.gemm(deltaog, wog, epsilonNextSlice, false, true, 1.0D, 1.0D);
                } else {
                    Nd4j.gemm(deltaifogNext, inputWeights, epsilonNextSlice, false, true, 1.0D, 1.0D);
                }

                if (maskArray != null) {
                    epsilonNextSlice.muliColumnVector(timeStepMaskColumn);
                }
            } catch (Throwable var86) {
                var54 = var86;
                throw var86;
            } finally {
                if (ws != null) {
                    if (var54 != null) {
                        try {
                            ws.close();
                        } catch (Throwable var85) {
                            var54.addSuppressed(var85);
                        }
                    } else {
                        ws.close();
                    }
                }

            }
        }

        Gradient retGradient = new DefaultGradient();
        retGradient.gradientForVariable().put(inputWeightKey, iwGradientsOut);
        retGradient.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
        retGradient.gradientForVariable().put(biasWeightKey, bGradientsOut);
        return new Pair(retGradient, epsilonNext);
    }

    public static LayerMemoryReport getMemoryReport(AbstractLSTM lstmLayer, InputType inputType) {
        boolean isGraves = lstmLayer instanceof GravesLSTM;
        return getMemoryReport(isGraves, lstmLayer, inputType);
    }

    public static LayerMemoryReport getMemoryReport(GravesBidirectionalLSTM lstmLayer, InputType inputType) {
        LayerMemoryReport r = getMemoryReport(true, lstmLayer, inputType);
        Map<CacheMode, Long> fixedTrain = new HashMap();
        Map<CacheMode, Long> varTrain = new HashMap();
        Map<CacheMode, Long> cacheFixed = new HashMap();
        Map<CacheMode, Long> cacheVar = new HashMap();
        CacheMode[] var7 = CacheMode.values();
        int var8 = var7.length;

        for(int var9 = 0; var9 < var8; ++var9) {
            CacheMode cm = var7[var9];
            fixedTrain.put(cm, 2L * (Long)r.getWorkingMemoryFixedTrain().get(cm));
            varTrain.put(cm, 2L * (Long)r.getWorkingMemoryVariableTrain().get(cm));
            cacheFixed.put(cm, 2L * (Long)r.getCacheModeMemFixed().get(cm));
            cacheVar.put(cm, 2L * (Long)r.getCacheModeMemVariablePerEx().get(cm));
        }

        return (new LayerMemoryReport.Builder(r.getLayerName(), r.getClass(), r.getInputType(), r.getOutputType())).standardMemory(2L * r.getParameterSize(), 2L * r.getUpdaterStateSize()).workingMemory(2L * r.getWorkingMemoryFixedInference(), 2L * r.getWorkingMemoryVariableInference(), fixedTrain, varTrain).cacheMemory(cacheFixed, cacheVar).build();
    }

    public static LayerMemoryReport getMemoryReport(boolean isGraves, FeedForwardLayer lstmLayer, InputType inputType) {
        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent)inputType;
        long tsLength = itr.getTimeSeriesLength();
        InputType outputType = lstmLayer.getOutputType(-1, inputType);
        long numParams = lstmLayer.initializer().numParams(lstmLayer);
        int updaterSize = (int)lstmLayer.getIUpdater().stateSize(numParams);
        long workingMemInferencePerEx = tsLength * 4L * lstmLayer.getNOut();
        long fwdPassPerTimeStepTrainCache = tsLength * 6L * lstmLayer.getNOut();
        long backpropWorkingSpace = (long)(isGraves ? 9 : 6) * tsLength * lstmLayer.getNOut();
        Map<CacheMode, Long> trainVariable = new HashMap();
        Map<CacheMode, Long> cacheVariable = new HashMap();
        CacheMode[] var18 = CacheMode.values();
        int var19 = var18.length;

        for(int var20 = 0; var20 < var19; ++var20) {
            CacheMode cm = var18[var20];
            long trainWorking;
            long cacheMem;
            if (cm == CacheMode.NONE) {
                trainWorking = workingMemInferencePerEx + fwdPassPerTimeStepTrainCache + backpropWorkingSpace;
                cacheMem = 0L;
            } else {
                trainWorking = workingMemInferencePerEx + backpropWorkingSpace;
                cacheMem = fwdPassPerTimeStepTrainCache;
            }

            trainVariable.put(cm, trainWorking);
            cacheVariable.put(cm, cacheMem);
        }

        return (new LayerMemoryReport.Builder((String)null, lstmLayer.getClass(), inputType, outputType)).standardMemory(numParams, (long)updaterSize).workingMemory(0L, workingMemInferencePerEx, MemoryReport.CACHE_MODE_ALL_ZEROS, trainVariable).cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, cacheVariable).build();
    }
}
