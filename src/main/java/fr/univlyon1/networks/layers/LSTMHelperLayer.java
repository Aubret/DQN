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
    private static final Logger log = LoggerFactory.getLogger(LSTMHelpers.class);

    private LSTMHelperLayer() {
    }

    public static FwdPassReturn activateHelper(BaseLayer layer, NeuralNetConfiguration conf, IActivation gateActivationFn, INDArray input, INDArray recurrentWeights, INDArray originalInputWeights, INDArray biases, boolean training, INDArray originalPrevOutputActivations, INDArray originalPrevMemCellState, boolean forBackprop, boolean forwards, String inputWeightKey, INDArray maskArray, boolean hasPeepholeConnections, LSTMHelper helper, CacheMode cacheMode) {
        if(input != null && input.length() != 0) {
            INDArray inputWeights = originalInputWeights;
            INDArray prevOutputActivations = originalPrevOutputActivations;
            boolean is2dInput = input.rank() < 3;
            int timeSeriesLength = is2dInput?1:input.size(2);
            int hiddenLayerSize = recurrentWeights.size(0);
            int miniBatchSize = input.size(0);
            INDArray prevMemCellState;
            if(originalPrevMemCellState == null) {
                prevMemCellState = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize}, 'f');
            } else {
                prevMemCellState = originalPrevMemCellState.dup('f');
            }

            INDArray recurrentWeightsIFOG = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, 4 * hiddenLayerSize)}).dup('f');
            INDArray wFFTranspose = null;
            INDArray wOOTranspose = null;
            INDArray wGGTranspose = null;
            if(hasPeepholeConnections) {
                wFFTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(4 * hiddenLayerSize, 4 * hiddenLayerSize + 1)}).transpose();
                wOOTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(4 * hiddenLayerSize + 1, 4 * hiddenLayerSize + 2)}).transpose();
                wGGTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(4 * hiddenLayerSize + 2, 4 * hiddenLayerSize + 3)}).transpose();
                if(timeSeriesLength > 1 || forBackprop) {
                    wFFTranspose = Shape.toMmulCompatible(wFFTranspose);
                    wOOTranspose = Shape.toMmulCompatible(wOOTranspose);
                    wGGTranspose = Shape.toMmulCompatible(wGGTranspose);
                }
            }

            boolean sigmoidGates = gateActivationFn instanceof ActivationSigmoid;
            IActivation afn = layer.layerConf().getActivationFn();
            INDArray outputActivations = null;
            FwdPassReturn toReturn = new FwdPassReturn();
            if(forBackprop) {
                toReturn.fwdPassOutputAsArrays = new INDArray[timeSeriesLength];
                toReturn.memCellState = new INDArray[timeSeriesLength];
                toReturn.memCellActivations = new INDArray[timeSeriesLength];
                toReturn.iz = new INDArray[timeSeriesLength];
                toReturn.ia = new INDArray[timeSeriesLength];
                toReturn.fa = new INDArray[timeSeriesLength];
                toReturn.oa = new INDArray[timeSeriesLength];
                toReturn.ga = new INDArray[timeSeriesLength];
                if(!sigmoidGates) {
                    toReturn.fz = new INDArray[timeSeriesLength];
                    toReturn.oz = new INDArray[timeSeriesLength];
                    toReturn.gz = new INDArray[timeSeriesLength];
                }

                if(cacheMode != CacheMode.NONE) {
                    MemoryWorkspace l1BLAS = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                    Throwable iTimeIndex = null;

                    try {
                        outputActivations = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength}, 'f');
                        toReturn.fwdPassOutput = outputActivations;
                    } catch (Throwable var51) {
                        iTimeIndex = var51;
                        throw var51;
                    } finally {
                        if(l1BLAS != null) {
                            if(iTimeIndex != null) {
                                try {
                                    l1BLAS.close();
                                } catch (Throwable var50) {
                                    iTimeIndex.addSuppressed(var50);
                                }
                            } else {
                                l1BLAS.close();
                            }
                        }

                    }
                }
            }
            outputActivations = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize, timeSeriesLength}, 'f');
            toReturn.fwdPassOutput = outputActivations;

            Level1 var53 = Nd4j.getBlasWrapper().level1();
            if(input.size(1) != originalInputWeights.size(0)) {
                throw new DL4JInvalidInputException("Received input with size(1) = " + input.size(1) + " (input array shape = " + Arrays.toString(input.shape()) + "); input.size(1) must match layer nIn size (nIn = " + originalInputWeights.size(0) + ")");
            } else if(originalPrevOutputActivations != null && originalPrevOutputActivations.size(0) != input.size(0)) {
                throw new DL4JInvalidInputException("Previous activations (stored state) number of examples = " + originalPrevOutputActivations.size(0) + " but input array number of examples = " + input.size(0) + ". Possible cause: using rnnTimeStep() without calling rnnClearPreviousState() between different sequences?");
            } else {
                if(originalPrevOutputActivations == null) {
                    prevOutputActivations = Nd4j.zeros(new int[]{miniBatchSize, hiddenLayerSize});
                }

                if(helper != null) {
                    FwdPassReturn var54 = helper.activate(layer, conf, gateActivationFn, input, recurrentWeights, originalInputWeights, biases, training, prevOutputActivations, prevMemCellState, forBackprop, forwards, inputWeightKey, maskArray, hasPeepholeConnections);
                    if(var54 != null) {
                        return var54;
                    }
                }

                for(int var55 = 0; var55 < timeSeriesLength; ++var55) {
                    int time = var55;
                    if(!forwards) {
                        time = timeSeriesLength - var55 - 1;
                    }

                    INDArray miniBatchData = is2dInput?input:input.tensorAlongDimension(time, new int[]{1, 0});
                    miniBatchData = Shape.toMmulCompatible(miniBatchData);
                    if(cacheMode != CacheMode.NONE) {
                        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                    }

                    INDArray ifogActivations = miniBatchData.mmul(inputWeights);
                    if(cacheMode != CacheMode.NONE) {
                        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                    }

                    Nd4j.gemm(prevOutputActivations, recurrentWeightsIFOG, ifogActivations, false, false, 1.0D, 1.0D);
                    ifogActivations.addiRowVector(biases);
                    INDArray inputActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize)});
                    if(forBackprop) {
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                        }

                        toReturn.iz[time] = inputActivations.dup('f');
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                        }
                    }

                    layer.layerConf().getActivationFn().getActivation(inputActivations, training);
                    if(forBackprop) {
                        toReturn.ia[time] = inputActivations;
                    }

                    INDArray forgetGateActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize)});
                    INDArray inputModGateActivations;
                    if(hasPeepholeConnections) {
                        inputModGateActivations = prevMemCellState.dup('f').muliRowVector(wFFTranspose);
                        var53.axpy(inputModGateActivations.length(), 1.0D, inputModGateActivations, forgetGateActivations);
                    }

                    if(forBackprop && !sigmoidGates) {
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                        }

                        toReturn.fz[time] = forgetGateActivations.dup('f');
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                        }
                    }

                    gateActivationFn.getActivation(forgetGateActivations, training);
                    if(forBackprop) {
                        toReturn.fa[time] = forgetGateActivations;
                    }

                    inputModGateActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(3 * hiddenLayerSize, 4 * hiddenLayerSize)});
                    INDArray currentMemoryCellState;
                    if(hasPeepholeConnections) {
                        currentMemoryCellState = prevMemCellState.dup('f').muliRowVector(wGGTranspose);
                        var53.axpy(currentMemoryCellState.length(), 1.0D, currentMemoryCellState, inputModGateActivations);
                    }

                    if(forBackprop && !sigmoidGates) {
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                        }

                        toReturn.gz[time] = inputModGateActivations.dup('f');
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                        }
                    }

                    gateActivationFn.getActivation(inputModGateActivations, training);
                    if(forBackprop) {
                        toReturn.ga[time] = inputModGateActivations;
                    }

                    INDArray inputModMulInput;
                    if(forBackprop) {
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                        }

                        currentMemoryCellState = prevMemCellState.dup('f').muli(forgetGateActivations);
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                        }

                        inputModMulInput = inputModGateActivations.dup('f').muli(inputActivations);
                    } else {
                        currentMemoryCellState = forgetGateActivations.muli(prevMemCellState);
                        inputModMulInput = inputModGateActivations.muli(inputActivations);
                    }

                    var53.axpy(currentMemoryCellState.length(), 1.0D, inputModMulInput, currentMemoryCellState);
                    INDArray outputGateActivations = ifogActivations.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)});
                    INDArray currMemoryCellActivation;
                    if(hasPeepholeConnections) {
                        currMemoryCellActivation = currentMemoryCellState.dup('f').muliRowVector(wOOTranspose);
                        var53.axpy(currMemoryCellActivation.length(), 1.0D, currMemoryCellActivation, outputGateActivations);
                    }

                    if(forBackprop && !sigmoidGates) {
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                        }

                        toReturn.oz[time] = outputGateActivations.dup('f');
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                        }
                    }

                    gateActivationFn.getActivation(outputGateActivations, training);
                    if(forBackprop) {
                        toReturn.oa[time] = outputGateActivations;
                    }

                    if(cacheMode != CacheMode.NONE) {
                        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                    }

                    currMemoryCellActivation = afn.getActivation(currentMemoryCellState.dup('f'), training);
                    if(cacheMode != CacheMode.NONE) {
                        Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                    }

                    INDArray currHiddenUnitActivations;
                    if(forBackprop) {
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeBorrowed();
                        }

                        currHiddenUnitActivations = currMemoryCellActivation.dup('f').muli(outputGateActivations);
                        if(cacheMode != CacheMode.NONE) {
                            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread("LOOP_CACHE").notifyScopeLeft();
                        }
                    } else {
                        currHiddenUnitActivations = currMemoryCellActivation.muli(outputGateActivations);
                    }

                    if(maskArray != null) {
                        INDArray timeStepMaskColumn = maskArray.getColumn(time);
                        currHiddenUnitActivations.muliColumnVector(timeStepMaskColumn);
                        currentMemoryCellState.muliColumnVector(timeStepMaskColumn);
                    }

                    if(forBackprop) {
                        toReturn.fwdPassOutputAsArrays[time] = currHiddenUnitActivations;
                        toReturn.memCellState[time] = currentMemoryCellState;
                        toReturn.memCellActivations[time] = currMemoryCellActivation;
                    }
                    outputActivations.tensorAlongDimension(time, new int[]{1, 0}).assign(currHiddenUnitActivations);

                    prevOutputActivations = currHiddenUnitActivations;
                    prevMemCellState = currentMemoryCellState;
                    toReturn.lastAct = currHiddenUnitActivations;
                    toReturn.lastMemCell = currentMemoryCellState;
                }

                toReturn.prevAct = originalPrevOutputActivations;
                toReturn.prevMemCell = originalPrevMemCellState;
                return toReturn;
            }
        } else {
            throw new IllegalArgumentException("Invalid input: not set or 0 length");
        }
    }

    public static Pair<Gradient, INDArray> backpropGradientHelper(NeuralNetConfiguration conf, IActivation gateActivationFn, INDArray input, INDArray recurrentWeights, INDArray inputWeights, INDArray epsilon, boolean truncatedBPTT, int tbpttBackwardLength, FwdPassReturn fwdPass, boolean forwards, String inputWeightKey, String recurrentWeightKey, String biasWeightKey, Map<String, INDArray> gradientViews, INDArray maskArray, boolean hasPeepholeConnections, LSTMHelper helper) {
        int hiddenLayerSize = recurrentWeights.size(0);
        int prevLayerSize = inputWeights.size(0);
        int miniBatchSize = epsilon.size(0);
        boolean is2dInput = epsilon.rank() < 3;
        int timeSeriesLength = is2dInput?1:epsilon.size(2);
        INDArray wFFTranspose = null;
        INDArray wOOTranspose = null;
        INDArray wGGTranspose = null;
        if(hasPeepholeConnections) {
            wFFTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize)}).transpose();
            wOOTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 1)}).transpose();
            wGGTranspose = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 2)}).transpose();
        }

        INDArray wIFOG = recurrentWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, 4 * hiddenLayerSize)});
        INDArray epsilonNext = Nd4j.create(new int[]{miniBatchSize, prevLayerSize, timeSeriesLength}, 'f');
        INDArray nablaCellStateNext = null;
        INDArray deltaifogNext = Nd4j.create(new int[]{miniBatchSize, 4 * hiddenLayerSize}, 'f');
        INDArray deltaiNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize)});
        INDArray deltafNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(hiddenLayerSize, 2 * hiddenLayerSize)});
        INDArray deltaoNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 3 * hiddenLayerSize)});
        INDArray deltagNext = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(3 * hiddenLayerSize, 4 * hiddenLayerSize)});
        Level1 l1BLAS = Nd4j.getBlasWrapper().level1();
        int endIdx = 0;
        if(truncatedBPTT) {
            endIdx = Math.max(0, timeSeriesLength - tbpttBackwardLength);
        }

        INDArray iwGradientsOut = (INDArray)gradientViews.get(inputWeightKey);
        INDArray rwGradientsOut = (INDArray)gradientViews.get(recurrentWeightKey);
        INDArray bGradientsOut = (INDArray)gradientViews.get(biasWeightKey);
        iwGradientsOut.assign(Integer.valueOf(0));
        rwGradientsOut.assign(Integer.valueOf(0));
        bGradientsOut.assign(Integer.valueOf(0));
        INDArray rwGradientsIFOG = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, 4 * hiddenLayerSize)});
        INDArray rwGradientsFF = null;
        INDArray rwGradientsOO = null;
        INDArray rwGradientsGG = null;
        if(hasPeepholeConnections) {
            rwGradientsFF = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize)});
            rwGradientsOO = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 1)});
            rwGradientsGG = rwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.point(4 * hiddenLayerSize + 2)});
        }

        if(helper != null) {
            Pair sigmoidGates = helper.backpropGradient(conf, gateActivationFn, input, recurrentWeights, inputWeights, epsilon, truncatedBPTT, tbpttBackwardLength, fwdPass, forwards, inputWeightKey, recurrentWeightKey, biasWeightKey, gradientViews, maskArray, hasPeepholeConnections);
            if(sigmoidGates != null) {
                return sigmoidGates;
            }
        }

        boolean var71 = gateActivationFn instanceof ActivationSigmoid;
        IActivation afn = ((org.deeplearning4j.nn.conf.layers.BaseLayer)conf.getLayer()).getActivationFn();
        MemoryWorkspace workspace = Nd4j.getMemoryManager().getCurrentWorkspace() != null && !Nd4j.getMemoryManager().getCurrentWorkspace().getId().equals("LOOP_EXTERNAL")?Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(ComputationGraph.workspaceConfigurationLSTM, "LOOP_LSTM"):null;
        INDArray timeStepMaskColumn = null;

        for(int retGradient = timeSeriesLength - 1; retGradient >= endIdx; --retGradient) {
            if(workspace != null) {
                workspace.notifyScopeEntered();
            }

            int time = retGradient;
            byte inext = 1;
            if(!forwards) {
                time = timeSeriesLength - retGradient - 1;
                inext = -1;
            }

            INDArray nablaCellState;
            if(retGradient != timeSeriesLength - 1 && hasPeepholeConnections) {
                nablaCellState = deltafNext.dup('f').muliRowVector(wFFTranspose);
                l1BLAS.axpy(nablaCellState.length(), 1.0D, deltagNext.dup('f').muliRowVector(wGGTranspose), nablaCellState);
            } else {
                nablaCellState = Nd4j.create(new int[]{miniBatchSize, hiddenLayerSize}, 'f');
            }

            INDArray prevMemCellState = retGradient == 0?fwdPass.prevMemCell:fwdPass.memCellState[time - inext];
            INDArray prevHiddenUnitActivation = retGradient == 0?fwdPass.prevAct:fwdPass.fwdPassOutputAsArrays[time - inext];
            INDArray currMemCellState = fwdPass.memCellState[time];
            INDArray epsilonSlice = is2dInput?epsilon:epsilon.tensorAlongDimension(time, new int[]{1, 0});
            INDArray nablaOut = Shape.toOffsetZeroCopy(epsilonSlice, 'f');
            if(retGradient != timeSeriesLength - 1) {
                Nd4j.gemm(deltaifogNext, wIFOG, nablaOut, false, true, 1.0D, 1.0D);
            }

            INDArray sigmahOfS = fwdPass.memCellActivations[time];
            INDArray ao = fwdPass.oa[time];
            Nd4j.getExecutioner().exec(new OldMulOp(nablaOut, sigmahOfS, deltaoNext));
            INDArray temp;
            if(var71) {
                temp = Nd4j.getExecutioner().execAndReturn(new TimesOneMinus(ao.dup('f')));
                deltaoNext.muli(temp);
            } else {
                deltaoNext.assign((INDArray)gateActivationFn.backprop(fwdPass.oz[time], deltaoNext).getFirst());
            }

            temp = (INDArray)afn.backprop(currMemCellState.dup('f'), ao.muli(nablaOut)).getFirst();
            l1BLAS.axpy(nablaCellState.length(), 1.0D, temp, nablaCellState);
            INDArray af;
            if(hasPeepholeConnections) {
                af = deltaoNext.dup('f').muliRowVector(wOOTranspose);
                l1BLAS.axpy(nablaCellState.length(), 1.0D, af, nablaCellState);
            }

            if(retGradient != timeSeriesLength - 1) {
                af = fwdPass.fa[time + inext];
                int deltaf = nablaCellState.length();
                l1BLAS.axpy(deltaf, 1.0D, af.muli(nablaCellStateNext), nablaCellState);
            }

            nablaCellStateNext = workspace == null?nablaCellState:nablaCellState.leverage();
            af = fwdPass.fa[time];
            INDArray var73 = null;
            INDArray ag;
            if(retGradient > 0 || prevMemCellState != null) {
                var73 = deltafNext;
                if(var71) {
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
            if(var71) {
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
            if(maskArray != null) {
                timeStepMaskColumn = maskArray.getColumn(time);
                deltaifogNext.muliColumnVector(timeStepMaskColumn);
            }

            INDArray prevLayerActivationSlice = Shape.toMmulCompatible(is2dInput?input:input.tensorAlongDimension(time, new int[]{1, 0}));
            INDArray epsilonNextSlice;
            INDArray wi;
            INDArray deltaog;
            if(retGradient <= 0 && prevHiddenUnitActivation == null) {
                epsilonNextSlice = iwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize)});
                Nd4j.gemm(prevLayerActivationSlice, deltaiNext, epsilonNextSlice, true, false, 1.0D, 1.0D);
                wi = iwGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 4 * hiddenLayerSize)});
                deltaog = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 4 * hiddenLayerSize)});
                Nd4j.gemm(prevLayerActivationSlice, deltaog, wi, true, false, 1.0D, 1.0D);
            } else {
                Nd4j.gemm(prevLayerActivationSlice, deltaifogNext, iwGradientsOut, true, false, 1.0D, 1.0D);
            }

            if(retGradient > 0 || prevHiddenUnitActivation != null) {
                Nd4j.gemm(prevHiddenUnitActivation, deltaifogNext, rwGradientsIFOG, true, false, 1.0D, 1.0D);
                if(hasPeepholeConnections) {
                    epsilonNextSlice = var73.dup('f').muli(prevMemCellState).sum(new int[]{0});
                    l1BLAS.axpy(hiddenLayerSize, 1.0D, epsilonNextSlice, rwGradientsFF);
                    wi = deltagNext.dup('f').muli(prevMemCellState).sum(new int[]{0});
                    l1BLAS.axpy(hiddenLayerSize, 1.0D, wi, rwGradientsGG);
                }
            }

            if(hasPeepholeConnections) {
                epsilonNextSlice = deltaoNext.dup('f').muli(currMemCellState).sum(new int[]{0});
                l1BLAS.axpy(hiddenLayerSize, 1.0D, epsilonNextSlice, rwGradientsOO);
            }

            if(retGradient <= 0 && prevHiddenUnitActivation == null) {
                l1BLAS.axpy(hiddenLayerSize, 1.0D, deltaiNext.sum(new int[]{0}), bGradientsOut);
                epsilonNextSlice = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 4 * hiddenLayerSize)}).sum(new int[]{0});
                wi = bGradientsOut.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(2 * hiddenLayerSize, 4 * hiddenLayerSize)});
                l1BLAS.axpy(2 * hiddenLayerSize, 1.0D, epsilonNextSlice, wi);
            } else {
                l1BLAS.axpy(4 * hiddenLayerSize, 1.0D, deltaifogNext.sum(new int[]{0}), bGradientsOut);
            }

            epsilonNextSlice = epsilonNext.tensorAlongDimension(time, new int[]{1, 0});
            if(retGradient <= 0 && prevHiddenUnitActivation == null) {
                wi = inputWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(0, hiddenLayerSize)});
                Nd4j.gemm(deltaiNext, wi, epsilonNextSlice, false, true, 1.0D, 1.0D);
                deltaog = deltaifogNext.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 4 * hiddenLayerSize)});
                INDArray wog = inputWeights.get(new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.interval(2 * hiddenLayerSize, 4 * hiddenLayerSize)});
                Nd4j.gemm(deltaog, wog, epsilonNextSlice, false, true, 1.0D, 1.0D);
            } else {
                Nd4j.gemm(deltaifogNext, inputWeights, epsilonNextSlice, false, true, 1.0D, 1.0D);
            }

            if(maskArray != null) {
                epsilonNextSlice.muliColumnVector(timeStepMaskColumn);
            }

            if(workspace != null) {
                workspace.close();
            }
        }

        DefaultGradient var72 = new DefaultGradient();
        var72.gradientForVariable().put(inputWeightKey, iwGradientsOut);
        var72.gradientForVariable().put(recurrentWeightKey, rwGradientsOut);
        var72.gradientForVariable().put(biasWeightKey, bGradientsOut);
        return new Pair(var72, epsilonNext);
    }

    public static LayerMemoryReport getMemoryReport(AbstractLSTM lstmLayer, InputType inputType) {
        boolean isGraves = lstmLayer instanceof GravesLSTM;
        return getMemoryReport(isGraves, lstmLayer, inputType);
    }

    public static LayerMemoryReport getMemoryReport(GravesBidirectionalLSTM lstmLayer, InputType inputType) {
        LayerMemoryReport r = getMemoryReport(true, lstmLayer, inputType);
        HashMap fixedTrain = new HashMap();
        HashMap varTrain = new HashMap();
        HashMap cacheFixed = new HashMap();
        HashMap cacheVar = new HashMap();
        CacheMode[] var7 = CacheMode.values();
        int var8 = var7.length;

        for(int var9 = 0; var9 < var8; ++var9) {
            CacheMode cm = var7[var9];
            fixedTrain.put(cm, Long.valueOf(2L * ((Long)r.getWorkingMemoryFixedTrain().get(cm)).longValue()));
            varTrain.put(cm, Long.valueOf(2L * ((Long)r.getWorkingMemoryVariableTrain().get(cm)).longValue()));
            cacheFixed.put(cm, Long.valueOf(2L * ((Long)r.getCacheModeMemFixed().get(cm)).longValue()));
            cacheVar.put(cm, Long.valueOf(2L * ((Long)r.getCacheModeMemVariablePerEx().get(cm)).longValue()));
        }

        return (new LayerMemoryReport.Builder(r.getLayerName(), r.getClass(), r.getInputType(), r.getOutputType())).standardMemory(2L * r.getParameterSize(), 2L * r.getUpdaterStateSize()).workingMemory(2L * r.getWorkingMemoryFixedInference(), 2L * r.getWorkingMemoryVariableInference(), fixedTrain, varTrain).cacheMemory(cacheFixed, cacheVar).build();
    }

    public static LayerMemoryReport getMemoryReport(boolean isGraves, FeedForwardLayer lstmLayer, InputType inputType) {
        InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent)inputType;
        int tsLength = itr.getTimeSeriesLength();
        InputType outputType = lstmLayer.getOutputType(-1, inputType);
        int numParams = lstmLayer.initializer().numParams(lstmLayer);
        int updaterSize = (int)lstmLayer.getIUpdater().stateSize((long)numParams);
        int workingMemInferencePerEx = tsLength * 4 * lstmLayer.getNOut();
        int fwdPassPerTimeStepTrainCache = tsLength * 6 * lstmLayer.getNOut();
        int backpropWorkingSpace = (isGraves?9:6) * tsLength * lstmLayer.getNOut();
        HashMap trainVariable = new HashMap();
        HashMap cacheVariable = new HashMap();
        CacheMode[] var13 = CacheMode.values();
        int var14 = var13.length;

        for(int var15 = 0; var15 < var14; ++var15) {
            CacheMode cm = var13[var15];
            long trainWorking;
            long cacheMem;
            if(cm == CacheMode.NONE) {
                trainWorking = (long)(workingMemInferencePerEx + fwdPassPerTimeStepTrainCache + backpropWorkingSpace);
                cacheMem = 0L;
            } else {
                trainWorking = (long)(workingMemInferencePerEx + backpropWorkingSpace);
                cacheMem = (long)fwdPassPerTimeStepTrainCache;
            }

            trainVariable.put(cm, Long.valueOf(trainWorking));
            cacheVariable.put(cm, Long.valueOf(cacheMem));
        }

        return (new LayerMemoryReport.Builder((String)null, lstmLayer.getClass(), inputType, outputType)).standardMemory((long)numParams, (long)updaterSize).workingMemory(0L, (long)workingMemInferencePerEx, MemoryReport.CACHE_MODE_ALL_ZEROS, trainVariable).cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, cacheVariable).build();
    }
}
