package fr.univlyon1.networks;

import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.OneTimeLogger;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.enums.MirroringPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class EpsilonMultiLayerNetwork extends MultiLayerNetwork {
    public EpsilonMultiLayerNetwork(MultiLayerConfiguration conf) {
        super(conf);
    }

    public EpsilonMultiLayerNetwork(String conf, INDArray params) {
        super(conf, params);
    }

    public EpsilonMultiLayerNetwork(MultiLayerConfiguration conf, INDArray params) {
        super(conf, params);
    }

    public void backpropEpsilon(INDArray epsilon){
        Pair pair = this.calcBackpropGradients(epsilon, false);
        this.gradient = pair == null?null:(Gradient)pair.getFirst();
        this.epsilon = pair == null?null:(INDArray)pair.getSecond();
    }

    public void computeGradientFromEpsilon(INDArray epsilon){
        List workspace;
        Iterator actSecondLastLayer;
        TrainingListener tl;
        if(this.layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
            workspace = this.rnnActivateUsingStoredState(this.getInput(), true, true);
            for(IterationListener it : this.getListeners()){
                if(it instanceof TrainingListener){
                    tl = (TrainingListener)it;
                    tl.onForwardPass(this, workspace);
                }
            }
            this.truncatedBPTTGradient(epsilon);
        } else {
            workspace = this.feedForwardToLayer(this.layers.length - 2, true);
            for(IterationListener it : this.getListeners()){
                if(it instanceof TrainingListener){
                    tl = (TrainingListener)it;
                    tl.onForwardPass(this, workspace);
                }
            }

            INDArray actSecondLastLayer1 = (INDArray)workspace.get(workspace.size() - 1);
            if(this.layerWiseConfigurations.getInputPreProcess(this.layers.length - 1) != null) {
                actSecondLastLayer1 = this.layerWiseConfigurations.getInputPreProcess(this.layers.length - 1).preProcess(actSecondLastLayer1, this.getInputMiniBatchSize());
            }

            this.getOutputLayer().setInput(actSecondLastLayer1);
            this.backpropEpsilon(epsilon);
        }

            if(this.getOutputLayer() instanceof IOutputLayer) {
                this.score = ((IOutputLayer) this.getOutputLayer()).computeScore(this.calcL1(true), this.calcL2(true), true);
            }else {
                this.score = epsilon.mul(epsilon).sumNumber().doubleValue()/(epsilon.shape()[0]);
            }
            if(this.hasTrainingListener()) {
                MemoryWorkspace workspace1 = Nd4j.getMemoryManager().scopeOutOfWorkspaces();
                Throwable actSecondLastLayer2 = null;
                for(IterationListener it : this.getListeners()) {
                    if (it instanceof TrainingListener) {
                        try {
                            TrainingListener tl1 = (TrainingListener) it;
                            tl1.onBackwardPass(this);
                        } catch (Throwable var12) {
                            actSecondLastLayer2 = var12;
                            throw var12;
                        } finally {
                            if (workspace1 != null) {
                                if (actSecondLastLayer2 != null) {
                                    try {
                                        workspace1.close();
                                    } catch (Throwable var11) {
                                        actSecondLastLayer2.addSuppressed(var11);
                                    }
                                } else {
                                    workspace1.close();
                                }
                            }

                        }
                    }
                }

            }

        //}



    }

    public boolean hasTrainingListener(){
        for(IterationListener it : this.getListeners()){
            if(it instanceof TrainingListener){
                return true ;
            }
        }
        return false;
    }

    public EpsilonMultiLayerNetwork clone() {
        MultiLayerConfiguration conf = this.layerWiseConfigurations.clone();
        EpsilonMultiLayerNetwork ret = new EpsilonMultiLayerNetwork(conf);
        ret.init(this.params().dup(), false);
        if(this.solver != null) {
            Updater clonedLayers = this.getUpdater();
            INDArray i = clonedLayers.getStateViewArray();
            if(i != null) {
                ret.getUpdater().setStateViewArray(ret, i.dup(), false);
            }
        }

        if(this.hasAFrozenLayer()) {
            Layer[] var5 = ret.getLayers();
            for(int var6 = 0; var6 < this.layers.length; ++var6) {
                if(this.layers[var6] instanceof FrozenLayer) {
                    var5[var6] = new FrozenLayer(ret.getLayer(var6));
                }
            }
            ret.setLayers(var5);
        }
        return ret;
    }

    private boolean hasAFrozenLayer() {
        for(int i = 0; i < this.layers.length - 1; ++i) {
            if(this.layers[i] instanceof FrozenLayer) {
                return true;
            }
        }

        return false;
    }

    protected void truncatedBPTTGradient(INDArray epsilon) {
        if (this.flattenedGradients == null) {
            this.initGradientsView();
        }

        this.gradient = new DefaultGradient(this.flattenedGradients);
        if (!(this.getOutputLayer() instanceof IOutputLayer)) {
            //.warn("Warning: final layer isn't output layer. You cannot use backprop (truncated BPTT) without an output layer.");
        } else {
            IOutputLayer outputLayer = (IOutputLayer)this.getOutputLayer();
            if (outputLayer instanceof BaseLayer && ((BaseLayer)outputLayer.conf().getLayer()).getWeightInit() == WeightInit.ZERO) {
                throw new IllegalStateException("Output layer weights cannot be initialized to zero when using backprop.");
            } else {
                outputLayer.setLabels(this.labels);
                int numLayers = this.getnLayers();
                LinkedList<Pair<String, INDArray>> gradientList = new LinkedList();
                Pair<Gradient, INDArray> currPair = outputLayer.backpropGradient(epsilon);
                Iterator var7 = ((Gradient)currPair.getFirst()).gradientForVariable().entrySet().iterator();

                String multiGradientKey;
                while(var7.hasNext()) {
                    Map.Entry<String, INDArray> entry = (Map.Entry)var7.next();
                    multiGradientKey = numLayers - 1 + "_" + (String)entry.getKey();
                    gradientList.addLast(new Pair(multiGradientKey, entry.getValue()));
                }

                if (this.getLayerWiseConfigurations().getInputPreProcess(numLayers - 1) != null) {
                    currPair = new Pair(currPair.getFirst(), this.layerWiseConfigurations.getInputPreProcess(numLayers - 1).backprop((INDArray)currPair.getSecond(), this.getInputMiniBatchSize()));
                }

                for(int j = numLayers - 2; j >= 0; --j) {
                    Layer currLayer = this.getLayer(j);
                    if (currLayer instanceof RecurrentLayer) {
                        currPair = ((RecurrentLayer)currLayer).tbpttBackpropGradient((INDArray)currPair.getSecond(), this.layerWiseConfigurations.getTbpttBackLength());
                    } else {
                        currPair = currLayer.backpropGradient((INDArray)currPair.getSecond());
                    }

                    LinkedList<Pair<String, INDArray>> tempList = new LinkedList();
                    Iterator var9 = ((Gradient)currPair.getFirst()).gradientForVariable().entrySet().iterator();

                    while(var9.hasNext()) {
                        Map.Entry<String, INDArray> entry = (Map.Entry)var9.next();
                        multiGradientKey = j + "_" + (String)entry.getKey();
                        tempList.addFirst(new Pair(multiGradientKey, entry.getValue()));
                    }

                    var9 = tempList.iterator();

                    while(var9.hasNext()) {
                        Pair<String, INDArray> pair = (Pair)var9.next();
                        gradientList.addFirst(pair);
                    }

                    if (this.getLayerWiseConfigurations().getInputPreProcess(j) != null) {
                        currPair = new Pair(currPair.getFirst(), this.getLayerWiseConfigurations().getInputPreProcess(j).backprop((INDArray)currPair.getSecond(), this.getInputMiniBatchSize()));
                    }
                }

                var7 = gradientList.iterator();

                while(var7.hasNext()) {
                    Pair<String, INDArray> pair = (Pair)var7.next();
                    this.gradient.setGradientFor((String)pair.getFirst(), (INDArray)pair.getSecond());
                }

            }
        }
    }


}