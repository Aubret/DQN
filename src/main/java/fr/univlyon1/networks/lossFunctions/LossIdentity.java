package fr.univlyon1.networks.lossFunctions;

import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.serde.RowVectorDeserializer;
import org.nd4j.linalg.lossfunctions.serde.RowVectorSerializer;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.shade.jackson.annotation.JsonIgnore;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

public class LossIdentity implements ILossFunction,SaveScore{
    @JsonIgnore
    private INDArray lastScoreArray ;

    public LossIdentity(){}

    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        if(labels.size(1) != preOutput.size(1))
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer number of outputs (nOut = " + preOutput.size(1) + ") ");
        INDArray res = computeScoreArray(labels, preOutput,activationFn,mask);
        double val = res.sumNumber().doubleValue();
        if(average){
            val /= res.size(0);
        }
        return val ;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(labels.size(1) != preOutput.size(1))
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer number of outputs (nOut = " + preOutput.size(1) + ") ");
        INDArray res = activationFn.getActivation(preOutput.dup(), true);
        if( mask != null){
            LossUtil.applyMask(res,mask);
        }
        this.lastScoreArray = res.dup() ;
        return res;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(labels.size(1) != preOutput.size(1))
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer number of outputs (nOut = " + preOutput.size(1) + ") ");

        INDArray dLda = Nd4j.ones(preOutput.shape()) ;
        if(mask != null && LossUtil.isPerOutputMasking(dLda,mask))
            LossUtil.applyMask(dLda,mask);
        INDArray grad = activationFn.backprop(preOutput,dLda).getFirst();
        if(mask != null)
            LossUtil.applyMask(grad,mask);
        return grad ;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(this.computeScore(labels,preOutput,activationFn,mask,average),this.computeGradient(labels,preOutput,activationFn,mask));
    }

    @Override
    public String name() {
        return null;
    }

    public INDArray getLastScoreArray(){
        return this.lastScoreArray;
    }

    @Override
    @JsonIgnore
    public INDArray getValues() {
        return this.lastScoreArray ;
    }
}
