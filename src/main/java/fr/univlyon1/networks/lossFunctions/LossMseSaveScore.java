package fr.univlyon1.networks.lossFunctions;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossL2;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

import java.util.List;

public class LossMseSaveScore extends LossL2 implements SaveScore,ILossFunction {

    @JsonIgnore
    private INDArray lastScoreArray ;

    @JsonIgnore
    private INDArray values ;

    public LossMseSaveScore(){}

    public LossMseSaveScore(INDArray weights) {
        super(weights);
    }

    /*@Override
    protected INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray res = super.scoreArray(labels,preOutput,activationFn,mask);
        System.out.println("ouais");
        this.lastScoreArray = res.dup().detach();
        this.values = activationFn.getActivation(preOutput.dup(),true).detach();
        return res;
    }*/

    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        double score = super.computeScore(labels, preOutput, activationFn, mask, average);
        score /= (double)labels.size(1);
        return score;
    }

    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray scoreArr = super.computeScoreArray(labels, preOutput, activationFn, mask);
        this.lastScoreArray = scoreArr.detach();//.detach();
        this.values = activationFn.getActivation(preOutput,true).detach();//.detach();
        return scoreArr.divi(labels.size(1));
    }

    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray gradients = super.computeGradient(labels, preOutput, activationFn, mask);
        this.computeScoreArray(labels,preOutput,activationFn,mask);
        return gradients.divi(labels.size(1));
    }

    public String name() {
        return this.toString();
    }

    public String toString() {
        return this.weights == null ? "LossMSE()" : "LossMSE(weights=" + this.weights + ")";
    }

    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }

    public INDArray getLastScoreArray() {
        return lastScoreArray;
    }

    public INDArray getValues() {
        return this.values;
    }
}
