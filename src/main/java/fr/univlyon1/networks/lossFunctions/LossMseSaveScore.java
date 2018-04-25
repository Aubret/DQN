package fr.univlyon1.networks.lossFunctions;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.shade.jackson.annotation.JsonIgnore;

public class LossMseSaveScore extends LossMSE implements SaveScore,ILossFunction {

    @JsonIgnore
    private INDArray lastScoreArray ;

    @JsonIgnore
    private INDArray values ;

    public LossMseSaveScore(){}

    public LossMseSaveScore(INDArray weights) {
        super(weights);
    }

    @Override
    protected INDArray scoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        INDArray res = super.scoreArray(labels,preOutput,activationFn,mask);
        this.lastScoreArray = res;
        this.values = activationFn.getActivation(preOutput.dup(),true);
        return res;
    }

    public INDArray getLastScoreArray() {
        return lastScoreArray;
    }

    public INDArray getValues() {
        return this.values;
    }
}
