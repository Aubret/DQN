package fr.univlyon1.networks.lossFunctions;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.primitives.Pair;

public class LossError implements ILossFunction {


    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        if(labels.size(1) != preOutput.size(1))
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer number of outputs (nOut = " + preOutput.size(1) + ") ");
        INDArray res = computeScoreArray(labels, preOutput,activationFn,mask);
        double val = res.muli(res).sumNumber().doubleValue();
        if(average){
            val /= res.size(0);
        }
        return val ;
    }

    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(labels.size(1) != preOutput.size(1))
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer number of outputs (nOut = " + preOutput.size(1) + ") ");
        INDArray res = activationFn.getActivation(preOutput.dup(), true).muli(labels);
        if( mask != null){
            LossUtil.applyMask(res,mask);
        }
        //System.out.println("Use of score");
        return res;
    }

    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask) {
        if(labels.size(1) != preOutput.size(1))
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer number of outputs (nOut = " + preOutput.size(1) + ") ");
        //INDArray dLda = activationFn.backprop(preOutput.dup(), labels).getFirst();
        INDArray dLda = labels ;

        if(mask != null)
            LossUtil.applyMask(dLda,mask);
        return dLda ;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(this.computeScore(labels,preOutput,activationFn,mask,average),this.computeGradient(labels,preOutput,activationFn,mask));
    }

    @Override
    public String name() {
        return null;
    }
}

