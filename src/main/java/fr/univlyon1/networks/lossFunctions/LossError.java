package fr.univlyon1.networks.lossFunctions;

import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossUtil;
import org.nd4j.linalg.primitives.Pair;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.List;
import java.util.Map;

public class LossError extends DifferentialFunction implements ILossFunction {


    @Override
    public double computeScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        if(labels.size(1) != preOutput.size(1))
            throw new IllegalArgumentException("Labels array numColumns (size(1) = " + labels.size(1) + ") does not match output layer number of outputs (nOut = " + preOutput.size(1) + ") ");
        INDArray res = computeScoreArray(labels.dup(), preOutput,activationFn,mask);
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
        INDArray res = labels.dup();//activationFn.getActivation(preOutput.dup(), true).muli(labels);
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
        INDArray dLda = labels.dup();

        if(mask != null)
            LossUtil.applyMask(dLda,mask);
        return dLda ;
    }

    @Override
    public Pair<Double, INDArray> computeGradientAndScore(INDArray labels, INDArray preOutput, IActivation activationFn, INDArray mask, boolean average) {
        return new Pair<>(this.computeScore(labels,preOutput,activationFn,mask,average),this.computeGradient(labels,preOutput,activationFn,mask));
    }

    public String name() {
        return "Loss error";
    }

    public String toString() {
        return "LossMSE()";
    }

    public SDVariable[] outputVariables() {
        return new SDVariable[0];
    }

    public SDVariable[] outputVariables(String baseName) {
        return new SDVariable[0];
    }

    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return null;
    }

    public String opName() {
        return this.name();
    }

    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
    }

    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
    }

    @Override
    public String onnxName() {
        return "Loss error";
    }

    @Override
    public String tensorflowName() {
        return "Loss error";
    }
}

