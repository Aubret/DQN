package fr.univlyon1.networks.layers;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.AbstractLSTM;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelpers;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.*;

public class LSTMLayerConf extends AbstractLSTM {

    private double forgetGateBiasInit;
    private IActivation gateActivationFn;

    private LSTMLayerConf(LSTMLayerConf.Builder builder) {
        super(builder);
        this.gateActivationFn = new ActivationSigmoid();
        this.forgetGateBiasInit = builder.getForgetGateBiasInit();
        this.gateActivationFn = builder.getGateActivationFn();
        this.initializeConstraints(builder);
    }

    protected void initializeConstraints(org.deeplearning4j.nn.conf.layers.Layer.Builder<?> builder) {
        super.initializeConstraints(builder);
        if(((LSTMLayerConf.Builder)builder).getRecurrentConstraints() != null) {
            if(this.constraints == null) {
                this.constraints = new ArrayList();
            }

            Iterator var2 = ((LSTMLayerConf.Builder)builder).getRecurrentConstraints().iterator();

            while(var2.hasNext()) {
                LayerConstraint c = (LayerConstraint)var2.next();
                LayerConstraint c2 = c.clone();
                c2.setParams(Collections.singleton("RW"));
                this.constraints.add(c2);
            }
        }

    }

    public Layer instantiate(NeuralNetConfiguration conf, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerValidation.assertNInNOutSet("LSTM", this.getLayerName(), layerIndex, this.getNIn(), this.getNOut());
        LSTMLayer ret = new LSTMLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map paramTable = this.initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    public ParamInitializer initializer() {
        return LSTMLayerParamInitializer.getInstance();
    }

    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return LSTMHelpers.getMemoryReport(this, inputType);
    }

    public double getForgetGateBiasInit() {
        return this.forgetGateBiasInit;
    }

    public IActivation getGateActivationFn() {
        return this.gateActivationFn;
    }

    public void setForgetGateBiasInit(double forgetGateBiasInit) {
        this.forgetGateBiasInit = forgetGateBiasInit;
    }

    public void setGateActivationFn(IActivation gateActivationFn) {
        this.gateActivationFn = gateActivationFn;
    }

    public LSTMLayerConf() {
        this.gateActivationFn = new ActivationSigmoid();
    }

    public String toString() {
        return "LSTMLayer(super=" + super.toString() + ", forgetGateBiasInit=" + this.getForgetGateBiasInit() + ", gateActivationFn=" + this.getGateActivationFn() + ")";
    }

    public boolean equals(Object o) {
        if(o == this) {
            return true;
        } else if(!(o instanceof LSTMLayerConf)) {
            return false;
        } else {
            LSTMLayerConf other = (LSTMLayerConf) o;
            if(!other.canEqual(this)) {
                return false;
            } else if(!super.equals(o)) {
                return false;
            } else if(Double.compare(this.getForgetGateBiasInit(), other.getForgetGateBiasInit()) != 0) {
                return false;
            } else {
                IActivation this$gateActivationFn = this.getGateActivationFn();
                IActivation other$gateActivationFn = other.getGateActivationFn();
                if(this$gateActivationFn == null) {
                    if(other$gateActivationFn == null) {
                        return true;
                    }
                } else if(this$gateActivationFn.equals(other$gateActivationFn)) {
                    return true;
                }

                return false;
            }
        }
    }

    protected boolean canEqual(Object other) {
        return other instanceof LSTMLayerConf;
    }

    public int hashCode() {
        boolean PRIME = true;
        int result = super.hashCode();
        long $forgetGateBiasInit = Double.doubleToLongBits(this.getForgetGateBiasInit());
        result = result * 59 + (int)($forgetGateBiasInit >>> 32 ^ $forgetGateBiasInit);
        IActivation $gateActivationFn = this.getGateActivationFn();
        result = result * 59 + ($gateActivationFn == null?43:$gateActivationFn.hashCode());
        return result;
    }

    public static class Builder extends org.deeplearning4j.nn.conf.layers.AbstractLSTM.Builder<LSTMLayerConf.Builder> {
        public LSTMLayerConf build() {
            return new LSTMLayerConf(this);
        }

        public Builder() {
        }

        public double getForgetGateBiasInit(){
            return this.forgetGateBiasInit;
        }

        public IActivation getGateActivationFn(){
            return this.gateActivationFn ;
        }

        public List<LayerConstraint> getRecurrentConstraints(){
            return this.recurrentConstraints ;
        }

    }
}
