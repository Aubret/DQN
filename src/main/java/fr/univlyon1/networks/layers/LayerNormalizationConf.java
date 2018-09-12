package fr.univlyon1.networks.layers;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.NoOp;

import java.util.*;

@Getter
@Setter
public class LayerNormalizationConf extends FeedForwardLayer {

    protected double eps ;
    protected double gamma ;
    protected double beta ;
    protected boolean isLockGammaBeta;

    private LayerNormalizationConf(Builder builder) {
        super(builder);
        this.eps = builder.eps;
        this.gamma = builder.gamma;
        this.beta = builder.beta;
        this.isLockGammaBeta = builder.isLockGammaBeta;
    }

    public LayerNormalizationConf() {
        this.eps = 1.0E-5D;
        this.gamma = 1.0D;
        this.beta = 0.0D;
        this.isLockGammaBeta = false;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration neuralNetConfiguration, Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView, boolean initializeParams) {
        LayerNormalization ret = new LayerNormalization(neuralNetConfiguration);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map paramTable = this.initializer().init(neuralNetConfiguration, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(neuralNetConfiguration);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return LayerNormalizationInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        InputType outputType = this.getOutputType(-1, inputType);
        int numParams = this.initializer().numParams(this);
        int updaterStateSize = 0;

        String trainWorkFixed;
        for (Iterator inferenceWorkingSize = LayerNormalizationInitializer.keys().iterator(); inferenceWorkingSize.hasNext(); updaterStateSize = (int) ((long) updaterStateSize + this.getUpdaterByParam(trainWorkFixed).stateSize((long) this.nOut))) {
            trainWorkFixed = (String) inferenceWorkingSize.next();
        }

        int inferenceWorkingSize1 = 2 * inputType.arrayElementsPerExample();
        int trainWorkFixed1 = 2 * this.nOut;
        int trainWorkingSizePerExample = inferenceWorkingSize1 + outputType.arrayElementsPerExample() + 2 * this.nOut;
        return (new org.deeplearning4j.nn.conf.memory.LayerMemoryReport.Builder(this.layerName, LayerNormalization.class, inputType, outputType)).standardMemory((long) numParams, (long) updaterStateSize).workingMemory(0L, 0L, (long) trainWorkFixed1, (long) trainWorkingSizePerExample).cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS).build();
    }

    public void setNIn(InputType inputType, boolean override) {
        if(this.nIn <= 0 || override) {
            this.nIn = ((InputType.InputTypeFeedForward)inputType).getSize();
            this.nOut = this.nIn;
        }

    }
    //Here's an implementation of a builder pattern, to allow us to easily configure the layer
    //Note that we are inheriting all of the FeedForwardLayer.Builder options: things like n
    public static class Builder extends FeedForwardLayer.Builder<Builder> {

        protected double eps=1.0E-5D;
        protected double gamma=1.;
        protected double beta=0.;
        protected boolean isLockGammaBeta=false;

        public Builder(){

        }

        public LayerNormalizationConf.Builder eps(double eps) {
            this.eps = eps;
            return this;
        }

        public LayerNormalizationConf.Builder gamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        public LayerNormalizationConf.Builder beta(double beta) {
            this.beta = beta;
            return this;
        }

        public LayerNormalizationConf.Builder beta(boolean isLockGammaBeta) {
            this.isLockGammaBeta = isLockGammaBeta;
            return this;
        }


        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public LayerNormalizationConf build() {
            return new LayerNormalizationConf(this);
        }
    }

    public IUpdater getUpdaterByParam(String paramName) {
        byte var3 = -1;
        switch (paramName.hashCode()) {
            case 116519:
                if (paramName.equals("var")) {
                    var3 = 3;
                }
                break;
            case 3020272:
                if (paramName.equals("beta")) {
                    var3 = 0;
                }
                break;
            case 3347397:
                if (paramName.equals("mean")) {
                    var3 = 2;
                }
                break;
            case 98120615:
                if (paramName.equals("gamma")) {
                    var3 = 1;
                }
        }

        switch (var3) {
            case 0:
            case 1:
                return this.iUpdater;
            case 2:
            case 3:
                return new NoOp();
            default:
                throw new IllegalArgumentException("Unknown parameter: \"" + paramName + "\"");

        }

    }

    public double getL1ByParam(String paramName) {
        return 0.0D;
    }

    public double getL2ByParam(String paramName) {
        return 0.0D;
    }

    public String toString() {
        return "LayerNormalization(super=" + super.toString() + ", eps=" + this.getEps()  + ", gamma=" + this.getGamma() + ", beta=" + this.getBeta() + ", lockGammaBeta=" + this.isLockGammaBeta() + ")";
    }

    protected boolean canEqual(Object other) {
        return other instanceof LayerNormalizationConf;
    }


    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        if (!super.equals(o)) return false;

        LayerNormalizationConf that = (LayerNormalizationConf) o;

        if (Double.compare(that.eps, eps) != 0) return false;
        if (Double.compare(that.gamma, gamma) != 0) return false;
        if (Double.compare(that.beta, beta) != 0) return false;
        return isLockGammaBeta == that.isLockGammaBeta;

    }

    @Override
    public int hashCode() {
        int result = super.hashCode();
        long temp;
        temp = Double.doubleToLongBits(eps);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(gamma);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(beta);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (isLockGammaBeta ? 1 : 0);
        return result;
    }
}
