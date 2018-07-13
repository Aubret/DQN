package fr.univlyon1.networks.layers;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.util.OneTimeLogger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;
import java.util.Map;
import java.util.Properties;

public class LSTMLayer extends BaseRecurrentLayer<LSTMLayerConf> {
    private static final Logger log = LoggerFactory.getLogger(LSTM.class);
    public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";
    public static final String STATE_KEY_PREV_MEMCELL = "prevMem";
    protected LSTMHelper helper = null;
    protected FwdPassReturn cachedFwdPass;
    protected FwdPassReturn saved ;

    public LSTMLayer(NeuralNetConfiguration conf) {
        super(conf);
        //this.cacheMode = CacheMode.HOST;
        this.initializeHelper();

    }

    public LSTMLayer(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
        //this.cacheMode = CacheMode.HOST;
        this.initializeHelper();
    }

    public FwdPassReturn getCachedFwdPass(){
        return this.cachedFwdPass ;
    }

    public FwdPassReturn getSaved(){ return this.saved ;}



    void initializeHelper() {
        try {
            this.helper = (LSTMHelper)Class.forName("org.deeplearning4j.nn.layers.recurrent.CudnnLSTMHelper").asSubclass(LSTMHelper.class).newInstance();
            log.debug("CudnnLSTMHelper successfully initialized");
            if(!this.helper.checkSupported(((LSTMLayerConf)this.layerConf()).getGateActivationFn(), ((LSTMLayerConf)this.layerConf()).getActivationFn(), false)) {
                this.helper = null;
            }
        } catch (Throwable var3) {
            if(!(var3 instanceof ClassNotFoundException)) {
                log.warn("Could not initialize CudnnLSTMHelper", var3);
            } else {
                Properties p = Nd4j.getExecutioner().getEnvironmentInformation();
                if(p.getProperty("backend").equals("CUDA")) {
                    OneTimeLogger.info(log, "cuDNN not found: use cuDNN for better GPU performance by including the deeplearning4j-cuda module. For more information, please refer to: https://deeplearning4j.org/cudnn", new Object[]{var3});
                }
            }
        }

    }

    public Gradient gradient() {
        throw new UnsupportedOperationException("gradient() method for layerwise pretraining: not supported for LSTMs (pretraining not possible) " + this.layerId());
    }

    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        return this.backpropGradientHelper(epsilon, false, -1);
    }

    public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackwardLength) {
        return this.backpropGradientHelper(epsilon, true, tbpttBackwardLength);
    }

    private Pair<Gradient, INDArray> backpropGradientHelper(INDArray epsilon, boolean truncatedBPTT, int tbpttBackwardLength) {
        INDArray inputWeights = this.getParamWithNoise("W", true);
        INDArray recurrentWeights = this.getParamWithNoise("RW", true);
        FwdPassReturn fwdPass;
        if(truncatedBPTT) {
            fwdPass = this.activateHelper(true, (INDArray)this.stateMap.get("prevAct"), (INDArray)this.stateMap.get("prevMem"), true);
            this.tBpttStateMap.put("prevAct", fwdPass.lastAct.leverageTo("LOOP_TBPTT"));
            this.tBpttStateMap.put("prevMem", fwdPass.lastMemCell.leverageTo("LOOP_TBPTT"));
        } else {
            //fwdPass = this.activateHelper(true, (INDArray)null, (INDArray)null, true);
            fwdPass=this.saved ;
        }

        Pair p = LSTMHelpers.backpropGradientHelper(this.conf, ((LSTMLayerConf)this.layerConf()).getGateActivationFn(), this.input, recurrentWeights, inputWeights, epsilon, truncatedBPTT, tbpttBackwardLength, fwdPass, true, "W", "RW", "b", this.gradientViews, (INDArray)null, false, this.helper);
        this.weightNoiseParams.clear();
        return p;
    }

    public INDArray preOutput(INDArray x) {
        return this.activate(x, true);
    }

    public INDArray preOutput(INDArray x, boolean training) {
        return this.activate(x, training);
    }

    public INDArray activate(INDArray input, boolean training) {
        this.setInput(input);
        return this.activateHelper(training, (INDArray)null, (INDArray)null, false).fwdPassOutput;
    }

    public INDArray activate(INDArray input) {
        this.setInput(input);
        return this.activateHelper(true, (INDArray)null, (INDArray)null, false).fwdPassOutput;
    }

    public INDArray activate(boolean training) {
        return this.activateHelper(training, (INDArray)null, (INDArray)null, false).fwdPassOutput;
    }

    public INDArray activate() {
        return this.activateHelper(false, (INDArray)null, (INDArray)null, false).fwdPassOutput;
    }

    private FwdPassReturn activateHelper(boolean training, INDArray prevOutputActivations, INDArray prevMemCellState, boolean forBackprop) {
        this.applyDropOutIfNecessary(training);
        if(this.cacheMode == null) {
            this.cacheMode = CacheMode.NONE;
        }

        if(forBackprop && this.cachedFwdPass != null) {
            FwdPassReturn recurrentWeights1 = this.cachedFwdPass;
            this.cachedFwdPass = null;
            return recurrentWeights1;
        } else {
            INDArray recurrentWeights = this.getParamWithNoise("RW", training);
            INDArray inputWeights = this.getParamWithNoise("W", training);
            INDArray biases = this.getParamWithNoise("b", training);
            FwdPassReturn fwd = LSTMHelperLayer.activateHelper(this, this.conf, ((LSTMLayerConf)this.layerConf()).getGateActivationFn(), this.input, recurrentWeights, inputWeights, biases, training, prevOutputActivations, prevMemCellState, training && this.cacheMode != CacheMode.NONE || forBackprop, true, "W", this.maskArray, false, this.helper, forBackprop?this.cacheMode:CacheMode.NONE);
            if(training && this.cacheMode != CacheMode.NONE) {
                this.cachedFwdPass = fwd;
            }
            return fwd;
        }
    }

    public Type type() {
        return Type.RECURRENT;
    }

    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported " + this.layerId());
    }

    public boolean isPretrainLayer() {
        return false;
    }

    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return new Pair(maskArray, MaskState.Passthrough);
    }

    public double calcL2(boolean backpropParamsOnly) {
        double l2Sum = 0.0D;
        Iterator var4 = this.paramTable().entrySet().iterator();

        while(var4.hasNext()) {
            Map.Entry entry = (Map.Entry)var4.next();
            double l2 = this.conf.getL2ByParam((String)entry.getKey());
            if(l2 > 0.0D) {
                double norm2 = this.getParam((String)entry.getKey()).norm2Number().doubleValue();
                l2Sum += 0.5D * l2 * norm2 * norm2;
            }
        }

        return l2Sum;
    }

    public double calcL1(boolean backpropParamsOnly) {
        double l1Sum = 0.0D;
        Iterator var4 = this.paramTable().entrySet().iterator();

        while(var4.hasNext()) {
            Map.Entry entry = (Map.Entry)var4.next();
            double l1 = this.conf.getL1ByParam((String)entry.getKey());
            if(l1 > 0.0D) {
                double norm1 = this.getParam((String)entry.getKey()).norm1Number().doubleValue();
                l1Sum += l1 * norm1;
            }
        }

        return l1Sum;
    }

    public INDArray rnnTimeStep(INDArray input) {
        this.setInput(input);
        FwdPassReturn fwdPass = this.activateHelper(false, (INDArray)this.stateMap.get("prevAct"), (INDArray)this.stateMap.get("prevMem"), false);
        INDArray outAct = fwdPass.fwdPassOutput;
        this.stateMap.put("prevAct", fwdPass.lastAct.detach());
        this.stateMap.put("prevMem", fwdPass.lastMemCell.detach());
        return outAct;
    }

    public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
        this.setInput(input);
        FwdPassReturn fwdPass = this.activateHelper(training, (INDArray)this.tBpttStateMap.get("prevAct"), (INDArray)this.tBpttStateMap.get("prevMem"), true);
        INDArray outAct = fwdPass.fwdPassOutput;
        //this.cachedFwdPass = fwdPass ;
        this.saved = fwdPass ;

        if(storeLastForTBPTT) {
            this.tBpttStateMap.put("prevAct", fwdPass.lastAct.leverageTo("LOOP_TBPTT"));
            this.tBpttStateMap.put("prevMem", fwdPass.lastMemCell.leverageTo("LOOP_TBPTT"));
        }
        return outAct;
    }
}
