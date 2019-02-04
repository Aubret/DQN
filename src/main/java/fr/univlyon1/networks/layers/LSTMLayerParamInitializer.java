package fr.univlyon1.networks.layers;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.LSTMParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

public class LSTMLayerParamInitializer implements ParamInitializer {
    private static final LSTMLayerParamInitializer INSTANCE = new LSTMLayerParamInitializer();
    public static final String RECURRENT_WEIGHT_KEY = "RW";
    public static final String BIAS_KEY = "b";
    public static final String INPUT_WEIGHT_KEY = "W";
    private static final List<String> LAYER_PARAM_KEYS = Collections.unmodifiableList(Arrays.asList(new String[]{"W", "RW", "b"}));
    private static final List<String> WEIGHT_KEYS = Collections.unmodifiableList(Arrays.asList(new String[]{"W", "RW"}));
    private static final List<String> BIAS_KEYS = Collections.unmodifiableList(Collections.singletonList("b"));

    public LSTMLayerParamInitializer() {
    }

    public static LSTMLayerParamInitializer getInstance() {
        return INSTANCE;
    }

    public long numParams(NeuralNetConfiguration conf) {
        return this.numParams(conf.getLayer());
    }

    public long numParams(Layer l) {
        LSTMLayerConf layerConf = (LSTMLayerConf) l;
        long nL = layerConf.getNOut();
        long nLast = layerConf.getNIn();
        long nParams = nLast * 4 * nL + nL * 4 * nL + 4 * nL;
        return nParams;
    }

    public List<String> paramKeys(Layer layer) {
        return LAYER_PARAM_KEYS;
    }

    public List<String> weightKeys(Layer layer) {
        return WEIGHT_KEYS;
    }

    public List<String> biasKeys(Layer layer) {
        return BIAS_KEYS;
    }

    public boolean isWeightParam(Layer layer, String key) {
        return "RW".equals(key) || "W".equals(key);
    }

    public boolean isBiasParam(Layer layer, String key) {
        return "b".equals(key);
    }

    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        Map params = Collections.synchronizedMap(new LinkedHashMap());
        LSTMLayerConf layerConf = (LSTMLayerConf)conf.getLayer();
        double forgetGateInit = layerConf.getForgetGateBiasInit();
        Distribution dist = Distributions.createDistribution(layerConf.getDist());
        long nL = layerConf.getNOut();
        long nLast = layerConf.getNIn();
        conf.addVariable("W");
        conf.addVariable("RW");
        conf.addVariable("b");
        long length = this.numParams(conf);
        if(paramsView.length() != length) {
            throw new IllegalStateException("Expected params view of length " + length + ", got length " + paramsView.length());
        } else {
            long nParamsIn = nLast * 4 * nL;
            long nParamsRecurrent = nL * 4 * nL;
            long nBias = 4 * nL;
            INDArray inputWeightView = paramsView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(0, nParamsIn)});
            INDArray recurrentWeightView = paramsView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent)});
            INDArray biasView = paramsView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias)});
            if(initializeParams) {
                long fanOut = nLast + nL;
                long[] inputWShape = new long[]{nLast, 4 * nL};
                long[] recurrentWShape = new long[]{nL, 4 * nL};
                Distribution rwDist = dist;
                WeightInit rwInit;
                if(layerConf.getWeightInitRecurrent() != null) {
                    rwInit = layerConf.getWeightInitRecurrent();
                    if(layerConf.getDistRecurrent() != null) {
                        rwDist = Distributions.createDistribution(layerConf.getDistRecurrent());
                    }
                } else {
                    rwInit = layerConf.getWeightInit();
                }

                params.put("W", WeightInitUtil.initWeights((double)nL, (double)fanOut, inputWShape, layerConf.getWeightInit(), dist, inputWeightView));
                params.put("RW", WeightInitUtil.initWeights((double)nL, (double)fanOut, recurrentWShape, rwInit, rwDist, recurrentWeightView));
                biasView.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(nL, 2 * nL)}, Nd4j.valueArrayOf(1, nL, forgetGateInit));
                params.put("b", biasView);
            } else {
                params.put("W", WeightInitUtil.reshapeWeights(new long[]{nLast, 4 * nL}, inputWeightView));
                params.put("RW", WeightInitUtil.reshapeWeights(new long[]{nL, 4 * nL}, recurrentWeightView));
                params.put("b", biasView);
            }

            return params;
        }
    }

    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        LSTMLayerConf layerConf = (LSTMLayerConf)conf.getLayer();
        long nL = layerConf.getNOut();
        long nLast = layerConf.getNIn();
        long length = this.numParams(conf);
        if(gradientView.length() != length) {
            throw new IllegalStateException("Expected gradient view of length " + length + ", got length " + gradientView.length());
        } else {
            long nParamsIn = nLast * 4 * nL;
            long nParamsRecurrent = nL * 4 * nL;
            long nBias = 4 * nL;
            INDArray inputWeightGradView = gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(0, nParamsIn)}).reshape('f', nLast, 4 * nL);
            INDArray recurrentWeightGradView = gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent)}).reshape('f', nL, 4 * nL);
            INDArray biasGradView = gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias)});
            LinkedHashMap out = new LinkedHashMap();
            out.put("W", inputWeightGradView);
            out.put("RW", recurrentWeightGradView);
            out.put("b", biasGradView);
            return out;
        }
    }
}
