package fr.univlyon1.networks.layers;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

public class LayerNormalizationInitializer implements ParamInitializer {
    private static final LayerNormalizationInitializer INSTANCE = new LayerNormalizationInitializer();
    public static final String GAMMA = "gamma";
    public static final String BETA = "beta";

    public static List<String> keys() {
        return Arrays.asList(new String[]{"gamma", "beta"});
    }


    public int numParams(NeuralNetConfiguration neuralNetConfiguration) {
        return this.numParams(neuralNetConfiguration.getLayer());
    }

    public int numParams(Layer l) {
        LayerNormalizationConf layer = (LayerNormalizationConf) l;
        return 2 * layer.getNOut();
    }


    public List<String> paramKeys(Layer layer) {
        return Arrays.asList(new String[]{"gamma", "beta", "mean", "var"});
    }

    public List<String> weightKeys(Layer layer) {
        return Collections.emptyList();
    }

    public List<String> biasKeys(Layer layer) {
        return Collections.emptyList();
    }

    public boolean isWeightParam(Layer layer, String key) {
        return false;
    }

    public boolean isBiasParam(Layer layer, String key) {
        return false;
    }

    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramView, boolean initializeParams) {
        Map params = Collections.synchronizedMap(new LinkedHashMap());
        LayerNormalizationConf layer = (LayerNormalizationConf) conf.getLayer();
        int nOut = layer.getNOut();
        int meanOffset = 0;
        INDArray globalMeanView;
        INDArray globalVarView;
        if(!layer.isLockGammaBeta()) {
            globalMeanView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut)});
            globalVarView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(nOut, 2 * nOut)});

            params.put("gamma", this.createGamma(conf, globalMeanView, initializeParams));
            conf.addVariable("gamma");
            params.put("beta", this.createBeta(conf, globalVarView, initializeParams));
            conf.addVariable("beta");
            //meanOffset = 2 * nOut;
            //meanOffset = 0 ;
        }

        /*globalMeanView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(meanOffset, meanOffset + nOut)});
        globalVarView = paramView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(meanOffset + nOut, meanOffset + 2 * nOut)});
        if(initializeParams) {
            globalMeanView.assign(Integer.valueOf(0));
            globalVarView.assign(Integer.valueOf(1));
        }

        params.put("mean", globalMeanView);
        conf.addVariable("mean");
        params.put("var", globalVarView);
        conf.addVariable("var");*/
        return params;
    }

    public static LayerNormalizationInitializer getInstance() {
        return INSTANCE;
    }

    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        LayerNormalizationConf layer = (LayerNormalizationConf)conf.getLayer();
        int nOut = layer.getNOut();
        LinkedHashMap out = new LinkedHashMap();
        int meanOffset = 0;
        if(!layer.isLockGammaBeta()) {
            INDArray gammaView = gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(0, nOut)});
            INDArray betaView = gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(nOut, 2 * nOut)});
            out.put("gamma", gammaView);
            out.put("beta", betaView);
            //meanOffset = 2 * nOut;
        }

        //out.put("mean", gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(meanOffset, meanOffset + nOut)}));
        //out.put("var", gradientView.get(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.interval(meanOffset + nOut, meanOffset + 2 * nOut)}));
        return out;
    }

    private INDArray createBeta(NeuralNetConfiguration conf, INDArray betaView, boolean initializeParams) {
        LayerNormalizationConf layer = (LayerNormalizationConf)conf.getLayer();
        if(initializeParams) {
            betaView.assign(Double.valueOf(layer.getBeta()));
        }

        return betaView;
    }

    private INDArray createGamma(NeuralNetConfiguration conf, INDArray gammaView, boolean initializeParams) {
        LayerNormalizationConf layer = (LayerNormalizationConf)conf.getLayer();
        if(initializeParams) {
            gammaView.assign(layer.getGamma());
        }

        return gammaView;
    }

}
