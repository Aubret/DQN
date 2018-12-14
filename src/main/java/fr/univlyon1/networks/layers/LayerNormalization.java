package fr.univlyon1.networks.layers;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastAddOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastDivOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastSubOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.ConditionEquals;
import org.nd4j.linalg.indexing.conditions.EqualsCondition;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
    Do not handle more than 2D data size and gammalocking
 **/
public class LayerNormalization extends BaseLayer<LayerNormalizationConf>{

    protected List<TrainingListener> listeners = new ArrayList();
    protected INDArray std;
    protected INDArray xMu;
    protected INDArray xHat;

    public LayerNormalization(NeuralNetConfiguration conf) {
        super(conf);
    }

    public INDArray preOutput(INDArray x, LayerWorkspaceMgr workspaceMgr) {
        return this.preOutput(x, TrainingMode.TRAIN,workspaceMgr);
    }

    public INDArray preOutput(INDArray x, TrainingMode training, LayerWorkspaceMgr workspaceMgr) {
        /*System.out.println("-------------------");
        System.out.println(x);*/
        LayerNormalizationConf layerConf = (LayerNormalizationConf) this.layerConf();
        INDArray mean;
        INDArray var;

        switch(x.rank()) {
            case 2:
                mean = x.mean(1);
                var = x.var(false, 1).addi(((LayerNormalizationConf)this.layerConf()).getEps());
                break;
            default:
                throw new IllegalStateException("Layer normalization on activations of rank " + x.rank() + " not supported " + this.layerId());
        }
        this.std = Transforms.sqrt(var,true );
        //this.std = Transforms.sqrt(workspaceMgr.dup(ArrayType.INPUT, var).addi((this.layerConf()).getEps()), false);
        this.xMu = x.subColumnVector(mean).leverageTo("LOOP_EXTERNAL");
        this.xHat = this.xMu.divColumnVector(this.std).leverageTo("LOOP_EXTERNAL");
        /*this.xMu = workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
        this.xMu = Nd4j.getExecutioner().execAndReturn(new BroadcastSubOp(x, mean, this.xMu, new int[]{1}));

        this.xHat = workspaceMgr.createUninitialized(ArrayType.INPUT, x.shape(), x.ordering());
        this.xHat = Nd4j.getExecutioner().execAndReturn(new BroadcastDivOp(this.xMu, this.std, this.xHat, new int[]{1}));*/
        INDArray gamma = this.getParam("gamma");//.add(1);
        //INDArray gamma = Nd4j.create(new double[]{1.,2.,3.,4.,5.});
        INDArray beta = this.getParam("beta");

        INDArray activations = this.xHat.mulRowVector(gamma).addiRowVector(beta);

        /*INDArray activations = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, x.shape(), x.ordering());
        activations = Nd4j.getExecutioner().execAndReturn(new BroadcastMulOp(this.xHat, gamma, activations, new int[]{1}));
        activations = Nd4j.getExecutioner().execAndReturn(new BroadcastAddOp(activations, beta, activations, new int[]{1}));*/


        return activations ;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        long[] shape = this.getShape(epsilon);
        if(shape.length != 2 ){
            throw new IllegalStateException("bad input size, must be two, 4 is not implemented");
        }
        LayerNormalizationConf layerConf = this.layerConf();
        INDArray gamma = this.getParam("gamma");
        INDArray dGammaView = (INDArray)this.gradientViews.get("gamma");
        INDArray dBetaView = (INDArray)this.gradientViews.get("beta");
        /*INDArray dGlobalMeanView = (INDArray)this.gradientViews.get("mean");
        INDArray dGlobalVarView = (INDArray)this.gradientViews.get("var");*/

        DefaultGradient retGradient1 = new DefaultGradient();
        /*System.out.println("----------");
        System.out.println(epsilon);*/
        INDArray dBeta1 = epsilon.sum(0);
        INDArray dGamma = epsilon.mul(this.xHat).sum(0);

        INDArray dxhat = epsilon.mulRowVector(gamma);//step one, bizarre le mulroxvectior
        //---- rank 2
        INDArray dLdVar = dxhat.mul(this.xMu).sum(1).muli(Double.valueOf(-0.5D)).muli(Transforms.pow(this.std, Double.valueOf(-3.0D), true));
        double numberUnits = shape[1];
        double batchsize = shape[0];
        INDArray dxmu1 = this.xMu.sum(1).muli(Double.valueOf(-2.0D / numberUnits)).muli(dLdVar);
        INDArray dxmu = dxhat.sum(1).divi(this.std).negi();
        dxmu.addi(dxmu1);
        INDArray dLdmu = dxhat
                .diviColumnVector(this.std)
                .addi(this.xMu.muliColumnVector(dLdVar.muli(Double.valueOf(2.0D / numberUnits))))
                .addiColumnVector(dxmu.muli(Double.valueOf(1.0D /numberUnits)));


        dGammaView.assign(dGamma);
        dBetaView.assign(dBeta1);
        retGradient1.setGradientFor("gamma", dGammaView);
        retGradient1.setGradientFor("beta", dBetaView);
        /*dGlobalMeanView.assign(Integer.valueOf(0));
        dGlobalVarView.assign(Integer.valueOf(0));
        retGradient1.setGradientFor("mean", dGlobalMeanView);
        retGradient1.setGradientFor("var", dGlobalVarView);*/
        INDArray nextEpsilon = dLdmu;
        //--
        /*System.out.println(xMu);
        System.out.println(this.std);
        System.out.println(nextEpsilon);*/
        //System.out.println(nextEpsilon);
        return new Pair(retGradient1, nextEpsilon);
    }

    public long[] getShape(INDArray x) {
        if (x.rank() != 2 && x.rank() != 4) {
            if (x.rank() == 3) {
                long wDim = x.size(1);
                long hdim = x.size(2);
                if (x.size(0) > 1L && wDim * hdim == x.length()) {
                    throw new IllegalArgumentException("Illegal input for batch size " + this.layerId());
                } else {
                    return new long[]{1L, wDim * hdim};
                }
            } else {
                throw new IllegalStateException("Unable to process input of rank " + x.rank() + " " + this.layerId());
            }
        } else {
            return new long[]{1L, x.size(1)};
        }
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    public void setListeners(TrainingListener... listeners) {
        this.listeners = new ArrayList(Arrays.asList(listeners));
    }

    public Collection<TrainingListener> getListeners() {
        return this.listeners;
    }


    public double calcL2(boolean backpropParamsOnly) {
        return 0.0D;
    }

    public double calcL1(boolean backpropParamsOnly) {
        return 0.0D;
    }

    public Type type() {
        return Type.NORMALIZATION;
    }

    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        this.assertInputSet(false);
        return this.preOutput(this.input, training ? TrainingMode.TRAIN : TrainingMode.TEST, workspaceMgr);
    }


    public void fit(INDArray data) {
    }

}
