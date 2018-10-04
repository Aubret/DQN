package fr.univlyon1.networks;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.Arrays;
import java.util.Collections;

public class LSTMMeanPooling extends LSTM2D {
    public LSTMMeanPooling(LSTM2D lstm, boolean listener) {
        super(lstm, listener);
    }

    public LSTMMeanPooling(int input, int output, long seed){
        super(input,output,seed);
    }


    public void init(){
        int cursor = 0 ;
        if(this.updater ==null){
            this.updater = new Sgd(this.learning_rate);
        }
        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .seed(this.seed+1)
                .trainingWorkspaceMode(WorkspaceMode.SEPARATE)
                .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                //.cacheMode(CacheMode.DEVICE)
                //.l2(0.001)Mlp
                .biasInit(0.1)
                .weightInit(WeightInit.XAVIER)
                .updater(this.updater);
        if(l2 != null) {
            b.l2(this.l2);
        }
        NeuralNetConfiguration.ListBuilder builder = b.list() ;

        //-------------------------------------- Initialisation des couches------------------
        /*Layer lay = new BatchNormalization.Builder().nIn(input).nOut(input).build();
        builder.layer(cursor, lay);
        cursor++;*/

        int node = this.numNodesPerLayer.size() >0 ? this.numNodesPerLayer.get(0) : numNodes ;
        //node = cursor+1 == this.numLayers ? output : node ;
        builder.layer(cursor, new org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                .activation(this.hiddenActivation)
                //.units(node)
                .gateActivationFunction(Activation.SIGMOID)
                .forgetGateBiasInit(1.)
                //.weightInit(WeightInit.XAVIER_UNIFORM)
                .nIn(input).nOut(node)
                .build()
        );

        cursor++;
        for (int i = 1; i < numLayers; i++){
            int previousNode = this.numNodesPerLayer.size() > i-1 ? this.numNodesPerLayer.get(i-1) : numNodes ;
            node = this.numNodesPerLayer.size() > i ? this.numNodesPerLayer.get(i) : numNodes ;
            //node = cursor+1 == this.numLayers ? output : node ;
            //if(i == numLayers -1)
            builder.layer(cursor, new org.deeplearning4j.nn.conf.layers.LSTM.Builder()//new  org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                    .activation(this.hiddenActivation)
                    //.units(node)
                    .gateActivationFunction(Activation.SIGMOID)
                    .forgetGateBiasInit(1.)
                    //.weightInit(WeightInit.XAVIER_UNIFORM)
                    .nIn(previousNode).nOut(node)
                    .build()
            );
            cursor++ ;
        }
        /*node = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor,
                new RnnOutputLayer.Builder()
                        .lossFunction(this.lossFunction)
                        .nIn(node)
                        .nOut(output)
                        .activation(this.lastActivation)
                        .build());*/
        builder.layer(cursor,new GlobalPoolingLayer.Builder(PoolingType.AVG)
            .build()
        );
        cursor++ ;

        //---
        /*int previousNode = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor, new DenseLayer.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .nIn(previousNode).nOut(output)
                .build()
        );
        cursor++;*/
        /*int previousNode = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor, new DenseLayer.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .nIn(previousNode).nOut(output)
                .build()
        );
        cursor++;
        builder.layer(cursor, new LayerNormalizationConf.Builder().build());
        cursor++ ;
        builder.layer(cursor, new ActivationLayer.Builder()
                .activation(Activation.TANH)
                .build()
        );
        cursor++;*/
        //---
        builder.layer(cursor, new LossLayer.Builder().lossFunction(this.lossFunction).build());

        this.multiLayerConfiguration = builder
                .backpropType(BackpropType.Standard)
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        if(this.listener)
            this.attachListener(this.model);
        this.model.init();
        this.tmp = this.model.params().dup();
    }

    public INDArray uncrop2dData(INDArray labels, int number){
        return labels;
    }

    public INDArray crop3dData(INDArray data,INDArray maskLabel){
        return data;
    }

    @Override
    public StateApproximator clone(boolean listener) {
        LSTMMeanPooling m = new LSTMMeanPooling(this,listener);
        m.setHiddenActivation(Activation.TANH);
        m.init();
        m.setParams(this.getParams());
        return m ;
    }
}
