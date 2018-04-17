package fr.univlyon1.networks;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.RmsProp;

public class LSTM extends Mlp{

    public LSTM(int input, int output, long seed){
        super(input,output,seed);
        this.hiddenActivation = Activation.TANH ;
    }

    public void init(){
        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .seed(this.seed+1)
                //.l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(this.updater);
        if(this.dropout)
            b.setDropOut(0.5);
        NeuralNetConfiguration.ListBuilder builder = b.list() ;
        //-------------------------------------- Initialisation des couches------------------
        int cursor = 0 ;
        int node = this.numNodesPerLayer.size() >0 ? this.numNodesPerLayer.get(0) : numNodes ;
        builder.layer(cursor, new GravesLSTM.Builder()
                .activation(this.hiddenActivation)
                .nIn(input).nOut(node)
                .build()
        );
        cursor++ ;
        if(this.batchNormalization) {
            builder.layer(cursor, new BatchNormalization.Builder().activation(Activation.RELU).build());
            cursor++ ;
        }
        for (int i = 1; i < numLayers; i++){
            int previousNode = this.numNodesPerLayer.size() > i-1 ? this.numNodesPerLayer.get(i-1) : numNodes ;
            node = this.numNodesPerLayer.size() > i ? this.numNodesPerLayer.get(i) : numNodes ;
            builder.layer(cursor, new DenseLayer.Builder()
                    .activation(this.hiddenActivation)
                    .nIn(previousNode).nOut(node)
                    .build()
            );
            cursor++ ;
            if(i != numLayers-1 && this.batchNormalization) {
                builder.layer(cursor, new BatchNormalization.Builder().activation(Activation.RELU).build());
                cursor++ ;
            }
        }

        if(this.finalBatchNormalization){
            builder.layer(cursor, new BatchNormalization.Builder().activation(Activation.RELU).build());
            cursor++ ;
        }
        node = this.numNodesPerLayer.size() == numLayers ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor,
                new RnnOutputLayer.Builder()
                        .nIn(node)
                        .nOut(output)
                        .activation(this.lastActivation)
                        .build());
        cursor++ ;
        if(!epsilon)
            builder.layer(cursor, new LossLayer.Builder(this.lossFunction)
                    .build());

        this.multiLayerConfiguration = builder
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        if(this.listener)
            this.attachListener(this.model);
        this.model.init();
        this.tmp = this.model.params().dup();
    }
}
