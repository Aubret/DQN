package fr.univlyon1.networks;

import fr.univlyon1.environment.HiddenState;
import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import fr.univlyon1.networks.lossFunctions.SaveScore;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.ILossFunction;

import java.util.ArrayList;
import java.util.Map;

public class LSTM extends Mlp implements StateApproximator{

    public LSTM(LSTM lstm,boolean listener) {// MultiLayerNetwork model,int output){
        super(lstm,listener);
    }

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
        NeuralNetConfiguration.ListBuilder builder = b.list() ;
        //-------------------------------------- Initialisation des couches------------------
        int cursor = 0 ;
        int node = this.numNodesPerLayer.size() >0 ? this.numNodesPerLayer.get(0) : numNodes ;
        builder.layer(cursor, new GravesLSTM.Builder()
                .activation(this.hiddenActivation)
                .nIn(input).nOut(node)
                .build()
        );
        cursor++;
        for (int i = 1; i < numLayers; i++){
            int previousNode = this.numNodesPerLayer.size() > i-1 ? this.numNodesPerLayer.get(i-1) : numNodes ;
            node = this.numNodesPerLayer.size() > i ? this.numNodesPerLayer.get(i) : numNodes ;
            builder.layer(cursor, new GravesLSTM.Builder()
                    .activation(this.hiddenActivation)
                    .nIn(previousNode).nOut(node)
                    .build()
            );
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
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(1)
                .tBPTTBackwardLength(1)
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        if(this.listener)
            this.attachListener(this.model);
        this.model.init();
        this.tmp = this.model.params().dup();
    }

    @Override
    public Object getMemory() {
        ArrayList<Map<String,INDArray>> memories = new ArrayList<>();
        for(int i=0; i < this.numLayers-1 ; i++){
            memories.add(this.model.rnnGetPreviousState(i));
        }
        return new HiddenState(memories);
    }

    @Override
    public void setMemory(Object memory) {
        if (memory instanceof HiddenState){
            ArrayList<Map<String, INDArray>> mem = ((HiddenState) memory).getState();
            for(int i = 0;i< mem.size();i++){
                this.model.rnnSetPreviousState(i,mem.get(i));
            }
        }else{
            System.out.println("erreur casting");
        }
    }

    public INDArray getOneResult(INDArray data){
        //this.model.setInputMiniBatchSize(data.shape()[0]);
        INDArray res ;
        //System.out.println(this.model.params());
        //Map<String, INDArray> hiddenState = this.model.rnnGetPreviousState(0);
        res = this.model.rnnTimeStep(data);
        //System.out.println(hiddenState);
        //System.out.println(res);
        //System.out.println(this.model.output(data, org.deeplearning4j.nn.api.Layer.TrainingMode.TEST));
        //System.out.println(this.model.rnnActivateUsingStoredState(data,false,false));
        res = res.reshape(res.shape()[0],res.shape()[1]);
        return res;
    }


    @Override
    public StateApproximator clone() {
        return this.clone(false);
    }

    @Override
    public StateApproximator clone(boolean listener) {
        LSTM m = new LSTM(this,listener);
        return m ;
    }

}
