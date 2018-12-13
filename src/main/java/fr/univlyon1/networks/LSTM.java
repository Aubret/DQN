package fr.univlyon1.networks;

import fr.univlyon1.environment.states.HiddenState;
import fr.univlyon1.networks.lossFunctions.SaveScore;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.ILossFunction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Getter
@Setter
public class LSTM extends Mlp implements StateApproximator{

    protected INDArray mask;
    protected INDArray maskLabel;

    public LSTM(LSTM lstm,boolean listener) {// MultiLayerNetwork model,int output){
        super(lstm,listener);
    }

    public LSTM(int input, int output, long seed){
        super(input,output,seed);
    }

    public void init(){

        int cursor = 0 ;
        if(this.updater ==null){
            this.updater = new Sgd(this.learning_rate);
        }
        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .seed(this.seed+1)
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                //.l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(this.updater);
        if(l2 != null) {
            b.l2(this.l2);
        }
        NeuralNetConfiguration.ListBuilder builder = b.list() ;
        //-------------------------------------- Initialisation des couches------------------
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
            //if(i == numLayers -1)
            builder.layer(cursor, new GravesLSTM.Builder()
                    .activation(this.hiddenActivation)
                    .nIn(previousNode).nOut(node)
                    .build()
            );
            cursor++ ;
        }
        node = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor,
                new RnnOutputLayer.Builder()
                        .lossFunction(this.lossFunction)
                        .nIn(node)
                        .nOut(output)
                        .activation(this.lastActivation)
                        .build());
        /*builder.inputPreProcessor(cursor, new RnnToFeedForwardPreProcessor());
        builder.layer(cursor,new LossLayer.Builder(this.lossFunction).build());*/

        cursor++ ;
        /*if(!epsilon)
            builder.layer(cursor, new LossLayer.Builder(this.lossFunction)
                    .build());*/

        this.multiLayerConfiguration = builder
                .backpropType(BackpropType.Standard)
                /*.backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(forwardNumber)
                .tBPTTBackwardLength(backpropNumber)*/
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        if(this.listener)
            this.attachListener(this.model);
        this.model.init();
        if(this.importModel != null){
            try {
                MultiLayerNetwork m = ModelSerializer.restoreMultiLayerNetwork(this.importModel);
                this.model.setParams(m.params());
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        this.tmp = this.model.params().dup();

    }



    public INDArray getOneResult(INDArray data){
        //this.model.setInputMiniBatchSize(data.shape()[0]);
        INDArray res = this.model.rnnTimeStep(data);
        INDArray lastRes = res.get(NDArrayIndex.all(),NDArrayIndex.all());
        return lastRes;
    }


    public INDArray getOneTrainingResult(INDArray data){
        this.model.setLayerMaskArrays(this.mask, this.maskLabel);
        //this.model.rnnClearPreviousState();
        for(int i = 0 ; i < this.model.getnLayers()-1 ; i++) {
            if(this.model.getLayer(i) instanceof org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer)
                ((org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer) this.model.getLayer(i)).rnnSetTBPTTState(new HashMap<>());
        }
        List<INDArray> workspace = this.model.rnnActivateUsingStoredState(data, true, true);
        INDArray last = workspace.get(workspace.size()-1);
        INDArray lastRes = last.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(last.size(2)-1));
        return lastRes;
        //return last.getRow(last.size(0)-1);
    }


    @Override
    public INDArray forwardLearn(INDArray input,INDArray labels,int number,INDArray mask,INDArray maskLabel){
        this.model.rnnClearPreviousState();
        /*for(int i = 0 ; i < this.model.getnLayers()-1 ; i++) { // On nettoie d'abord l'état de sa mémoire
            ((org.deeplearning4j.nn.layers.recurrent.GravesLSTM) this.model.getLayer(i)).rnnSetTBPTTState(new HashMap<>());
        }*/
        this.mask=mask;
        this.maskLabel = maskLabel ;
        //System.out.println(((org.deeplearning4j.nn.layers.recurrent.GravesLSTM) this.model.getLayer(0)).rnnGetTBPTTState());
        this.model.setInputMiniBatchSize(number);
        this.model.setInput(input);
        List<INDArray> workspace = this.model.rnnActivateUsingStoredState(input, true, true);

        for(IterationListener it : this.model.getListeners()){
            if(it instanceof TrainingListener){
                TrainingListener tl = (TrainingListener)it;
                tl.onForwardPass(this.model, workspace);
            }
        }

        INDArray last = workspace.get(workspace.size()-1); // Dernière couche
        //System.out.println(last);
        INDArray getter = last.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(last.size(2)-1));
        //return last.getRow(last.size(0)-1);*/
        return getter ;
    }

    @Override
    public Object learn(INDArray input,INDArray labels,int number) {
        /*INDArray maskLabels = Nd4j.zeros(number,this.forwardNumber);
        maskLabels.putColumn(this.forwardNumber-1,Nd4j.ones(number,1)) ;*/

        /*INDArray nullLabel = Nd4j.zeros(number,numOut,input.size(2));
        INDArrayIndex[] indexs = new INDArrayIndex[]{NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(this.forwardNumber-1)};
        nullLabel.put(indexs,labels);*/

        this.model.setLabels(labels);
        this.model.setLayerMaskArrays(this.mask, this.maskLabel);

        this.model.mytruncatedBPTTGradient();
        this.score = this.model.score() ;
        //System.out.println(this.model.gradient());
        this.model.getUpdater().update(this.model, this.model.gradient(), iterations,this.epoch,number);
        if(this.minimize)
            this.model.params().subi(this.model.gradient().gradient());
        else {
            this.model.params().addi(this.model.gradient().gradient());
        }

        for(IterationListener it : this.model.getListeners()){
            if(it instanceof TrainingListener){
                ((TrainingListener)it).onGradientCalculation(this.model);
            }
            it.iterationDone(this.model, iterations,this.epoch );
        }
        this.iterations++ ;
        this.model.getLayerWiseConfigurations().setIterationCount(this.iterations);
        if(this.model.getOutputLayer() instanceof org.deeplearning4j.nn.layers.LossLayer){
            org.deeplearning4j.nn.layers.LossLayer l = (org.deeplearning4j.nn.layers.LossLayer)this.model.getOutputLayer() ;
            ILossFunction lossFunction = l.layerConf().getLossFn();
            if(lossFunction instanceof SaveScore){
                SaveScore lossfn = (SaveScore)lossFunction ;
                this.values = lossfn.getValues();
            }
        }
        //this.model.clearLayerMaskArrays();
        return this.model.epsilon() ;
    }

    public INDArray error(INDArray input,INDArray labels,int number){
        return null ;
    }


    @Override
    public StateApproximator clone() {
        return this.clone(false);
    }

    @Override
    public StateApproximator clone(boolean listener) {
        LSTM m = new LSTM(this,listener);
        m.setHiddenActivation( Activation.TANH);
        m.init();
        m.setParams(this.getParams().dup());
        return m ;
    }


    @Override
    public Object getMemory() {
        ArrayList<Map<String,INDArray>> memories = new ArrayList<>();

        for(int i=0; i < this.model.getnLayers() ; i++){
            if(this.model.getLayer(i)instanceof org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer) {
                memories.add(this.model.rnnGetPreviousState(i));
            }
            //emories.add(((org.deeplearning4j.nn.layers.recurrent.GravesLSTM) this.model.getLayer(i)).rnnGetTBPTTState());
        }
        return new HiddenState(memories);
    }

    @Override
    public Object getSecondMemory(){
        ArrayList<Map<String,INDArray>> memories = new ArrayList<>();

        for(int i=0; i < this.model.getnLayers() ; i++){
            //memories.add(this.model.rnnGetPreviousState(i));
            if(this.model.getLayer(i)instanceof org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer) {
                org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer recurrent = (org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer) this.model.getLayer(i);
                Map m = recurrent.rnnGetTBPTTState();
                memories.add(m);
            }

        }
        return new HiddenState(memories);
    }

    public Object allMemory(){
        return null ;
    }

    @Override
    public void setMemory(Object memory) {
        if (memory instanceof HiddenState){
            int k=0;
            ArrayList<Map<String, INDArray>> mem = ((HiddenState) memory).getState();
            for(int i = 0;i< this.model.getnLayers();i++){
                if(this.model.getLayer(i)instanceof org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer) {
                    this.model.rnnSetPreviousState(i, mem.get(k));
                    k=k+1;
                    if(k==mem.size())
                        break ;
                }
            }
        }else{
            System.out.println("erreur casting");
        }
    }

    @Override
    public void clear(){
        this.model.rnnClearPreviousState();
    }
}
