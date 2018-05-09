package fr.univlyon1.networks;

import fr.univlyon1.environment.HiddenState;
import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import fr.univlyon1.networks.lossFunctions.SaveScore;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.ILossFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Getter
@Setter
public class LSTM extends Mlp implements StateApproximator{

    private int backpropNumber = 1 ;
    private int forwardNumber = 1 ;
    private int numOut ;

    public LSTM(LSTM lstm,boolean listener) {// MultiLayerNetwork model,int output){
        super(lstm,listener);
    }

    public LSTM(int input, int output, long seed){
        super(input,output,seed);
        this.hiddenActivation = Activation.TANH ;
        this.numOut = 10 ;
    }

    public void init(){
        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .seed(this.seed+1)
                //.l2(0.001)
                .weightInit(WeightInit.XAVIER);
                //.updater(this.updater);
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
        //builder.inputPreProcessor(cursor, new RnnToFeedForwardPreProcessor());
        //builder.layer(cursor,new LossLayer.Builder(this.lossFunction).build());

        cursor++ ;
        /*if(!epsilon)
            builder.layer(cursor, new LossLayer.Builder(this.lossFunction)
                    .build());*/

        this.multiLayerConfiguration = builder
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(forwardNumber)
                .tBPTTBackwardLength(backpropNumber)
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        if(this.listener)
            this.attachListener(this.model);
        this.model.init();
        this.tmp = this.model.params().dup();
    }



    public INDArray getOneResult(INDArray data){
        //this.model.setInputMiniBatchSize(data.shape()[0]);
        INDArray res = this.model.output(data);
        INDArray lastRes = res.get(NDArrayIndex.all(),NDArrayIndex.all(),NDArrayIndex.point(res.size(2)-1));
        return lastRes;
    }


    public INDArray getOneTrainingResult(INDArray data){
        this.model.rnnClearPreviousState();
        List<INDArray> workspace = this.model.rnnActivateUsingStoredState(data, false, true);
        INDArray last = workspace.get(workspace.size()-1);
        return last.getRow(last.size(0)-1);
    }


    @Override
    public INDArray error(INDArray input,INDArray labels,int number){
        this.model.rnnClearPreviousState();
        this.model.getLayerWiseConfigurations().setTbpttBackLength(this.backpropNumber);
        this.model.getLayerWiseConfigurations().setTbpttFwdLength(this.forwardNumber);
        //System.out.println(this.model.getUpdater().getStateViewArray().getDouble(0));
        //this.model.clear();
        //System.out.println(this.model.getUpdater().getStateViewArray().getDouble(0));

        this.model.setInputMiniBatchSize(number);
        //this.model.clear();
        this.model.setInput(input);
        this.model.setLabels(labels);
        System.out.println(this.model.rnnGetPreviousState(0));
        List<INDArray> workspace = this.model.rnnActivateUsingStoredState(input, true, true);
        System.out.println(this.model.rnnGetPreviousState(0));

        for(IterationListener it : this.model.getListeners()){
            if(it instanceof TrainingListener){
                TrainingListener tl = (TrainingListener)it;
                tl.onForwardPass(this.model, workspace);
            }
        }

        //System.out.println(workspace);
        INDArray last = workspace.get(workspace.size()-1);
        //System.out.println(last);
        return last.getRow(last.size(0)-1);

    }

    @Override
    public Object learn(INDArray input,INDArray labels,int number) {
        this.model.getLayerWiseConfigurations().setTbpttFwdLength(this.forwardNumber);
        this.model.getLayerWiseConfigurations().setTbpttBackLength(this.backpropNumber);
        this.model.setLabels(labels);
        INDArray mask = Nd4j.zeros(number,this.forwardNumber-1);
        mask = Nd4j.concat(1,mask,Nd4j.ones(number,1)) ;
        this.model.getLayer(this.model.getnLayers()-1).setMaskArray(mask);

        this.model.mytruncatedBPTTGradient();
        //this.model.backpropGradient(Nd4j.create(new Double[]{this.model.getOutputLayer().activate()}))
        //System.out.println(grad.gradientForVariable());
        this.score = this.model.score() ;
        //System.out.println(this.model.acti);
        this.model.getUpdater().update(this.model, this.model.gradient(), iterations,this.epoch,number);
        //if(this.model.getOutputLayer() instanceof IOutputLayer)
        //   System.out.println(this.model.getOutputLayer().gradient().gradient() );
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
            if(lossFunction instanceof LossMseSaveScore){
                SaveScore lossfn = (SaveScore)lossFunction ;
                this.values = lossfn.getValues();
            }
        }
        //return this.model.getOutputLayer().com ;
        return this.model.epsilon() ;
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
}
