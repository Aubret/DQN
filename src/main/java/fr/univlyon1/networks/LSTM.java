package fr.univlyon1.networks;

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
            builder.layer(cursor, new DenseLayer.Builder()
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
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        if(this.listener)
            this.attachListener(this.model);
        this.model.init();
        this.tmp = this.model.params().dup();
    }

    @Override
    public Object learn(INDArray input, INDArray labels, int number) {
        this.model.setInputMiniBatchSize(number);
        this.model.setInput(input);
        if(this.epsilon) {
            this.model.computeGradientFromEpsilon(labels);
        }else {
            this.model.setLabels(labels);
            this.model.computeGradientAndScore();
        }

        //this.model.backpropGradient(Nd4j.create(new Double[]{this.model.getOutputLayer().activate()}))
        //System.out.println(grad.gradientForVariable());
        this.score = this.model.score() ;
        //System.out.println(this.model.acti);
        this.model.getUpdater().update(this.model, this.model.gradient(), iterations, number);
        //if(this.model.getOutputLayer() instanceof IOutputLayer)
        //   System.out.println(this.model.getOutputLayer().gradient().gradient() );
        if(this.minimize)
            this.model.params().subi(this.model.gradient().gradient());
        else {
            this.model.params().addi(this.model.gradient().gradient());
        }

        iterations++ ;
        for(IterationListener it : this.model.getListeners()){
            if(it instanceof TrainingListener){
                ((TrainingListener)it).onGradientCalculation(this.model);
            }
            it.iterationDone(this.model, iterations );
        }

        if(this.model.getOutputLayer() instanceof org.deeplearning4j.nn.layers.LossLayer){
            org.deeplearning4j.nn.layers.LossLayer l = (org.deeplearning4j.nn.layers.LossLayer)this.model.getOutputLayer() ;
            ILossFunction lossFunction = l.layerConf().getLossFn();
            if(lossFunction instanceof LossMseSaveScore){
                SaveScore lossfn = (SaveScore)lossFunction ;
                this.values = lossfn.getValues();
                return lossfn.getLastScoreArray();
            }
        }
        //return this.model.getOutputLayer().com ;
        return null ;
    }
}
