package main.java.fr.univlyon1.networks;

import main.java.fr.univlyon1.Configuration;
import main.java.fr.univlyon1.actorcritic.Learning;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Collection;
import java.util.List;

public class Mlp implements Approximator{
    private MultiLayerNetwork model ;
    private INDArray tmp ;
    private int iterations = 0;

    public Mlp(MultiLayerNetwork model){
        this.model = model ;
    }

    public Mlp(int input, int output,long seed,boolean listener,Learning learning) {
        Configuration conf = learning.getConf() ;
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(learning.getConf().getLearning_rate())
                .biasInit(0.1)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                //.regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(input).nOut(conf.getNumHiddenNodes())
                        .build());
        for (int i = 1; i <= learning.getConf().getNumLayers(); i++){
            builder.layer(i, new DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nIn(conf.getNumHiddenNodes()).nOut(conf.getNumHiddenNodes())
                    .build());
        }
        MultiLayerConfiguration multiLayerConfiguration = builder.layer(learning.getConf().getNumLayers()+1,
                new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                    .nIn(conf.getNumHiddenNodes())
                    .nOut(output)
                    .activation(Activation.IDENTITY)
                    .build())
            .backprop(true).pretrain(false)
            .build();
        this.model = new MultiLayerNetwork(multiLayerConfiguration);
        if(listener)
            this.attachListener(this.model);
        this.model.init();
        this.tmp = this.model.params();
    }

    public INDArray getOneResult(INDArray data){
        INDArray res = this.model.output(data) ;
        return res ;
    }

    @Override
    public void learn(INDArray input,INDArray labels,int number) {
        this.model.setInputMiniBatchSize(number);
        //this.model.fit(input,labels);
        this.model.setInput(input);
        this.model.setLabels(labels);
        this.model.computeGradientAndScore();
        iterations++ ;
        for(IterationListener it : this.model.getListeners()){
            if(it instanceof TrainingListener){
                ((TrainingListener)it).onGradientCalculation(this.model);
            }
            it.iterationDone(this.model, iterations );
        }
        /*System.out.println(("----------"));
        System.out.println(this.model.output(input));
        System.out.println(labels);
        System.out.println(this.model.score());*/
        //this.model.update(this.model.gradient());
        this.model.getUpdater().update(this.model, this.model.gradient(), iterations, number);
        this.model.params().subi(this.model.gradient().gradient());
        /*System.out.println(this.model.params());
        System.out.println(this.tmp);
        System.out.println(this.model.gradient().gradient());
        System.out.println(this.model.output(input));*/
        //this.model.update(this.model.gradient());


    }

    @Override
    public Approximator clone() {
        return this.clone(false);
    }

    @Override
    public Approximator clone(boolean listener) {
        /*if (listener)
            this.attachListener(this.model);
        return this ;*/
        MultiLayerNetwork mlp  = this.model.clone();
        mlp.setListeners();
        if(listener)
            this.attachListener(mlp);
        //return this ;
        Mlp m = new Mlp(mlp);
        m.tmp = this.tmp ;
        return m ;
    }

    @Override
    public void stop() {
        System.out.println(this.model.params());
    }

    private void attachListener(MultiLayerNetwork mlp){
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        mlp.setListeners(new StatsListener(statsStorage));
    }
}
