package fr.univlyon1.networks;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
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

@Slf4j
public class Mlp implements Approximator{
    protected EpsilonMultiLayerNetwork model ;
    protected INDArray tmp ;
    protected int iterations = 0;
    protected int output ;
    protected MultiLayerConfiguration multiLayerConfiguration ;
    protected boolean minimize ; // Minimize or maximize the loss function
    protected boolean epsilon ; // use a loss function with label or direct errors ?

    public Mlp(Mlp mlp,boolean listener){// MultiLayerNetwork model,int output){
        this.model = mlp.getModel().clone() ;
        this.output = mlp.getOutput() ;
        this.tmp = mlp.tmp ;
        this.minimize = mlp.minimize ;
        if(listener)
            this.attachListener(this.model);
    }



    public Mlp(int input, int output,long seed,boolean listener,Double learning_rate, int numLayers, int numNodes,boolean minimize,boolean epsilon) {
        this.output = output ;
        this.minimize = minimize ;
        this.epsilon = epsilon ;
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(seed+1)
                .learningRate(learning_rate)
                .biasInit(0)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .minimize(minimize)
                //.regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(input).nOut(numNodes)
                        .build());
        for (int i = 1; i <= numLayers; i++){
            builder.layer(i, new DenseLayer.Builder()
                    .activation(Activation.RELU)
                    .nIn(numNodes).nOut(numNodes)
                    .build());
        }
        builder.layer(numLayers+1,
                new DenseLayer.Builder()
                    .nIn(numNodes)
                    .nOut(output)
                    .activation(Activation.IDENTITY)
                    .build());
        if(!epsilon)
            builder.layer(numLayers+2, new LossLayer.Builder(LossFunctions.LossFunction.MSE)
                        .build());

        this.multiLayerConfiguration = builder
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
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
    public int numOutput() {
        return this.output;
    }

    @Override
    public Object learn(INDArray input,INDArray labels,int number) {
        //this.model.setInputMiniBatchSize(number);
        //this.model.fit(input,labels);
        this.model.setInput(input);
        if(this.epsilon) {
            System.out.println(("here"));
            System.out.println(this.model.output(input));
            this.model.computeGradientFromEpsilon(labels);
            //System.out.println(this.model.gradient().gradient());
        }else {
            this.model.setLabels(labels);
            this.model.computeGradientAndScore();
        }
        iterations++ ;
        for(IterationListener it : this.model.getListeners()){
            if(it instanceof TrainingListener){
                ((TrainingListener)it).onGradientCalculation(this.model);
            }
            it.iterationDone(this.model, iterations );
        }
        //this.model.backpropGradient(Nd4j.create(new Double[]{this.model.getOutputLayer().activate()}))
        //System.out.println(grad.gradientForVariable());
        this.model.getUpdater().update(this.model, this.model.gradient(), iterations, number);
        if(this.minimize)
            this.model.params().subi(this.model.gradient().gradient());
        else
            this.model.params().addi(this.model.gradient().gradient());
        if(epsilon)
            System.out.println(this.model.output(input));
        return null ;
    }

    @Override
    public Approximator clone() {
        return this.clone(false);
    }

    @Override
    public Approximator clone(boolean listener) {
        Mlp m = new Mlp(this,listener);
        return m ;
    }

    @Override
    public void stop() {
        System.out.println(this.model.params());
    }

    protected void attachListener(EpsilonMultiLayerNetwork mlp){
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //Then add the StatsListener to collect this information from the network, as it trains
        mlp.setListeners(new StatsListener(statsStorage));
    }

    public EpsilonMultiLayerNetwork getModel() {
        return this.model;
    }

    public void setModel(EpsilonMultiLayerNetwork model) {
        this.model = model;
    }

    public int getOutput() {
        return output;
    }

    public void setOutput(int output) {
        this.output = output;
    }
}
