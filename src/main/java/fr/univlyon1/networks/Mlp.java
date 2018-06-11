package fr.univlyon1.networks;

import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import fr.univlyon1.networks.lossFunctions.SaveScore;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
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
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;

@Getter
@Setter
public class Mlp implements Approximator{
    protected EpsilonMultiLayerNetwork model ;
    protected INDArray tmp ;
    protected int iterations = 0;
    protected MultiLayerConfiguration multiLayerConfiguration ;
    protected boolean minimize ; // Minimize or maximize the loss function
    protected boolean epsilon ; // use a loss function with label or direct errors ?

    protected int input ;
    protected int output ;
    protected long seed ;
    protected boolean listener ;
    protected Double learning_rate;
    protected int numLayers;
    protected ArrayList<Integer> numNodesPerLayer ;
    protected int numNodes ;
    protected Activation hiddenActivation ;
    protected Activation lastActivation ;
    protected ILossFunction lossFunction ;
    protected IUpdater updater;

    protected INDArray gradient;
    protected boolean batchNormalization;
    protected boolean finalBatchNormalization ;

    protected boolean dropout ;
    protected Double score ;
    protected INDArray values ;
    protected INDArray scoreArray ;

    protected Double l2 ;
    protected int epoch ;

    public Mlp(Mlp mlp,boolean listener){// MultiLayerNetwork model,int output){
        this.input = mlp.getInput();
        //this.model = mlp.getModel().clone();
        this.output = mlp.getOutput() ;
        this.input = mlp.getInput();
        this.tmp = mlp.tmp ;
        this.minimize = mlp.isMinimize() ;
        this.epsilon = mlp.isEpsilon();
        this.learning_rate = mlp.getLearning_rate();
        this.numNodes = mlp.getNumNodes();
        this.updater = mlp.getUpdater() ;
        this.hiddenActivation = mlp.getHiddenActivation() ;
        this.numLayers = mlp.getNumLayers() ;
        this.lastActivation = mlp.getLastActivation() ;
        this.lossFunction = mlp.getLossFunction() ;
        this.seed = mlp.getSeed() ;
        this.iterations = mlp.getIterations() ;
        this.batchNormalization = mlp.isBatchNormalization();
        this.finalBatchNormalization = mlp.isFinalBatchNormalization();
        this.numNodesPerLayer = mlp.getNumNodesPerLayer() ;
        this.dropout = mlp.isDropout() ;
        this.l2 = mlp.getL2();
        this.listener = listener ;
        this.epoch = mlp.getEpoch() ;
        this.init();
        this.setParams(mlp.getParams());
        /*if(listener)
            this.attachListener(this.model);*/
    }

    /**
     * Ici on met toutes les valeurs par défauts, on peut les changer avec des setters (Lombok plugin)
     * @param input
     * @param output
     * @param seed
     */
    public Mlp(int input, int output, long seed) {
        this.input = input ;
        this.updater =null ;
        this.output = output ;
        this.seed = seed ;
        this.learning_rate = 0.001 ;
        this.numLayers = 2 ;
        this.numNodes = 10 ;
        this.hiddenActivation = Activation.RELU ;
        this.lastActivation = Activation.IDENTITY ;
        this.listener = false ;
        //this.lossFunction = LossFunctions.LossFunction.MSE.getILossFunction() ;
        this.lossFunction = new LossMseSaveScore() ;
        this.batchNormalization = false ;
        this.finalBatchNormalization = false ;
        this.minimize = true ;
        this.epsilon = false ;
        this.numNodesPerLayer = new ArrayList<>();
        this.dropout = false ;
        this.l2 = null ;
        this.epoch = 0 ;
    }


    public void init() {
        int cursor = 0 ;
        if(this.updater ==null){
            this.updater = new Sgd(this.learning_rate);
        }

        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .seed(seed+1)
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(this.updater)
                //.learningRate(learning_rate)
                .biasInit(0.01)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .weightInit(WeightInit.XAVIER)
                .minimize(minimize);
                //
        if(l2 != null) {
            b.l2(this.l2);
        }
        NeuralNetConfiguration.ListBuilder builder = b.list() ;


        //
                //.list();

        /*
        Layer layerActiv = new ActivationLayer.Builder()
                .activation(Activation.IDENTITY)
                .build();
        layerActiv.setNIn(InputType.feedForward(input),true);
        builder.layer(cursor, layerActiv);
        cursor++ ;
        if(this.firstBatchNormalization){
            Layer l = new BatchNormalization.Builder().nIn(input).nOut(input).build();
            builder.layer(cursor, l);
            cursor++ ;
        }*/
        /*Layer lay = new BatchNormalization.Builder().nIn(input).nOut(input).build();
        builder.layer(cursor, lay);
        cursor++ ;*/

        int node = this.numNodesPerLayer.size() >0 ? this.numNodesPerLayer.get(0) : numNodes ;
        builder.layer(cursor, new DenseLayer.Builder()
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
        node = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        Layer.Builder l = new DenseLayer.Builder()
                .biasInit(0.1)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .nIn(node)
                .nOut(output)
                .activation(this.lastActivation)
                ;
        if(this.dropout){
            l.dropOut(0.5);
        }
        builder.layer(cursor, l.build());
        cursor++ ;
        if(!epsilon)
            builder.layer(cursor, new LossLayer.Builder(this.lossFunction)
                        .build());

        this.multiLayerConfiguration = builder
                .backprop(true).pretrain(false)
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        this.model.init();
        if(this.listener)
            this.attachListener(this.model);
        this.tmp = this.model.params().dup();
    }

    public INDArray getOneResult(INDArray data){
        //this.model.setInputMiniBatchSize(data.shape()[0]);
        INDArray res = this.model.output(data,false) ;
        return res ;
    }

    public INDArray getOneTrainingResult(INDArray data){
        //this.model.setInputMiniBatchSize(data.shape()[0]);
        //double[] datas =  new double[]{-0.74,  -0.91,  -0.76,  -0.91,  -0.68,  -0.84,  -0.74,  -0.91,  -0.74,  -0.91,  -0.68,  -0.84,  -0.78,  -0.91,  -0.68,  -0.91,  -0.68,  -0.84,  -0.43,  -0.74,  -0.90,  -0.74,  -0.91,  -0.70,  -0.84,  -0.76,  -0.91,  -0.74,  -0.91,  -0.70,  -0.84,  -0.76,  -0.94,  -0.76,  -0.94,  -0.68,  -0.83,  -0.44};
        INDArray res = this.model.output(data,false);
        return res;
    }

    @Override
    public int numOutput() {
        return this.output;
    }

    @Override
    public Object learn(INDArray input,INDArray labels,int number) {
        //System.out.println(this.model.getUpdater().getStateViewArray().getDouble(0));
        //this.model.clear();
        //System.out.println(this.model.getUpdater().getStateViewArray().getDouble(0));

        //this.model.setInputMiniBatchSize(number);
        //this.model.clear();
        this.model.setInput(input);
        this.model.setLabels(labels);
        if(this.epsilon) {
            this.model.backpropGradient(labels);
        }else {
            this.model.computeGradientAndScore();
        }
        //this.model.backpropGradient(Nd4j.create(new Double[]{this.model.getOutputLayer().activate()}))
        //System.out.println(grad.gradientForVariable());
        this.score = this.model.score() ;

        //System.out.println(this.model.acti);
        this.model.getUpdater().update(this.model, this.model.gradient(), iterations,this.epoch,number);
        //if(this.model.getOutputLayer() instanceof IOutputLayer)
         //   System.out.println(this.model.getOutputLayer().gradient().gradient() );
        if(this.minimize) {
            this.model.params().subi(this.model.gradient().gradient());
        }else {
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
                this.scoreArray = lossfn.getLastScoreArray().detach();
            }
        }
        //return this.model.getOutputLayer().com ;
        return this.model.epsilon() ;
    }

    @Override
    public INDArray error(INDArray input, INDArray labels, int number) {
        //System.out.println(this.gradPolicyMlp.params());
        //this.model.clear();
        this.model.setInputMiniBatchSize(number);
        this.model.setInput(input);
        this.model.setLabels(labels);//Nd4j.create(new double[]{0}));
        this.model.computeGradientAndScore();
        this.score = this.model.score();
        iterations++ ;
        for(IterationListener it : this.model.getListeners()){
            if(it instanceof TrainingListener){
                ((TrainingListener)it).onGradientCalculation(this.model);
            }
            it.iterationDone(this.model, iterations,this.epoch );
        }
        return this.model.epsilon().detach() ;
    }

    @Override
    public void epoch() {
        this.epoch++ ;
    }

    @Override
    public INDArray getParams() {
        return this.model.params();
    }

    @Override
    public void setParams(INDArray params) {
        if(params != this.model.params()) {
            this.model.setParams(params);
        }
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
        System.out.println(this.tmp);
        System.out.println(this.model.params());
    }

    @Override
    public void clear() {

    }

    protected void attachListener(MultiLayerNetwork mlp){
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);


        //Then add the StatsListener to collect this information from the network, as it trains
        mlp.setListeners(new StatsListener(statsStorage));
    }

    public MultiLayerNetwork getModel() {
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

    @Override
    public Object getAction(INDArray inputs) {
        return this.getOneTrainingResult(inputs);
    }

    public INDArray getValues(){
        return this.values ;
    }

    public INDArray getScoreArray(){
        return this.scoreArray ;
    }

}
