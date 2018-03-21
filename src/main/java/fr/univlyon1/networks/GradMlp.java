package fr.univlyon1.networks;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

public class GradMlp extends Mlp {
    private MultiLayerNetwork gradPolicyMlp ;
    public GradMlp(GradMlp mlp,boolean b) {
        super(mlp,b);
        this.gradPolicyMlp = mlp.getGradPolicyMlp().clone();
    }

    public GradMlp(int input, int output, long seed, boolean listener, Double learning_rate, int numLayers, int numNodes,boolean minimize,boolean epsilon) {
        super(input, output, seed, listener, learning_rate, numLayers, numNodes,minimize,epsilon);
        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                //.learningRate(learning_rate)
                .biasInit(0)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.RMSPROP)
                .minimize(false)
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
        this.multiLayerConfiguration = builder.layer(numLayers+1,
                new DenseLayer.Builder()
                        .nIn(numNodes)
                        .nOut(output)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(numLayers+2, new LossLayer.Builder(new LossIdentity())
                        .build())
                .backprop(true).pretrain(false)
                .build();
        this.gradPolicyMlp = new MultiLayerNetwork(this.multiLayerConfiguration);
        if(listener)
            this.attachListener(this.model);
        this.gradPolicyMlp.init();
    }

    /**
     * Super refresh the weights, and this learn calculate error input in order to maximize Q
     * @param input
     * @param labels
     * @param number
     * @return
     */
    @Override
    public Object learn(INDArray input, INDArray labels, int number) {
        super.learn(input,labels,number);
        this.gradPolicyMlp.clear();
        this.gradPolicyMlp.setParameters(this.model.params()) ;
        //System.out.println(this.gradPolicyMlp.params());
        this.gradPolicyMlp.setInput(input);
        this.gradPolicyMlp.setLabels(Nd4j.create(new double[]{0}));
        this.gradPolicyMlp.computeGradientAndScore();

        /*System.out.println(this.gradPolicyMlp.score());
        System.out.println(this.gradPolicyMlp.epsilon());*/
        return this.gradPolicyMlp.epsilon() ;
        //Pair<Gradient, INDArray> res = this.model.backpropGradient(score);
        //this.model.pwet();
        /*System.out.println("here");
        System.out.println(res.getFirst());
        System.out.println(res.getSecond());
        return res ;*/
        /*for(int i = this.model.getnLayers()-1 ; i >= 0 ; i++){
            if( i == this.model.getnLayers( )-1) {
                this.model.p
                Layer l= this.model.getLayer(i);
            }

        }*/
        //this.model.calc
        //System.out.println(this.model.score());
        //System.out.println(LossFunctions.LossFunction.MSE.this.model.getOutputLayer().activate()));
        //System.out.println(this.model.getOutputLayer().params());
        //System.out.println(this.model.getLayer(1).activate());
        //System.out.println(this.model.getOutputLayer().activate());
        //LossMSE loss = new LossMSE();
        //double score = loss.computeScore(labels,this.model.getOutputLayer().activate(), new ActivationIdentity(),null,true);
        //double score = LossFunctions.score(this.model.getOutputLayer().activate(), LossFunctions.LossFunction.MSE,labels,0,0,false);
        //System.out.println(score);
    }

    public MultiLayerNetwork getGradPolicyMlp() {
        return gradPolicyMlp;
    }

    public void setGradPolicyMlp(MultiLayerNetwork gradPolicyMlp) {
        this.gradPolicyMlp = gradPolicyMlp;
    }


    public Approximator clone(boolean b){
        return new GradMlp(this,b);
    }
}
