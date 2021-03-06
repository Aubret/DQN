package fr.univlyon1.networks;

import fr.univlyon1.networks.layers.LayerNormalizationConf;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.Sgd;

@Getter
@Setter
/**
 * Mlp which enable normalizing layer like layer normalization or batch normalization
 */
public class NormalizedMlp extends Mlp {

    protected boolean batchNormalization ;
    protected boolean layerNormalization ;


    public NormalizedMlp(NormalizedMlp mlp, boolean listener) {
        super(mlp, listener);
    }

    public NormalizedMlp(int input, int output, long seed){
        super(input,output,seed);
        this.batchNormalization = false ;
        this.layerNormalization = true ;
    }

    // training arg is important because it decides wether ornot we use global variables mean and variance
    public INDArray getOneResult(INDArray data){
        //this.model.setInputMiniBatchSize(data.shape()[0]);
        INDArray res = this.model.output(data,true) ;
        return res ;
    }

    public INDArray getOneTrainingResult(INDArray data){
        //this.model.setInputMiniBatchSize(data.shape()[0]);
        //double[] datas =  new double[]{-0.74,  -0.91,  -0.76,  -0.91,  -0.68,  -0.84,  -0.74,  -0.91,  -0.74,  -0.91,  -0.68,  -0.84,  -0.78,  -0.91,  -0.68,  -0.91,  -0.68,  -0.84,  -0.43,  -0.74,  -0.90,  -0.74,  -0.91,  -0.70,  -0.84,  -0.76,  -0.91,  -0.74,  -0.91,  -0.70,  -0.84,  -0.76,  -0.94,  -0.76,  -0.94,  -0.68,  -0.83,  -0.44};
        INDArray res = this.model.output(data,true);
        return res;
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
        int node = this.numNodesPerLayer.size() >0 ? this.numNodesPerLayer.get(0) : numNodes ;
        builder.layer(cursor, new DenseLayer.Builder()
                .nIn(input).nOut(node)
                .activation(Activation.IDENTITY)
                .build()
        );
        cursor++ ;
        if(this.batchNormalization) {
            builder.layer(cursor, new BatchNormalization.Builder().build());
        }else{
            builder.layer(cursor, new LayerNormalizationConf.Builder().activation(Activation.IDENTITY).build());
        }
        cursor++ ;
        builder.layer(cursor, new ActivationLayer.Builder()
                .activation(this.hiddenActivation)
                .build()
        );
        cursor++;

        for (int i = 1; i < numLayers; i++){
            int previousNode = this.numNodesPerLayer.size() > i-1 ? this.numNodesPerLayer.get(i-1) : numNodes ;
            node = this.numNodesPerLayer.size() > i ? this.numNodesPerLayer.get(i) : numNodes ;
            builder.layer(cursor, new DenseLayer.Builder()
                    .activation(Activation.IDENTITY)
                    .nIn(previousNode).nOut(node)
                    .build()
            );
            cursor++ ;
            if(this.batchNormalization) {
                builder.layer(cursor, new BatchNormalization.Builder().build());
            }else{
                builder.layer(cursor, new LayerNormalizationConf.Builder().activation(Activation.IDENTITY).build());
            }
            cursor++ ;
            builder.layer(cursor, new ActivationLayer.Builder()
                    .activation(this.hiddenActivation)
                    .build()
            );
            cursor++;
        }

        node = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        Layer.Builder l = new DenseLayer.Builder()
                .biasInit(0.1)
                .weightInit(WeightInit.XAVIER_UNIFORM)
                .nIn(node)
                .nOut(output)
                .activation(this.lastActivation)
                ;

        builder.layer(cursor, l.build());
        cursor++ ;
        if(!epsilon)
            builder.layer(cursor, new LossLayer.Builder(this.lossFunction)
                    .build());

        this.multiLayerConfiguration = builder
                .build();

        this.model = new EpsilonMultiLayerNetwork(this.multiLayerConfiguration);
        this.model.init();
        if(this.listener)
            this.attachListener(this.model);
        this.tmp = this.model.params().dup();
    }

    @Override
    public Approximator clone(boolean listener) {
        NormalizedMlp m = new NormalizedMlp(this,listener);
        m.setBatchNormalization(this.batchNormalization);
        m.setLayerNormalization(this.layerNormalization);
        m.init();
        m.setParams(this.getParams());
        return m ;
    }
}
