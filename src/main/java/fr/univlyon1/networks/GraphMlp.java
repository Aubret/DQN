package fr.univlyon1.networks;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;

public class GraphMlp extends Mlp{

    ComputationGraph graph ;

    public GraphMlp(Mlp mlp, boolean listener) {
        super(mlp, listener);
    }



    public void init(){
        int cursor = 0 ;
        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .seed(seed+1)
                //.learningRate(learning_rate)
                .biasInit(0.1)
                //.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .weightInit(WeightInit.XAVIER)
                //.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //.updater(this.updater)
                .minimize(minimize);

                //.graphBuilder();
        //
        if(l2 != null)
            b.l2(this.l2);

        ComputationGraphConfiguration.GraphBuilder graphBuilder = b.graphBuilder();
        int node = this.numNodesPerLayer.size() >0 ? this.numNodesPerLayer.get(0) : numNodes ;
        graphBuilder.addLayer("L"+cursor,new DenseLayer.Builder()
                .activation(this.hiddenActivation)
                .nIn(input).nOut(node)
                .build()
        );
        cursor++ ;
        for (int i = 1; i < numLayers; i++){
            int previousNode = this.numNodesPerLayer.size() > i-1 ? this.numNodesPerLayer.get(i-1) : numNodes ;
            node = this.numNodesPerLayer.size() > i ? this.numNodesPerLayer.get(i) : numNodes ;
            graphBuilder.addLayer("L"+cursor, new DenseLayer.Builder()
                    .activation(this.hiddenActivation)
                    .nIn(previousNode).nOut(node)
                    .build()
            );
            cursor++ ;
        }

        node = this.numNodesPerLayer.size() == numLayers ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        Layer.Builder l = new DenseLayer.Builder()
                .biasInit(0.01)
                .weightInit(WeightInit.UNIFORM)
                .nIn(node)
                .nOut(output)
                .activation(this.lastActivation)
                ;
        if(this.dropout){
            l.dropOut(0.5);
        }
        graphBuilder.addLayer("L"+cursor, l.build());
        cursor++ ;
        if(!epsilon)
            graphBuilder.addLayer("L"+cursor, new LossLayer.Builder(this.lossFunction)
                    .build());

        ComputationGraphConfiguration compuGraph = graphBuilder.build();

        this.graph = new ComputationGraph(compuGraph);//compuGraph ;new EpsilonMultiLayerNetwork(compuGraph    );
        if(this.listener)
            this.attachListener(this.model);
        this.graph.init();
        //this.tmp = this.graph.params().dup();

    }
}

