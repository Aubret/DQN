package fr.univlyon1.networks;

import fr.univlyon1.environment.states.AllHiddenState;
import fr.univlyon1.networks.layers.LSTMLayer;
import fr.univlyon1.networks.layers.LSTMLayerConf;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.RnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.conditions.GreaterThan;
import org.nd4j.linalg.learning.config.Sgd;

import java.io.IOException;
import java.util.*;

@Slf4j
/**
 * LSTM neural network which take as input 3D data, batch, sequence, inputvector.
 */
public class LSTM2D extends LSTM {


    protected INDArray indices ;

    public LSTM2D(LSTM2D lstm, boolean listener) {
        super(lstm, listener);
    }

    public LSTM2D(int input, int output, long seed){
        super(input,output,seed);
    }

    public void init(){
        int cursor = 0 ;
        if(this.updater ==null){
            this.updater = new Sgd(this.learning_rate);
        }
        NeuralNetConfiguration.Builder b = new NeuralNetConfiguration.Builder()
                .seed(this.seed+1)
                //.cacheMode(CacheMode.DEVICE)
                //.l2(0.001)Mlp
                .biasInit(0.1)
                .weightInit(WeightInit.XAVIER)
                .updater(this.updater);// It's Adam
        if(l2 != null) {
            b.l2(this.l2);
        }
        NeuralNetConfiguration.ListBuilder builder = b.list() ;

        //-------------------------------------- Initialisation des couches------------------
        /*Layer lay = new BatchNormalization.Builder().nIn(input).nOut(input).build();
        builder.layer(cursor, lay);
        cursor++;*/

        int node = this.numNodesPerLayer.size() >0 ? this.numNodesPerLayer.get(0) : numNodes ;
        //node = cursor+1 == this.numLayers ? output : node ;
        builder.layer(cursor, new org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                //.activation(this.hiddenActivation)
                .activation(Activation.TANH)
                //.gateActivationFunction(Activation.SIGMOID)
                .forgetGateBiasInit(0.1)
                //.weightInit(WeightInit.XAVIER_UNIFORM)
                .nIn(input).nOut(node)
                .build()
        );

        cursor++;
        for (int i = 1; i < numLayers; i++){ // 2 Two layers only
            int previousNode = this.numNodesPerLayer.size() > i-1 ? this.numNodesPerLayer.get(i-1) : numNodes ;
            node = this.numNodesPerLayer.size() > i ? this.numNodesPerLayer.get(i) : numNodes ;
            //node = cursor+1 == this.numLayers ? output : node ;
            //if(i == numLayers -1)
            builder.layer(cursor, new org.deeplearning4j.nn.conf.layers.LSTM.Builder()//new  org.deeplearning4j.nn.conf.layers.LSTM.Builder()
                    .activation(this.hiddenActivation)
                    //.units(node)
                    .gateActivationFunction(Activation.SIGMOID)
                    .forgetGateBiasInit(1.)
                    //.weightInit(WeightInit.XAVIER_UNIFORM)
                    .nIn(previousNode).nOut(node)
                    .build()
            );
            cursor++ ;
        }
        /*node = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor,
                new RnnOutputLayer.Builder()
                        .lossFunction(this.lossFunction)
                        .nIn(node)
                        .nOut(output)
                        .activation(this.lastActivation)
                        .build());*/
        builder.inputPreProcessor(cursor,new RnnToFeedForwardPreProcessor());

        //builder.layer(cursor, new BatchNormalization.Builder().build());
        //cursor++ ;


        //---
        /*int previousNode = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor, new DenseLayer.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .nIn(previousNode).nOut(output)
                .build()
        );
        cursor++;*/
        /*int previousNode = this.numNodesPerLayer.size() > numLayers-1 ? this.numNodesPerLayer.get(numLayers-1) : numNodes ;
        builder.layer(cursor, new DenseLayer.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .nIn(previousNode).nOut(output)
                .build()
        );
        cursor++;
        builder.layer(cursor, new LayerNormalizationConf.Builder().build());
        cursor++ ;
        builder.layer(cursor, new ActivationLayer.Builder()
                .activation(Activation.TANH)
                .build()
        );
        cursor++;*/
        //---
        builder.layer(cursor, new LossLayer.Builder().lossFunction(this.lossFunction).build());

        this.multiLayerConfiguration = builder
                .backpropType(BackpropType.Standard)
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
        if(data.rank()==2)
            this.model.clearLayerMaskArrays();
        INDArray res = this.model.rnnTimeStep(data);
        if(data.rank() ==2)
            return res ;
        return crop3dData(res,this.maskLabel);
    }


    public INDArray getOneTrainingResult(INDArray data){
        //this.model.rnnClearPreviousState();
        for(int i = 0 ; i < this.model.getnLayers()-1 ; i++) {
            if(this.model.getLayer(i) instanceof org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer)
                ((org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer) this.model.getLayer(i)).rnnSetTBPTTState(new HashMap<>());
        }
        List<INDArray> workspace = this.model.rnnActivateUsingStoredState(data, true, true);
        INDArray last = workspace.get(workspace.size()-1); // Dernière couche
        return crop3dData(last,this.maskLabel);
        //return last.getRow(last.size(0)-1);
    }



    @Override
    public INDArray forwardLearn(INDArray input, INDArray labels, int number, INDArray mask, INDArray maskLabel){
        this.model.rnnClearPreviousState();
        /*for(int i = 0 ; i < this.model.getnLayers()-1 ; i++) { // On nettoie d'abord l'état de sa mémoire
            ((org.deeplearning4j.nn.layers.recurrent.GravesLSTM) this.model.getLayer(i)).rnnSetTBPTTState(new HashMap<>());
        }*/
        this.mask=mask;
        this.maskLabel = maskLabel ;
        this.model.setInputMiniBatchSize(number);
        this.model.setInput(input);
        this.model.setLayerMaskArrays(mask,maskLabel);

        LayerHelper h = this.model.getLayer(0).getHelper();
        List<INDArray> workspace = this.model.rnnActivateUsingStoredState(input, true, true);
        for(TrainingListener it : this.model.getListeners()){
            it.onForwardPass(this.model, workspace);
        }
        INDArray last = workspace.get(workspace.size()-1); // Dernière couche

        return crop3dData(last,maskLabel);
    }


    public INDArray crop3dData(INDArray data,INDArray maskLabel){
        INDArray linspace = Nd4j.linspace(1,data.shape()[0],data.shape()[0]).transpose();
        INDArray indicesAndZeros = maskLabel.mul(linspace);//.reshape(data.shape()[0]);
        this.indices = BooleanIndexing.chooseFrom(new INDArray[]{indicesAndZeros},Arrays.asList(0.0), Collections.emptyList(),new GreaterThan()).addi(-1);
        INDArray newData = data.get(this.indices);
        newData = newData.reshape(newData.shape()[0],newData.shape()[2]);
        return newData;
    }

    @Override
    public Object learn(INDArray input,INDArray labels,int number) {
        return super.learn(input,this.uncrop2dData(labels,number),number);
    }

    public INDArray uncrop2dData(INDArray labels,int number){
        INDArray newLabels = Nd4j.zeros(number,this.output);
        //System.out.println(labels);
        for(int i = 0 ; i < this.indices.size(0); i++){
            //INDArrayIndex[] ndi = new INDArrayIndex[]{NDArrayIndex.point(i)};
            //System.out.println(this.indices.size(0));
            //System.out.println(i);
            INDArray lab = labels.get(NDArrayIndex.point(i),NDArrayIndex.all()) ;
            Double ind = this.indices.getDouble(i);
            newLabels.put(new INDArrayIndex[]{NDArrayIndex.point(ind.intValue()),NDArrayIndex.all()},lab);
        }
        return newLabels ;

    }

    public Object allMemory(){
        ArrayList<INDArray[]> memories = new ArrayList<>();
        ArrayList<INDArray[]> prevAct = new ArrayList<>();
        for(int i=0; i < this.model.getnLayers() ; i++){
            //memories.add(this.model.rnnGetPreviousState(i));
            if(this.model.getLayer(i) instanceof LSTMLayer) {
                LSTMLayer recurrent = (LSTMLayer) this.model.getLayer(i);
                memories.add(recurrent.getSaved().memCellState);
                prevAct.add(recurrent.getSaved().fwdPassOutputAsArrays);
            }

        }
        return new AllHiddenState(memories,prevAct) ;
    }

    @Override
    public StateApproximator clone(boolean listener) {
        LSTM2D m = new LSTM2D(this,listener);
        m.setHiddenActivation(Activation.TANH);
        m.init();
        m.setParams(this.getParams().dup());
        return m ;
    }
}
