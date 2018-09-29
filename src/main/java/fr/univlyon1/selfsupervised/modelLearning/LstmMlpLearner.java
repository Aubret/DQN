package fr.univlyon1.selfsupervised.modelLearning;

import fr.univlyon1.configurations.Configuration;
import fr.univlyon1.configurations.SupervisedConfiguration;
import fr.univlyon1.environment.space.ActionSpace;
import fr.univlyon1.environment.space.ObservationSpace;
import fr.univlyon1.memory.ExperienceReplay;
import fr.univlyon1.memory.ObservationsReplay.SpecificObservationReplay;
import fr.univlyon1.memory.SequentialExperienceReplay;
import fr.univlyon1.networks.Approximator;
import fr.univlyon1.networks.LSTM2D;
import fr.univlyon1.networks.Mlp;
import fr.univlyon1.networks.NormalizedMlp;
import fr.univlyon1.networks.lossFunctions.LossError;
import fr.univlyon1.networks.lossFunctions.LossMseSaveScore;
import fr.univlyon1.selfsupervised.dataConstructors.LstmDataConstructors;
import fr.univlyon1.selfsupervised.dataTransfer.ModelBasedData;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;

public class LstmMlpLearner<A> implements Learner {
    protected LSTM2D lstm;
    protected Mlp regression ;
    protected SupervisedConfiguration conf;
    protected LstmDataConstructors<A> dataConstructors ;
    protected ActionSpace<A> actionSpace ;
    protected long seed ;


    public LstmMlpLearner(Approximator lstm, SupervisedConfiguration supervisedConfiguration, ExperienceReplay<A> timeEp, ExperienceReplay<A> labelEp, Configuration configuration, ActionSpace<A> actionSpace, ObservationSpace observationSpace, long seed ){
        assert(lstm instanceof LSTM2D);
        assert(timeEp instanceof SequentialExperienceReplay);
        assert(labelEp instanceof SpecificObservationReplay);
        this.lstm =(LSTM2D)lstm;
        this.conf = supervisedConfiguration ;
        this.actionSpace = actionSpace ;
        this.dataConstructors = new LstmDataConstructors<A>((SequentialExperienceReplay<A>)timeEp,(SpecificObservationReplay<A>)labelEp,configuration,actionSpace,observationSpace,supervisedConfiguration);
        this.seed=seed ;
        this.initRegression();
    }

    public void learn(){
        ModelBasedData mbd = this.dataConstructors.construct();
        INDArray result =this.lstm.forwardLearn(mbd.getInputs(), Nd4j.zeros(1,this.lstm.numOutput()),conf.getBatchSize(),mbd.getMask(),mbd.getMaskLabel());
        INDArray epsilon = (INDArray)this.regression.learn(Nd4j.concat(1,result,mbd.getAddings()),mbd.getLabels(),conf.getBatchSize());
        INDArray epsilonObservation = epsilon.get(NDArrayIndex.all(), NDArrayIndex.interval(0, this.lstm.numOutput()));
        this.lstm.learn(mbd.getInputs(),epsilonObservation,conf.getBatchSize());

    }

    @Override
    public void stop() {
        System.out.println("regression stop");
        this.regression.stop();
    }

    public void initRegression(){
        //this.policyApproximator =new Mlp(conf.getNumLstmOutputNodes()+observationSpace.getShape()[0],this.actionSpace.getSize(),this.seed);
        this.regression =new Mlp(lstm.getOutput()+this.dataConstructors.numAddings(),this.actionSpace.getSize(),this.seed);
        this.regression.setLearning_rate(conf.getLearning_rate());
        this.regression.setNumNodes(conf.getNumHiddenNodes());
        this.regression.setNumLayers(conf.getNumLayers());
        //this.policyApproximator.setEpsilon(true); // On retropropage le gradient avec epsilon et non une foncitno de perte
        this.regression.setLossFunction(new LossMseSaveScore());
        this.regression.setEpsilon(false);
        this.regression.setMinimize(true); // On souhaite minimiser le gradient
        this.regression.setListener(true);
        this.regression.setUpdater(new Adam(conf.getLearning_rate()));
        this.regression.setLastActivation(Activation.IDENTITY);
        this.regression.setHiddenActivation(Activation.ELU);
        this.regression.setNumNodesPerLayer(conf.getLayersHiddenNodes());
        //this.policyApproximator.setL2(0.0002);
        /*((NormalizedMlp)this.policyApproximator).setLayerNormalization(true);
        ((NormalizedMlp)this.policyApproximator).setBatchNormalization(false);*/

        this.regression.init() ; // A la fin
    }
}
